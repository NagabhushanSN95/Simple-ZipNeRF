import logging
import os
import sys
import time
from pathlib import Path

import accelerate
from absl import app
import gin
from tqdm import tqdm

from internal import configs
from internal import datasets
from internal import image
from internal import models
from internal import raw_utils
from internal import ref_utils
from internal import train_utils
from internal import checkpoints
from internal import utils
from internal import vis
import numpy as np
import torch
import tensorboardX
from torch.utils._pytree import tree_map

configs.define_common_flags()


def summarize_results(folder, scene_names, num_buckets):
    metric_names = ['psnrs', 'ssims', 'lpips']
    num_iters = 1000000
    precisions = [3, 4, 4, 4]

    results = []
    for scene_name in scene_names:
        test_preds_folder = os.path.join(folder, scene_name, 'test_preds')
        values = []
        for metric_name in metric_names:
            filename = os.path.join(folder, scene_name, 'test_preds', f'{metric_name}_{num_iters}.txt')
            with utils.open_file(filename) as f:
                v = np.array([float(s) for s in f.readline().split(' ')])
                values.append(np.mean(np.reshape(v, [-1, num_buckets]), 0))
        results.append(np.concatenate(values))
    avg_results = np.mean(np.array(results), 0)

    psnr, ssim, lpips = np.mean(np.reshape(avg_results, [-1, num_buckets]), 1)

    mse = np.exp(-0.1 * np.log(10.) * psnr)
    dssim = np.sqrt(1 - ssim)
    avg_avg = np.exp(np.mean(np.log(np.array([mse, dssim, lpips]))))

    s = []
    for i, v in enumerate(np.reshape(avg_results, [-1, num_buckets])):
        s.append(' '.join([f'{s:0.{precisions[i]}f}' for s in v]))
    s.append(f'{avg_avg:0.{precisions[-1]}f}')
    return ' | '.join(s)


def main(unused_argv):
    config = configs.load_config()
    config.exp_path = Path(f'../runs/testing/test{config.test_num:04}/{config.scene_name}')
    config.logs_dir = config.exp_path / 'logs'
    config.checkpoint_dir = Path(f'../runs/training/train{config.train_num:04}/{config.scene_name}/saved_models')
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    accelerator = accelerate.Accelerator()

    # setup logger
    logging.basicConfig(
        format="%(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler((config.logs_dir / 'log_eval.txt').as_posix())],
        level=logging.INFO,
    )
    sys.excepthook = utils.handle_exception
    logger = accelerate.logging.get_logger(__name__)
    logger.info(config)
    logger.info(accelerator.state, main_process_only=False)

    config.world_size = accelerator.num_processes
    config.global_rank = accelerator.process_index
    accelerate.utils.set_seed(config.seed, device_specific=True)
    model = models.Model(config=config, model_name='main')
    model.eval()
    model.to(accelerator.device)

    # Render both test and train frames
    test_dataset = datasets.load_dataset('test', config.data_dir, config)
    train_dataset = datasets.load_dataset('train', config.data_dir, config)
    train_dataset._next_fn = train_dataset._next_test
    dataset = torch.utils.data.ConcatDataset([test_dataset, train_dataset])
    dataset_size = test_dataset.size + train_dataset.size
    dataloader = torch.utils.data.DataLoader(np.arange(dataset_size),
                                             shuffle=False,
                                             batch_size=1,
                                             collate_fn= lambda item: test_dataset.collate_fn(item) if item[0] < test_dataset.size else train_dataset.collate_fn([item[0] - test_dataset.size]),
                                             )
    tb_process_fn = lambda x: x.transpose(2, 0, 1) if len(x.shape) == 3 else x[None]
    if config.rawnerf_mode:
        postprocess_fn = dataset.metadata['postprocess_fn']
    else:
        postprocess_fn = lambda z: z

    if config.eval_raw_affine_cc:
        cc_fun = raw_utils.match_images_affine
    else:
        cc_fun = image.color_correct

    model = accelerator.prepare(model)

    metric_harness = image.MetricHarness()

    last_step = 0
    out_dir = os.path.join(config.exp_path,
                           'path_renders' if config.render_path else 'predicted_frames')  # TODO SNB: change path_renders appropriately
    path_fn = lambda dirname, frame_num, extension: config.exp_path / f'{dirname}/{frame_num:04}.{extension}'

    if not config.eval_only_once:
        summary_writer = tensorboardX.SummaryWriter(
            os.path.join(config.exp_path, 'eval'))
    while True:
        step = checkpoints.restore_checkpoint(config.checkpoint_dir, accelerator, logger)
        if step <= last_step:
            logger.info(f'Checkpoint step {step} <= last step {last_step}, sleeping.')
            time.sleep(10)
            continue
        logger.info(f'Evaluating checkpoint at step {step}.')
        if config.eval_save_output and (not utils.isdir(out_dir)):
            utils.makedirs(out_dir)

        num_eval = min(dataset_size, config.eval_dataset_limit)
        perm = np.random.permutation(num_eval)
        showcase_indices = np.sort(perm[:config.num_showcase_images])
        metrics = []
        metrics_cc = []
        showcases = []
        render_times = []
        for idx, batch in enumerate(tqdm(dataloader)):
            batch = accelerate.utils.send_to_device(batch, accelerator.device)
            frame_num = int(batch['frame_name'])
            del batch['frame_name']
            eval_start_time = time.time()
            if idx >= num_eval:
                logger.info(f'Skipping image {idx + 1}/{dataset_size}')
                continue
            # logger.info(f'Evaluating image {idx + 1}/{dataset_size}')
            rendering = models.render_image(model, accelerator,
                                            batch, False, 1, config)

            if not accelerator.is_main_process:  # Only record via host 0.
                continue

            render_times.append((time.time() - eval_start_time))
            logger.info(f'Rendered in {render_times[-1]:0.3f}s')

            cc_start_time = time.time()
            # rendering['rgb_cc'] = cc_fun(rendering['rgb'], batch['rgb'])
            rendering['rgb_cc'] = rendering['rgb']
            rendering = tree_map(lambda x: x.detach().cpu().numpy() if x is not None else None, rendering)
            # batch = tree_map(lambda x: x.detach().cpu().numpy() if x is not None else None, batch)
            # batch = tree_map(lambda x: x.detach().cpu().numpy() if x is not None and type(x) is not int else None, batch)
            for x in batch.keys():
                if batch[x] is not None:
                    if(type(batch[x]) is not int and type(batch[x]) is not dict):
                        batch[x] = batch[x].detach().cpu().numpy()
                else:
                    batch[x] = None
            batch = tree_map(lambda x: x if x is not None else None, batch)
            gt_rgb = batch['rgb']
            logger.info(f'Color corrected in {(time.time() - cc_start_time):0.3f}s')

            if not config.eval_only_once and idx in showcase_indices:
                showcase_idx = idx if config.deterministic_showcase else len(showcases)
                showcases.append((showcase_idx, rendering, batch))
            if not config.render_path:
                rgb = postprocess_fn(rendering['rgb'])
                rgb_cc = postprocess_fn(rendering['rgb_cc'])
                rgb_gt = postprocess_fn(gt_rgb)

                if config.eval_quantize_metrics:
                    # Ensures that the images written to disk reproduce the metrics.
                    rgb = np.round(rgb * 255) / 255
                    rgb_cc = np.round(rgb_cc * 255) / 255

                if config.eval_crop_borders > 0:
                    crop_fn = lambda x, c=config.eval_crop_borders: x[c:-c, c:-c]
                    rgb = crop_fn(rgb)
                    rgb_cc = crop_fn(rgb_cc)
                    rgb_gt = crop_fn(rgb_gt)

                metric = metric_harness(rgb, rgb_gt)
                metric_cc = metric_harness(rgb_cc, rgb_gt)

                if config.compute_disp_metrics:
                    for tag in ['mean', 'median']:
                        key = f'distance_{tag}'
                        if key in rendering:
                            disparity = 1 / (1 + rendering[key])
                            metric[f'disparity_{tag}_mse'] = float(
                                ((disparity - batch['disps']) ** 2).mean())

                if config.compute_normal_metrics:
                    weights = rendering['acc'] * batch['alphas']
                    normalized_normals_gt = ref_utils.l2_normalize_np(batch['normals'])
                    for key, val in rendering.items():
                        if key.startswith('normals') and val is not None:
                            normalized_normals = ref_utils.l2_normalize_np(val)
                            metric[key + '_mae'] = ref_utils.compute_weighted_mae_np(
                                weights, normalized_normals, normalized_normals_gt)

                for m, v in metric.items():
                    logger.info(f'{m:30s} = {v:.4f}')

                metrics.append(metric)
                metrics_cc.append(metric_cc)

            if config.eval_save_output and (config.eval_render_interval > 0):
                if (idx % config.eval_render_interval) == 0:
                    utils.save_img_u8(postprocess_fn(rendering['rgb']),
                                      path_fn('predicted_frames', frame_num, 'png'))
                    utils.save_img_u8(postprocess_fn(rendering['rgb_cc']),
                                      path_fn(f'predicted_frames_cc', frame_num, 'png'))
                    utils.save_depth(postprocess_fn(rendering['depth']),
                                     path_fn(f'predicted_depths', frame_num, 'npy'), as_png=True)

                    for key in ['distance_mean', 'distance_median']:
                        if key in rendering:
                            utils.save_img_f32(rendering[key],
                                               path_fn(f'predicted_{key}', frame_num, 'tiff'))

                    for key in ['normals']:
                        if key in rendering:
                            utils.save_img_u8(rendering[key] / 2. + 0.5,
                                              path_fn(f'predicted_{key}', frame_num, 'png'))

                    utils.save_img_f32(rendering['acc'], path_fn(f'predicted_acc', frame_num, 'tiff'))

        if (not config.eval_only_once) and accelerator.is_main_process:
            summary_writer.add_scalar('eval_median_render_time', np.median(render_times),
                                      step)
            for name in metrics[0]:
                scores = [m[name] for m in metrics]
                summary_writer.add_scalar('eval_metrics/' + name, np.mean(scores), step)
                summary_writer.add_histogram('eval_metrics/' + 'perimage_' + name, scores,
                                             step)
            for name in metrics_cc[0]:
                scores = [m[name] for m in metrics_cc]
                summary_writer.add_scalar('eval_metrics_cc/' + name, np.mean(scores), step)
                summary_writer.add_histogram('eval_metrics_cc/' + 'perimage_' + name,
                                             scores, step)

            for i, r, b in showcases:
                if config.vis_decimate > 1:
                    d = config.vis_decimate
                    decimate_fn = lambda x, d=d: None if x is None else x[::d, ::d]
                else:
                    decimate_fn = lambda x: x
                r = tree_map(decimate_fn, r)
                b = tree_map(decimate_fn, b)
                visualizations = vis.visualize_suite(r, b)
                for k, v in visualizations.items():
                    if k == 'color':
                        v = postprocess_fn(v)
                    summary_writer.add_image(f'output_{k}_{i}', tb_process_fn(v), step)
                if not config.render_path:
                    target = postprocess_fn(b['rgb'])
                    summary_writer.add_image(f'true_color_{i}', tb_process_fn(target), step)
                    pred = postprocess_fn(visualizations['color'])
                    residual = np.clip(pred - target + 0.5, 0, 1)
                    summary_writer.add_image(f'true_residual_{i}', tb_process_fn(residual), step)
                    if config.compute_normal_metrics:
                        summary_writer.add_image(f'true_normals_{i}', tb_process_fn(b['normals']) / 2. + 0.5,
                                                 step)

        # SNB: No need to save metrics. We will compute them separately
        # if (config.eval_save_output and (not config.render_path) and
        #         accelerator.is_main_process):
        #     with utils.open_file(path_fn(f'render_times_{step}.txt'), 'w') as f:
        #         f.write(' '.join([str(r) for r in render_times]))
        #     logger.info(f'metrics:')
        #     results = {}
        #     num_buckets = config.multiscale_levels if config.multiscale else 1
        #     for name in metrics[0]:
        #         with utils.open_file(path_fn(f'metric_{name}_{step}.txt'), 'w') as f:
        #             ms = [m[name] for m in metrics]
        #             f.write(' '.join([str(m) for m in ms]))
        #             results[name] = ' | '.join(
        #                 list(map(str, np.mean(np.array(ms).reshape([-1, num_buckets]), 0).tolist())))
        #     with utils.open_file(path_fn(f'metric_avg_{step}.txt'), 'w') as f:
        #         for name in metrics[0]:
        #             f.write(f'{name}: {results[name]}\n')
        #             logger.info(f'{name}: {results[name]}')
        #     logger.info(f'metrics_cc:')
        #     results_cc = {}
        #     for name in metrics_cc[0]:
        #         with utils.open_file(path_fn(f'metric_cc_{name}_{step}.txt'), 'w') as f:
        #             ms = [m[name] for m in metrics_cc]
        #             f.write(' '.join([str(m) for m in ms]))
        #             results_cc[name] = ' | '.join(
        #                 list(map(str, np.mean(np.array(ms).reshape([-1, num_buckets]), 0).tolist())))
        #     with utils.open_file(path_fn(f'metric_cc_avg_{step}.txt'), 'w') as f:
        #         for name in metrics[0]:
        #             f.write(f'{name}: {results_cc[name]}\n')
        #             logger.info(f'{name}: {results_cc[name]}')
        #     if config.eval_save_ray_data:
        #         for i, r, b in showcases:
        #             rays = {k: v for k, v in r.items() if 'ray_' in k}
        #             np.set_printoptions(threshold=sys.maxsize)
        #             with utils.open_file(path_fn(f'ray_data_{step}_{i}.txt'), 'w') as f:
        #                 f.write(repr(rays))

        if config.eval_only_once:
            break
        if config.early_exit_steps is not None:
            num_steps = config.early_exit_steps
        else:
            num_steps = config.max_steps
        if int(step) >= num_steps:
            break
        last_step = step
    logger.info('Finish evaluation.')


if __name__ == '__main__':
    with gin.config_scope('eval'):
        app.run(main)
