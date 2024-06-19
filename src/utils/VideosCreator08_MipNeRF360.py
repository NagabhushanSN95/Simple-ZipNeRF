# Shree KRISHNAya Namaha
# Creates videos using the frames generated during rendering.
# Author: Nagabhushan S N
# Last Modified: 21/02/2024

import argparse
import json
import os
import time
import datetime
import traceback

from pathlib import Path
from typing import List

import numpy
import skimage.io
import skvideo.io
from matplotlib import pyplot
from matplotlib import colormaps
from tqdm import tqdm

from internal import vis

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def save_video(path: Path, video: numpy.ndarray, framerate: int):
    skvideo.io.vwrite(path.as_posix(), video,
                      inputdict={'-r': str(framerate)},
                      outputdict={'-c:v': 'libx264', '-vf': 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt': 'yuv420p'})
    return


def save_rgb_video(path: Path, frames_dirpath: Path, framerate: int):
    cmd = f"ffmpeg -y -framerate {framerate} -pattern_type glob -i '{frames_dirpath.as_posix()}/*.png' -c:v libx264 -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -pix_fmt yuv420p {path.as_posix()}"
    os.system(cmd)
    return


def save_depth_video(path: Path, depths_dirpath: Path, framerate: int):
    depth_colormap = pyplot.get_cmap('plasma')
    depths = []
    for depth_path in tqdm(sorted(depths_dirpath.glob('*.npy'))):
        depth = numpy.load(depth_path.as_posix())
        depths.append(1 / depth)
    depths = numpy.stack(depths, axis=0)
    max_depth = depths.max()
    depths_norm = (depths / max_depth * 255).astype(numpy.uint8)
    depths_colored = depth_colormap(depths_norm)[:, :, :, :3] * 255
    save_video(path, depths_colored, framerate)
    return


def save_distance_video(path: Path, distance_dirpath: Path, framerate: int, render_dist_percentile: float):
    distances = []
    distance_sample = skimage.io.imread(sorted(distance_dirpath.glob('*.tiff'))[0].as_posix())
    distance_limits = numpy.percentile(distance_sample.flatten(), [render_dist_percentile, 100-render_dist_percentile])
    low, high = distance_limits
    depth_curve_fn = lambda x: -numpy.log(x + numpy.finfo(numpy.float32).eps)
    for distance_path in tqdm(sorted(distance_dirpath.glob('*.tiff'))):
        distance = skimage.io.imread(distance_path.as_posix())
        distance_vis = vis.visualize_cmap(distance, numpy.ones_like(distance), colormaps.get_cmap('turbo'), low, high, curve_fn=depth_curve_fn)
        distance_image = (numpy.clip(numpy.nan_to_num(distance_vis), 0., 1.) * 255).astype(numpy.uint8)
        distances.append(distance_image)
    distances = numpy.stack(distances, axis=0)
    save_video(path, distances, framerate)
    return


def save_acc_video(path: Path, acc_dirpath: Path, framerate: int):
    accs = []
    for acc_path in tqdm(sorted(acc_dirpath.glob('*.tiff'))):
        acc = skimage.io.imread(acc_path.as_posix())
        accs.append(acc)
    accs = numpy.stack(accs, axis=0)
    max_acc = accs.max()
    accs_norm = (accs / max_acc * 255).astype(numpy.uint8)
    save_video(path, accs_norm, framerate)
    return


def generate_videos(test_dirpath: Path, scene_names: List[str], framerate: int, render_dist_percentile: float):
    for scene_name in scene_names:
        videos_dirpaths = sorted(test_dirpath.glob(f'{scene_name}_video*'))
        for videos_dirpath in videos_dirpaths:
            # Create RGB video
            pred_frames_dirpath = videos_dirpath / 'predicted_frames'
            pred_video_path = videos_dirpath / 'PredictedVideo.mp4'
            save_rgb_video(pred_video_path, pred_frames_dirpath, framerate)

            # Create depth video
            pred_depths_dirpath = videos_dirpath / 'predicted_depths'
            pred_depth_video_path = videos_dirpath / 'PredictedDepth.mp4'
            save_depth_video(pred_depth_video_path, pred_depths_dirpath, framerate)

            # Create distance mean videos
            pred_distance_mean_dirpath = videos_dirpath / 'predicted_distance_mean'
            pred_distance_mean_video_path = videos_dirpath / 'PredictedDistanceMean.mp4'
            save_distance_video(pred_distance_mean_video_path, pred_distance_mean_dirpath, framerate, render_dist_percentile)

            # Create distance median videos
            pred_distance_median_dirpath = videos_dirpath / 'predicted_distance_median'
            pred_distance_median_video_path = videos_dirpath / 'PredictedDistanceMedian.mp4'
            save_distance_video(pred_distance_median_video_path, pred_distance_median_dirpath, framerate, render_dist_percentile)

            # Create acc videos
            pred_acc_dirpath = videos_dirpath / 'predicted_acc'
            pred_acc_video_path = videos_dirpath / 'PredictedAcc.mp4'
            save_acc_video(pred_acc_video_path, pred_acc_dirpath, framerate)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dirpath', type=str)
    parser.add_argument('--scene-names', nargs='+', type=str)
    parser.add_argument('--framerate', type=int, default=30)
    parser.add_argument('--render-dist-percentile', type=float, default=0.5)

    args = parser.parse_args()
    generate_videos(Path(args.test_dirpath), args.scene_names, args.framerate, args.render_dist_percentile)
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = 'Error: ' + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
