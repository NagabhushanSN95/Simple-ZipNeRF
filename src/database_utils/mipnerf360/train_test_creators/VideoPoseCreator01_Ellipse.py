# Shree KRISHNAya Namaha
# creates poses for elliptical video
# Extended from NeRF_LLFF/VideoPoseCreator04_Spiral.py. The code to generate elliptical poses is borrowed from zip-nerf.
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024

import time
import datetime
import traceback
import numpy
import numpy as np
import simplejson
import skimage.io
import skvideo.io
import pandas
from zipnerf_utils import stepfun

from pathlib import Path

from scipy.spatial.transform import Rotation
from tqdm import tqdm
from matplotlib import pyplot

this_filepath = Path(__file__)
this_filename = this_filepath.stem
this_filenum = int(this_filename[16:18])


def focus_point_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def viewmatrix(lookdir, up, position):
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def generate_ellipse_path(poses: numpy.ndarray, num_frames=120, const_speed=True, z_variation=0., z_phase=0.):
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], 0])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
            low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                           (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.linspace(0, 2. * np.pi, num_frames + 1, endpoint=True)
    positions = get_positions(theta)

    if const_speed:
        # Resample theta angles so that the velocity is closer to constant.
        lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
        theta = stepfun.sample_np(None, theta, np.log(lengths), num_frames + 1)
        positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(p - center, up, p) for p in positions])


def transform_poses_pca(poses):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)
    poses_recentered = transform @ poses

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1, 1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    # transform = transform @ np.diag(np.array([1] * 3 + [scale_factor]))

    return poses_recentered, transform, scale_factor


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def pre_process_poses(poses: numpy.ndarray):
    """
    Apply operations on poses that are applied before generating the elliptical path poses
    """
    c2w_poses = numpy.linalg.inv(poses)
    std_poses = c2w_poses @ np.diag([1, -1, -1, 1])
    transformed_poses, transform, translation_scale = transform_poses_pca(std_poses)
    return transformed_poses, transform, translation_scale


def post_process_poses(poses: numpy.ndarray, transform: numpy.ndarray, translation_scale: float):
    """
    Undo the operations on poses that will be applied when loading
    """
    unscaled_poses = pad_poses(poses.copy())
    unscaled_poses[:, :3, 3] /= translation_scale
    untransformed_poses = np.linalg.inv(transform) @ unscaled_poses
    opencv_poses = untransformed_poses @ np.diag([1, -1, -1, 1])
    w2c_poses = np.linalg.inv(opencv_poses)
    return w2c_poses


def create_video_poses(poses: numpy.ndarray, num_frames: int, constant_speed: bool, z_variation: float, z_phase: float):
    pre_processed_poses, transform, translation_scale = pre_process_poses(poses)
    path_poses = generate_ellipse_path(pre_processed_poses, num_frames, constant_speed, z_variation, z_phase)
    post_processed_poses = post_process_poses(path_poses, transform, translation_scale)
    video_poses = numpy.concatenate([poses[:1], post_processed_poses], axis=0)
    return video_poses


def save_configs(output_dirpath: Path, configs: dict):
    configs_path = output_dirpath / 'Configs.json'
    if configs_path.exists():
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = simplejson.load(configs_file)
        for key in old_configs.keys():
            if key not in configs.keys():
                configs[key] = old_configs[key]
        if configs != old_configs:
            raise RuntimeError('Configs mismatch while resuming resuming video pose generation.')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return


def save_video_poses(configs: dict):
    root_dirpath = Path('../../../../data/databases/MipNeRF360/')
    set_num = configs['set_num']
    num_frames = configs['num_frames']
    constant_speed = configs['constant_speed']
    z_variation = configs['z_variation']
    z_phase = configs['z_phase']

    output_dirpath = root_dirpath / f'data/train_test_sets/set{set_num:02}/video_poses{this_filenum:02}'
    output_dirpath.mkdir(parents=True, exist_ok=False)
    save_configs(output_dirpath, configs)

    train_videos_path = root_dirpath / f'data/train_test_sets/set{set_num:02}/TrainVideosData.csv'
    train_videos_data = pandas.read_csv(train_videos_path)

    scene_names = numpy.unique(train_videos_data['scene_name'])
    for scene_name in scene_names:
        trans_mats_path = root_dirpath / f'data/all/database_data/{scene_name}/CameraExtrinsics.csv'
        trans_mats = numpy.loadtxt(trans_mats_path.as_posix(), delimiter=',').reshape((-1, 4, 4))

        video_poses = create_video_poses(trans_mats, num_frames, constant_speed, z_variation, z_phase)
        video_poses_flat = numpy.reshape(video_poses, (video_poses.shape[0], -1))

        output_path = output_dirpath / f'{scene_name}.csv'
        numpy.savetxt(output_path.as_posix(), video_poses_flat, delimiter=',')
    video_frame_nums = numpy.arange(num_frames)
    output_path = output_dirpath / 'VideoFrameNums.csv'
    numpy.savetxt(output_path.as_posix(), video_frame_nums, fmt='%i', delimiter=',')
    return


def demo1():
    configs = {
        'PosesCreator': this_filename,
        'set_num': 1,
        'num_frames': 960,
        'constant_speed': True,
        'z_variation': 0,
        'z_phase': 0,
    }
    save_video_poses(configs)

    configs = {
        'PosesCreator': this_filename,
        'set_num': 4,
        'num_frames': 960,
        'constant_speed': True,
        'z_variation': 0,
        'z_phase': 0,
    }
    save_video_poses(configs)

    configs = {
        'PosesCreator': this_filename,
        'set_num': 6,
        'num_frames': 960,
        'constant_speed': True,
        'z_variation': 0,
        'z_phase': 0,
    }
    save_video_poses(configs)

    configs = {
        'PosesCreator': this_filename,
        'set_num': 10,
        'num_frames': 960,
        'constant_speed': True,
        'z_variation': 0,
        'z_phase': 0,
    }
    save_video_poses(configs)
    return


def main():
    demo1()
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
