# Shree KRISHNAya Namaha
# Extracts RGB images, camera intrinsics and extrisics
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024

import json
import os
import shutil
import sys
import time
import datetime
import traceback
import zipfile
from collections import OrderedDict

import numpy
import simplejson
import skimage.io
import skvideo.io
import pandas

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def extract_rgb(src_path: Path, tgt_path: Path):
    tgt_path.parent.mkdir(parents=True, exist_ok=True)
    image = skimage.io.imread(src_path.as_posix())
    skimage.io.imsave(tgt_path.as_posix(), image)
    return 


def extract_depth(src_path: Path, tgt_path: Path):
    if src_path.exists():
        tgt_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_path, tgt_path)
    return


def extract_normal(src_path: Path, tgt_path: Path):
    if src_path.exists():
        tgt_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_path, tgt_path)
    return


def extract_intrinsic(camera_angle_x, camera_angle_y):
    h, w = 800, 800
    intrinsic = numpy.eye(3)
    fx = w/2 / numpy.tan(0.5 * camera_angle_x)
    fy = h/2 / numpy.tan(0.5 * camera_angle_y)
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = w/2
    intrinsic[1, 2] = h/2
    return intrinsic


def extract_extrinsic(transform_matrix):
    extrinsic_std_c2w = numpy.array(transform_matrix)
    extrinsic_std_w2c = numpy.linalg.inv(extrinsic_std_c2w)
    extrinsic_opencv = convert_pose_from_standard_to_opencv_coordinates(extrinsic_std_w2c[None])[0]
    return extrinsic_opencv


def convert_pose_from_standard_to_opencv_coordinates(poses):
    # Convert from Colmap/RE10K convention to NeRF convention: (x,-y,-z) to (x,y,z)
    perm_matrix = numpy.eye(3)
    perm_matrix[1, 1] = -1
    perm_matrix[2, 2] = -1
    std_poses = change_coordinate_system(poses, perm_matrix)
    return std_poses


def change_coordinate_system(poses: numpy.ndarray, p: numpy.ndarray):
    changed_poses = []
    for pose in poses:
        r = pose[:3, :3]
        t = pose[:3, 3:]
        rc = p.T @ r @ p
        tc = p @ t
        changed_pose = numpy.concatenate([numpy.concatenate([rc, tc], axis=1), pose[3:]], axis=0)
        changed_poses.append(changed_pose)
    changed_poses = numpy.stack(changed_poses)
    return changed_poses


def extract_scene_group_data(unzipped_dirpath, database_data_dirpath, scene_name, group_data):
    tgt_group_name, src_group_name, num_frames = group_data

    # Extract RGB, Depth and Normals
    for frame_num in tqdm(range(num_frames), desc=f'Extracting {scene_name} {tgt_group_name}'):
        rgb_src_path = unzipped_dirpath / f'nerf_synthetic/{scene_name}/{src_group_name}/r_{frame_num}.png'
        rgb_tgt_path = database_data_dirpath / f'{scene_name}/{tgt_group_name}/rgb/{frame_num:04}.png'
        extract_rgb(rgb_src_path, rgb_tgt_path)

        depth_src_path = unzipped_dirpath / f'nerf_synthetic/{scene_name}/{src_group_name}/r_{frame_num}_depth_0000.png'
        depth_tgt_path = database_data_dirpath / f'{scene_name}/{tgt_group_name}/depth/{frame_num:04}.png'
        extract_depth(depth_src_path, depth_tgt_path)

        normal_src_path = unzipped_dirpath / f'nerf_synthetic/{scene_name}/{src_group_name}/r_{frame_num}_normal_0000.png'
        normal_tgt_path = database_data_dirpath / f'{scene_name}/{tgt_group_name}/normal/{frame_num:04}.png'
        extract_normal(normal_src_path, normal_tgt_path)

    # Extract Camera Intrinsics and Extrinsics
    intrinsics, extrinsics = [], []
    camera_data_path = unzipped_dirpath / f'nerf_synthetic/{scene_name}/transforms_{src_group_name}.json'
    with open(camera_data_path, 'r') as json_file:
        camera_data = json.load(json_file)
    camera_angle_x = camera_data['camera_angle_x']
    intrinsic = extract_intrinsic(camera_angle_x, camera_angle_x)
    for frame_num in range(num_frames):
        extrinsic = extract_extrinsic(camera_data['frames'][frame_num]['transform_matrix'])
        intrinsics.append(intrinsic)
        extrinsics.append(extrinsic)
    intrinsics = numpy.stack(intrinsics).reshape((-1, 9))
    extrinsics = numpy.stack(extrinsics).reshape((-1, 16))
    intrinsics_tgt_path = database_data_dirpath / f'{scene_name}/{tgt_group_name}/CameraIntrinsics.csv'
    extrinsics_tgt_path = database_data_dirpath / f'{scene_name}/{tgt_group_name}/CameraExtrinsics.csv'
    numpy.savetxt(intrinsics_tgt_path.as_posix(), intrinsics, delimiter=',')
    numpy.savetxt(extrinsics_tgt_path.as_posix(), extrinsics, delimiter=',')
    return


def unzip_data(zip_filepath, unzipped_dirpath):
    print('Unzipping data...')
    if unzipped_dirpath.exists():
        shutil.rmtree(unzipped_dirpath)
    unzipped_dirpath.mkdir()

    with zipfile.ZipFile(zip_filepath, 'r') as zip_file:
        zip_file.extractall(unzipped_dirpath)
    print('Unzipping done.')
    return


def extract_data(unzipped_dirpath, database_data_dirpath, scene_names, group_names):
    print('Extracting data...')
    for scene_name in scene_names:
        for group_data in group_names:
            extract_scene_group_data(unzipped_dirpath, database_data_dirpath, scene_name, group_data)
    shutil.rmtree(unzipped_dirpath)
    print('Extracting done.')
    return


def main():
    root_dirpath = Path('../../../../')
    database_dirpath = root_dirpath / f'data/databases/NeRF_Synthetic/data'
    zip_filepath = database_dirpath / f'raw/downloaded_data/nerf_synthetic.zip'
    unzipped_dirpath = database_dirpath / 'raw/unzipped_data'
    database_data_dirpath = database_dirpath / 'all/database_data'

    scene_names = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
    group_names = [('train', 'train', 100), ('validation', 'val', 100), ('test', 'test', 200)]

    unzip_data(zip_filepath, unzipped_dirpath)
    extract_data(unzipped_dirpath, database_data_dirpath, scene_names, group_names)
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
