# Shree KRISHNAya Namaha
# Downloads the data and extracts rgb, camera intrinsics and extrisics
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024

import os
import shutil
import sys
import time
import datetime
import traceback
from collections import OrderedDict

import numpy
import simplejson
import skimage.io
import skvideo.io
import pandas

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot

import libraries.llff.poses.colmap_read_model as read_model

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def read_colmap_data(data_dirpath):
    camerasfile = os.path.join(data_dirpath, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)

    camera_intrinsics = {}
    for key in camdata.keys():
        cam = camdata[key]
        h, w, fx, fy = cam.height, cam.width, *cam.params[:2]
        intrinsics = numpy.eye(3)
        intrinsics[0, 0] = fx
        intrinsics[1, 1] = fy
        intrinsics[0, 2] = w / 2
        intrinsics[1, 2] = h / 2
        camera_intrinsics[key] = intrinsics

    bounds_file = os.path.join(data_dirpath, 'poses_bounds.npy')
    bounds = numpy.load(bounds_file)[:, 15:17]

    images_filepath = os.path.join(data_dirpath, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(images_filepath)

    frames_data = []
    bottom = numpy.array([0, 0, 0, 1]).reshape([1, 4])
    for i, k in enumerate(imdata):
        im = imdata[k]
        name = im.name[:-4]
        intrinsic = camera_intrinsics[im.camera_id]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3, 1])
        extrinsic = numpy.concatenate([numpy.concatenate([R, t], 1), bottom], 0)
        frames_data.append([name, intrinsic.ravel(), extrinsic.ravel(), bounds[i]])
    return frames_data


def copy_image(old_filepath: Path, new_filepath: Path):
    if old_filepath.suffix.lower() == new_filepath.suffix.lower():
        shutil.copy(old_filepath, new_filepath)
    else:
        image = skimage.io.imread(old_filepath.as_posix())
        skimage.io.imsave(new_filepath.as_posix(), image)
    return


def extract_data(unzipped_dirpath: Path, database_dirpath: Path):
    """
    Extracts data from all the scenes
    """
    print('Extracting data from all the scenes at ' + database_dirpath.as_posix())
    scene_dirpaths = sorted(filter(lambda path: path.is_dir(), unzipped_dirpath.iterdir()))

    for scene_dirpath in scene_dirpaths:
        scene_name = scene_dirpath.name
        camera_data = read_colmap_data(scene_dirpath)
        camera_data = numpy.array(camera_data)
        camera_data = camera_data[numpy.argsort(camera_data[:, 0])]

        names_mapping, intrinsics, extrinsics, bounds = [], [], [], []
        for frame_num in tqdm(range(len(camera_data)), desc=scene_name):
            old_frame_name, intrinsic, extrinsic, bound = camera_data[frame_num]
            names_mapping.append([old_frame_name, frame_num])
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)
            bounds.append(bound)

            old_filepath = unzipped_dirpath / f'{scene_name}/images/{old_frame_name}.JPG'
            new_filepath = database_dirpath / f'{scene_name}/rgb/{frame_num:04}.jpg'
            if not old_filepath.exists():
                print(f'{old_filepath.as_posix()} does not exist!')
                sys.exit(1)
            new_filepath.parent.mkdir(parents=True, exist_ok=True)
            copy_image(old_filepath, new_filepath)

            old_filepath = unzipped_dirpath / f'{scene_name}/images_2/{old_frame_name}.JPG'
            # if not old_filepath.exists():
            #     old_filepath = scene_dirpath / f'images_4/image{frame_num:03}.png'
            new_filepath = database_dirpath / f'{scene_name}/rgb_down2/{frame_num:04}.png'
            if not old_filepath.exists():
                print(f'{old_filepath.as_posix()} does not exist!')
                sys.exit(1)
            new_filepath.parent.mkdir(parents=True, exist_ok=True)
            copy_image(old_filepath, new_filepath)

            old_filepath = unzipped_dirpath / f'{scene_name}/images_4/{old_frame_name}.JPG'
            # if not old_filepath.exists():
            #     old_filepath = scene_dirpath / f'images_4/image{frame_num:03}.png'
            new_filepath = database_dirpath / f'{scene_name}/rgb_down4/{frame_num:04}.png'
            if not old_filepath.exists():
                print(f'{old_filepath.as_posix()} does not exist!')
                sys.exit(1)
            new_filepath.parent.mkdir(parents=True, exist_ok=True)
            copy_image(old_filepath, new_filepath)

            old_filepath = unzipped_dirpath / f'{scene_name}/images_8/{old_frame_name}.JPG'
            # if not old_filepath.exists():
            #     old_filepath = scene_dirpath / f'images_4/image{frame_num:03}.png'
            new_filepath = database_dirpath / f'{scene_name}/rgb_down8/{frame_num:04}.png'
            if not old_filepath.exists():
                print(f'{old_filepath.as_posix()} does not exist!')
                sys.exit(1)
            new_filepath.parent.mkdir(parents=True, exist_ok=True)
            copy_image(old_filepath, new_filepath)

        names_mapping_data = pandas.DataFrame(names_mapping, columns=['OldFrameName', 'NewFrameNum'])
        names_mapping_path = database_dirpath / f'{scene_name}/FrameNamesMapping.csv'
        names_mapping_data.to_csv(names_mapping_path, index=False)

        intrinsics_array = numpy.stack(intrinsics).reshape(-1, 9)
        intrinsics_path = database_dirpath / f'{scene_name}/CameraIntrinsics.csv'
        numpy.savetxt(intrinsics_path, intrinsics_array, delimiter=',')

        intrinsics_array1 = intrinsics_array.copy()
        intrinsics_array1[:, 0] /= 2
        intrinsics_array1[:, 4] /= 2
        intrinsics_array1[:, 2] /= 2
        intrinsics_array1[:, 5] /= 2
        intrinsics_path = database_dirpath / f'{scene_name}/CameraIntrinsics_down2.csv'
        numpy.savetxt(intrinsics_path, intrinsics_array1, delimiter=',')

        intrinsics_array1 = intrinsics_array.copy()
        intrinsics_array1[:, 0] /= 4
        intrinsics_array1[:, 4] /= 4
        intrinsics_array1[:, 2] /= 4
        intrinsics_array1[:, 5] /= 4
        intrinsics_path = database_dirpath / f'{scene_name}/CameraIntrinsics_down4.csv'
        numpy.savetxt(intrinsics_path, intrinsics_array1, delimiter=',')

        intrinsics_array1 = intrinsics_array.copy()
        intrinsics_array1[:, 0] /= 8
        intrinsics_array1[:, 4] /= 8
        intrinsics_array1[:, 2] /= 8
        intrinsics_array1[:, 5] /= 8
        intrinsics_path = database_dirpath / f'{scene_name}/CameraIntrinsics_down8.csv'
        numpy.savetxt(intrinsics_path, intrinsics_array1, delimiter=',')

        extrinsics_array = numpy.stack(extrinsics).reshape(-1, 16)
        extrinsics_path = database_dirpath / f'{scene_name}/CameraExtrinsics.csv'
        numpy.savetxt(extrinsics_path, extrinsics_array, delimiter=',')

        bounds_array = numpy.stack(bounds)
        bounds_path = database_dirpath / f'{scene_name}/DepthBounds.csv'
        numpy.savetxt(bounds_path, bounds_array, delimiter=',')
    return


def download_data(download_dirpath: Path):
    zip_filepath = download_dirpath / '360_v2.zip'
    if not zip_filepath.exists():
        print('Downloading data at ' + download_dirpath.as_posix())
        url = 'http://storage.googleapis.com/gresearch/refraw360/360_v2.zip'
        cmd = f'wget {url} -P {download_dirpath.as_posix()}'
        os.system(cmd)
        print('Download complete!')
    else:
        print('Data already downloaded at ' + download_dirpath.as_posix())
    return


def unzip_data(downloaded_dirpath: Path, unzipped_data_dirpath: Path):
    if not unzipped_data_dirpath.exists():
        print('Unzipping data at ' + unzipped_data_dirpath.as_posix())
        zip_filepath = downloaded_dirpath / '360_v2.zip'
        cmd = f'unzip {zip_filepath.as_posix()} -d {unzipped_data_dirpath.as_posix()}'
        os.system(cmd)
        print('Unzipping complete!')
    else:
        print('Data already unzipped at ' + unzipped_data_dirpath.as_posix())
    return


def main():
    root_dirpath = Path('../../../../')
    database_dirpath = root_dirpath / 'data/databases/MipNeRF360/data/'
    download_dirpath = database_dirpath / 'raw/downloaded_data'
    unzipped_dirpath = database_dirpath / 'raw/unzipped_data'
    database_data_dirpath = database_dirpath / 'all/database_data'

    download_data(download_dirpath)
    unzip_data(download_dirpath, unzipped_dirpath)
    extract_data(unzipped_dirpath, database_data_dirpath)
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
