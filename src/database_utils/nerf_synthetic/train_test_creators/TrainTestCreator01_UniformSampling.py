# Shree KRISHNAya Namaha
# Creates train-test sets.
# Authors: Nagabhushan S N, Harsha Mupparaju, Adithyan Karanayil
# Last Modified: 20/06/2024

import time
import datetime
import traceback
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


def create_scene_frames_data(scene_name, group, frame_nums):
    frames_data = [[scene_name, group, frame_num] for frame_num in frame_nums]
    return frames_data


def create_data_frame(frames_data: list):
    frames_array = numpy.array(frames_data)
    frames_data = pandas.DataFrame(frames_array, columns=['scene_name', 'group', 'pred_frame_num'])
    return frames_data


def create_train_test_set(configs: dict):
    root_dirpath = Path('../../')

    set_num = configs['set_num']
    set_dirpath = root_dirpath / f'data/train_test_sets/set{set_num:02}'
    set_dirpath.mkdir(parents=True, exist_ok=True)

    scenes_dirpath = root_dirpath / 'data/all/database_data/'
    scene_names = sorted(map(lambda path: path.stem, scenes_dirpath.iterdir()))
    num_train_frames = configs['num_train_frames']

    train_data, val_data, test_data = [], [], []
    for scene_name in scene_names:
        if num_train_frames == 100:
            train_frame_nums = range(100)
        else:
            train_frame_nums = numpy.round(numpy.linspace(-1, 100, num_train_frames+2)).astype('int')[1:-1]
        val_frame_nums = range(100)
        test_frame_nums = range(200)

        train_data.extend(create_scene_frames_data(scene_name, 'train', train_frame_nums))
        val_data.extend(create_scene_frames_data(scene_name, 'validation', val_frame_nums))
        test_data.extend(create_scene_frames_data(scene_name, 'test', test_frame_nums))

    train_data = create_data_frame(train_data)
    train_data_path = set_dirpath / 'TrainVideosData.csv'
    train_data.to_csv(train_data_path, index=False)

    val_data = create_data_frame(val_data)
    val_data_path = set_dirpath / 'ValidationVideosData.csv'
    val_data.to_csv(val_data_path, index=False)

    test_data = create_data_frame(test_data)
    test_data_path = set_dirpath / 'TestVideosData.csv'
    test_data.to_csv(test_data_path, index=False)

    configs_path = set_dirpath / 'Configs.json'
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return


def demo1():
    configs = {
        'TrainTestCreator': this_filename,
        'set_num': 1,
        'num_train_frames': 100
    }
    create_train_test_set(configs)

    configs = {
        'TrainTestCreator': this_filename,
        'set_num': 2,
        'num_train_frames': 4
    }
    create_train_test_set(configs)

    configs = {
        'TrainTestCreator': this_filename,
        'set_num': 3,
        'num_train_frames': 8
    }
    create_train_test_set(configs)

    configs = {
        'TrainTestCreator': this_filename,
        'set_num': 4,
        'num_train_frames': 12
    }
    create_train_test_set(configs)
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
