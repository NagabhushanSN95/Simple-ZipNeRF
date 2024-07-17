# Shree KRISHNAya Namaha
# Takes Configs.py and creates new configs file in scene_name for every scene. This can then be used by bash file to
# call individual train and test commands.
# Author: Nagabhushan S N
# Last Modified: 23/03/2024

import argparse
import importlib
import importlib.util
import json
import re
import shutil
from collections import OrderedDict
from typing import Dict, Any

from pathlib import Path

from deepdiff import DeepDiff

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def read_gin_configs(configs_path: Path) -> list:
    with open(configs_path.as_posix(), 'r') as configs_file:
        configs = configs_file.readlines()
    return configs


def save_gin_configs(scene_configs_path: Path, scene_configs: list):
    if scene_configs_path.exists():
        with open(scene_configs_path.as_posix(), 'r') as scene_configs_file:
            scene_configs_old = scene_configs_file.readlines()
        if scene_configs_old != scene_configs:
            raise RuntimeError(f'Scene configs file {scene_configs_path} already exists and is different from the new configs:\n'
                               f'{DeepDiff(scene_configs_old, scene_configs)}')
    with open(scene_configs_path.as_posix(), 'w') as scene_configs_file:
        scene_configs_file.writelines(scene_configs)
    return


def save_json_configs(configs_path: Path, configs: Dict[str, Any]):
    if configs_path.exists():
        with open(configs_path.as_posix(), 'r') as configs_file:
            configs_old = json.load(configs_file)
        if configs_old != configs:
            raise RuntimeError(f'Configs file {configs_path} already exists and is different from the new configs:\n'
                               f'{DeepDiff(configs_old, configs)}')
    with open(configs_path.as_posix(), 'w') as configs_file:
        json.dump(configs, configs_file, indent=4)
    return


def save_train_configs(args):
    configs_path = Path(args.configs_path)
    print(configs_path)
    train_dirpath = configs_path.parent
    configs = read_gin_configs(configs_path)

    # Insert the config line for train_num
    train_num = int(train_dirpath.stem[5:])
    config_line = f'Config.train_num = {train_num}\n'
    if config_line not in configs:
        configs.insert(2, f'Config.train_num = {train_num}\n')

    # Insert the config line for train_set_num
    config_matcher_line = 'Config.train_set_num *= *(\\d)\\n'
    match = re.search(config_matcher_line, ''.join(configs))
    if args.train_set_num is not None:
        train_set_num = args.train_set_num
    elif match is not None:
        train_set_num = int(match.group(1))
    else:
        raise RuntimeError(f'train_set_num not configured.')
    # If the config line already exists, we need to overwrite it. So, remove it.
    if match is not None:
        configs.remove(match.group(0))
    config_line = f'Config.train_set_num = {train_set_num}\n'
    configs.insert(3, config_line)

    # Insert the config line for test_set_num
    config_matcher_line = 'Config.test_set_num *= *(\\d)\\n'
    match = re.search(config_matcher_line, ''.join(configs))
    if args.test_set_num is not None:
        test_set_num = args.test_set_num
    elif match is not None:
        test_set_num = int(match.group(1))
    else:
        test_set_num = train_set_num
    # If the config line already exists, we need to overwrite it. So, remove it.
    if match is not None:
        configs.remove(match.group(0))
    config_line = f'Config.test_set_num = {test_set_num}\n'
    configs.insert(4, config_line)

    for scene_name in args.scene_names:
        scene_dirpath = train_dirpath / f'{scene_name}'
        scene_dirpath.mkdir(parents=True, exist_ok=True)

        scene_configs = configs.copy()
        scene_configs.insert(5, f'Config.scene_name = "{scene_name}"\n')

        scene_configs_path = scene_dirpath / f'{scene_name}.gin'
        save_gin_configs(scene_configs_path, scene_configs)
    return


def save_test_configs(args):
    configs_path = Path(args.configs_path)
    print(configs_path)
    train_dirpath = configs_path.parent
    configs = read_gin_configs(configs_path)

    # Insert the config line for train_num
    train_num = int(train_dirpath.stem[5:])
    config_line = f'Config.train_num = {train_num}\n'
    if config_line not in configs:
        configs.insert(2, f'Config.train_num = {train_num}\n')

    # Insert the config line for test_num
    config_matcher_line = 'Config.test_num *= *(\\d)\\n'
    match = re.search(config_matcher_line, ''.join(configs))
    if args.test_num is not None:
        test_num = args.test_num
    elif match is not None:
        test_num = int(match.group(1))
    else:
        test_num = train_num
    # If the config line already exists, we need to overwrite it. So, remove it.
    if match is not None:
        configs.remove(match.group(0))
    config_line = f'Config.test_num = {test_num}\n'
    configs.insert(3, config_line)
    test_dirpath = train_dirpath.parent.parent / f'testing/test{test_num:04}'

    # Insert the config line for train_set_num
    config_matcher_line = 'Config.train_set_num *= *(\\d)\\n'
    match = re.search(config_matcher_line, ''.join(configs))
    if args.train_set_num is not None:
        train_set_num = args.train_set_num
    elif match is not None:
        train_set_num = int(match.group(1))
    else:
        raise RuntimeError(f'train_set_num not configured.')
    # If the config line already exists, we need to overwrite it. So, remove it.
    if match is not None:
        configs.remove(match.group(0))
    config_line = f'Config.train_set_num = {train_set_num}\n'
    configs.insert(4, config_line)

    # Insert the config line for test_set_num
    config_matcher_line = 'Config.test_set_num *= *(\\d)\\n'
    match = re.search(config_matcher_line, ''.join(configs))
    if args.test_set_num is not None:
        test_set_num = args.test_set_num
    elif match is not None:
        test_set_num = int(match.group(1))
    else:
        raise RuntimeError(f'test_set_num not configured.')
    # If the config line already exists, we need to overwrite it. So, remove it.
    if match is not None:
        configs.remove(match.group(0))
    config_line = f'Config.test_set_num = {test_set_num}\n'
    configs.insert(5, config_line)

    for scene_name in args.scene_names:
        scene_dirpath = test_dirpath / f'{scene_name}'
        scene_dirpath.mkdir(parents=True, exist_ok=True)

        scene_configs = configs.copy()
        scene_configs.insert(6, f'Config.scene_name = "{scene_name}"\n')

        scene_configs_path = scene_dirpath / f'{scene_name}.gin'
        save_gin_configs(scene_configs_path, scene_configs)

    # Save test configs file
    test_configs = {
        'test_num': test_num,
        'train_num': train_num,
        'test_set_num': test_set_num,
    }
    test_configs_path = test_dirpath / 'Configs.json'
    save_json_configs(test_configs_path, test_configs)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs-path', type=str)
    parser.add_argument('--scene-names', type=str, nargs='*')
    parser.add_argument('--mode', type=str, choices=['train', 'test'])
    # parser.add_argument('--train-num', type=int, default=None)
    parser.add_argument('--test-num', type=int, default=None)
    parser.add_argument('--train-set-num', type=int, default=None)
    parser.add_argument('--test-set-num', type=int, default=None)

    args = parser.parse_args()
    if args.mode.lower() == 'train':
        save_train_configs(args)
    elif args.mode.lower() == 'test':
        save_test_configs(args)
    else:
        raise RuntimeError(f'Invalid mode: {args.mode}')
    return


if __name__ == '__main__':
    main()
