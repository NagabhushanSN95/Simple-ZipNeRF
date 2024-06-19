# Shree KRISHNAya Namaha
# A Factory method that returns a Loss
# Author: Nagabhushan S N
# Last Modified: 15/03/2024

import importlib.util
import inspect


def get_loss_function(loss_name):
    loss_function = None
    module = importlib.import_module(f'losses.{loss_name}')
    candidate_functions = inspect.getmembers(module, inspect.isfunction)
    for candidate_function in candidate_functions:
        if candidate_function[0] == 'compute_loss':
            loss_function = candidate_function[1]
            break
    if loss_function is None:
        raise RuntimeError(f'Unknown loss function: {loss_function}')
    return loss_function
