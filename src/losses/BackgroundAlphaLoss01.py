# Shree KRISHNAya Namaha
# Simple mean squared error loss between the accumulated weights and the alpha
# Author: Nagabhushan S N
# Last Modified: 25/03/2024

from pathlib import Path

import torch

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def compute_loss(batch: dict, renderings: dict, config):
    num_levels = len(renderings)
    total_loss = 0
    for i in range(num_levels):
        acc = renderings[i]['acc']
        alpha = batch['alpha']
        loss_i = torch.mean(torch.square(acc - alpha))
        total_loss += loss_i
    return total_loss
