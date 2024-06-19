# Shree KRISHNAya Namaha
# Depth MSE loss function between Main NeRF and Augmented NeRF. Reprojection error (patch-wise) is employed to determine
# the more accurate depth estimate.
# Extended from AugmentationsDepthLoss08.py with a bug fix in reading extrinsics
# Based on VSR001/AugmentationsDepthLoss11.py
# Author: Nagabhushan S N
# Last Modified: 17/03/2024

from pathlib import Path

import torch
from scipy.spatial.transform import Rotation
from torch.nn import functional as F

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def compute_loss(batch: dict, renderings_main: dict, renderings_aug: dict, config):
    total_loss = torch.tensor(0).to(batch['rgb'])
    
    indices_mask_nerf = batch.get('indices_mask_nerf', torch.ones_like(batch['rgb'][:, 0, 0, 0], dtype=torch.bool))
    rays_o = batch['origins'][indices_mask_nerf, 0, 0]                 # (num_rays, 3)
    rays_d = batch['directions'][indices_mask_nerf, 0, 0]              # (num_rays, 3)
    gt_images = batch['common_data']['images_all']                     # (num_views, H, W, 3)
    intrinsics = batch['common_data']['intrinsics_all']                # (num_views, 3, 3)
    extrinsics = batch['common_data']['extrinsics_all_c2w_nerf']       # (num_views, 4, 4)
    resolution = batch['common_data']['resolution']                    # (H, W)
    pixel_ids = batch['pixel_id'][indices_mask_nerf, 0, 0].long()      # (num_rays, 3)

    for i in range(len(renderings_main)):
        depth_main_t = renderings_main[i]['depth'][:, 0, 0]    # (num_rays, )
        depth_aug_t = renderings_aug[i][f'depth'][:, 0, 0]     # (num_rays, )
        depth_main_s = renderings_main[i]['depth_s'][:, 0, 0]  # (num_rays, )
        depth_aug_s = renderings_aug[i][f'depth_s'][:, 0, 0]   # (num_rays, )
        loss_i = compute_depth_loss(config, depth_main_t, depth_aug_t, depth_main_s, depth_aug_s,
                                         rays_o, rays_d, extrinsics, gt_images, pixel_ids, intrinsics, resolution)
        total_loss += loss_i

    return total_loss


def compute_depth_loss(config, depth1_t, depth2_t, depth1_s, depth2_s,
                       rays_o, rays_d, gt_poses, gt_images, pixel_ids, intrinsics, resolution) -> torch.Tensor:
    """
    Computes the loss for nerf samples (and not for sparse_depth or any other samples)

    Naming convention
    1, 2 -> refers to two different models
    a, b -> refers to source view and the other reprojection view

    :param config:
    :param depth1_t:
    :param depth2_t:
    :param depth1_s:
    :param depth2_s:
    :param rays_o:
    :param rays_d:
    :param gt_poses:
    :param gt_images:
    :param pixel_ids:
    :param intrinsics:
    :param resolution:
    :return:
    """
    px, py = config.augmentation_loss_patch_size
    hpx, hpy = px // 2, py // 2
    rmse_threshold = config.augmentation_loss_rmse_threshold
    h, w = resolution
    image_ids = pixel_ids[:, 0]

    closest_image_ids = get_closest_image_id(gt_poses, image_ids)
    image_ids_a = image_ids
    pixel_ids_a = pixel_ids
    image_ids_b = closest_image_ids

    poses_b = gt_poses[image_ids_b]
    points1a = rays_o + rays_d * depth1_t.unsqueeze(-1)
    points2a = rays_o + rays_d * depth2_t.unsqueeze(-1)

    pos1b = reproject(points1a.detach(), poses_b, intrinsics).round().long()
    pos2b = reproject(points2a.detach(), poses_b, intrinsics).round().long()

    x_a, y_a = pixel_ids_a[:, 1], pixel_ids_a[:, 2]
    x1b, y1b = pos1b[:, 0], pos1b[:, 1]
    x2b, y2b = pos2b[:, 0], pos2b[:, 1]

    # Ignore reprojections that were set outside the image
    valid_mask_a = (x_a >= hpx) & (x_a < w - hpx) & (y_a >= hpy) & (y_a < h - hpy)
    valid_mask_1b = (x1b >= hpx) & (x1b < w - hpx) & (y1b >= hpy) & (y1b < h - hpy)
    valid_mask_2b = (x2b >= hpx) & (x2b < w - hpx) & (y2b >= hpy) & (y2b < h - hpy)

    x1b1, y1b1 = torch.clip(x1b, 0, w - 1).long(), torch.clip(y1b, 0, h - 1).long()
    x2b1, y2b1 = torch.clip(x2b, 0, w - 1).long(), torch.clip(y2b, 0, h - 1).long()
    patches_a = torch.zeros(image_ids_a.shape[0], py, px, gt_images.shape[3]).to(image_ids_a.device)  # (nr, py, px, 3)
    patches1b = torch.zeros(image_ids_b.shape[0], py, px, gt_images.shape[3]).to(image_ids_b.device)  # (nr, py, px, 3)
    patches2b = torch.zeros(image_ids_b.shape[0], py, px, gt_images.shape[3]).to(image_ids_b.device)  # (nr, py, px, 3)
    gt_images_padded = F.pad(gt_images, (0, 0, 0, hpy, 0, hpx), mode='constant', value=0)
    for i, y_offset in enumerate(range(-hpy, hpy + 1)):  # y_offset: [-2, -1, 0, 1, 2]
        for j, x_offset in enumerate(range(-hpx, hpx + 1)):
            patches_a[:, i, j, :] = gt_images_padded[image_ids_a, y_a + y_offset, x_a + x_offset]
            patches1b[:, i, j, :] = gt_images_padded[image_ids_b, y1b1 + y_offset, x1b1 + x_offset]
            patches2b[:, i, j, :] = gt_images_padded[image_ids_b, y2b1 + y_offset, x2b1 + x_offset]

    rmse1 = compute_patch_rmse(patches_a, patches1b)
    rmse2 = compute_patch_rmse(patches_a, patches2b)

    # mask1 is true wherever model1 is more accurate
    mask1 = ((rmse1 < rmse2) | (~valid_mask_2b)) & (rmse1 < rmse_threshold) & valid_mask_1b & valid_mask_a
    # mask2 is true wherever model2 is more accurate
    mask2 = ((rmse2 < rmse1) | (~valid_mask_1b)) & (rmse2 < rmse_threshold) & valid_mask_2b & valid_mask_a

    # Find the pixels where all depths are invalid
    both_invalid_mask = ~(torch.stack([valid_mask_1b, valid_mask_2b], dim=0).any(dim=0))  # (nr, )
    # For the pixels where all depths are invalid, set mask1=True if depth1 > depth2
    mask1 = mask1 | (both_invalid_mask & (depth1_t > depth2_t))
    mask2 = mask2 | (both_invalid_mask & (depth2_t > depth1_t))

    # depth_mse1 is loss on depth1; depth_mse2 is loss on depth2
    depth_mse1 = compute_depth_mse(depth1_s, depth2_s.detach(), mask2)  # (nr, )
    depth_mse2 = compute_depth_mse(depth2_s, depth1_s.detach(), mask1)
    loss = depth_mse1 + depth_mse2
    return loss


def get_closest_image_id(poses: torch.Tensor, image_ids: torch.Tensor) -> torch.Tensor:
    poses_quats = get_quats(poses)
    distances = torch.sqrt(torch.sum(torch.square(poses_quats[image_ids].unsqueeze(1).repeat([1, poses.shape[0], 1]) - poses_quats), dim=2))
    # Taking second smallest value as the smallest distance will always be with the same view at 0.0. Kth value
    # by default randomly returns one index if two distances are the same. Which works for our use-case.
    closest_image_ids = torch.kthvalue(distances, 2, dim=1)[1]
    return closest_image_ids


def get_quats(poses: torch.Tensor):
    poses_np = poses.cpu().numpy()
    quats = Rotation.from_matrix(poses_np[:, :3, :3]).as_quat()
    quats_tr = torch.tensor(quats).to(poses)
    return quats_tr


def reproject(points_to_reproject: torch.Tensor, poses_to_reproject_to: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """

    Args:
        points_to_reproject: (num_rays, )         # TODO SNB: check if this is correct
        poses_to_reproject_to: (num_poses, 4, 4)  # TODO SNB: check if this is correct
        intrinsics: (num_poses, 3, 3)             # TODO SNB: check if this is correct

    Returns:

    """
    other_views_origins = poses_to_reproject_to[:, :3, 3]
    other_views_rotations = poses_to_reproject_to[:, :3, :3]
    reprojected_rays_d = points_to_reproject - other_views_origins

    # for changing coordinate system conventions
    permuter = torch.eye(3).to(points_to_reproject.device)
    permuter[1:] *= -1
    intrinsics = intrinsics[:1]  # TODO: Do not hard-code. Take intrinsic corresponding to each ray

    pos_2 = (intrinsics @ permuter[None] @ other_views_rotations.transpose(1, 2) @ reprojected_rays_d[..., None]).squeeze()
    pos_2 = pos_2[:, :2] / pos_2[:, 2:]
    return pos_2


def compute_patch_rmse(patch1: torch.Tensor, patch2: torch.Tensor) -> torch.Tensor:
    """

    Args:
        patch1: (num_rays, patch_size, patch_size, 3)
        patch2: (num_rays, patch_size, patch_size, 3)

    Returns:
        rmse: (num_rays, )

    """
    rmse = torch.sqrt(torch.mean(torch.square(patch1 - patch2), dim=(1, 2, 3)))
    return rmse


def compute_depth_mse(pred_depth: torch.Tensor, gt_depth: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """

    Args:
        pred_depth: (num_rays, )
        gt_depth: (num_rays, )
        mask: (num_rays, ); Loss is computed only where mask is True

    Returns:

    """
    zero_tensor = torch.tensor(0).to(pred_depth)
    if mask is not None:
        pred_depth[~mask] = 0
        gt_depth[~mask] = 0
    loss_map = torch.square(pred_depth - gt_depth)
    mse = torch.mean(loss_map) if pred_depth.numel() > 0 else zero_tensor
    return mse
