import os
import numpy as np
import cv2
import argparse
import open3d as o3d
import torch
import sys
sys.path.append(os.getcwd())
sys.path.append('/lpai/volumes/jointmodel/yanyunzhi/code/MoGe')
from waymo_helpers import load_calibration, load_track, image_filename_to_cam, image_filename_to_frame
from utils.pcd_utils import storePly, fetchPly
from utils.box_utils import bbox_to_corner3d, inbbox_points
from utils.base_utils import transform_points_numpy
from moge.model.v2 import MoGeModel # type: ignore
moge_model = MoGeModel.from_pretrained("/iag_ad_01/ad/yuanweizhong/ckpt/models--Ruicheng--moge-2-vitl/snapshots/39c4d5e957afe587e04eec59dc2bcc3be5ecd968/model.pt").cuda()

from tqdm import tqdm
import torch_cluster

from typing import Dict, Tuple
def perform_weighted(
    sparse_ori : torch.Tensor, pred_ori : torch.Tensor, dists : torch.Tensor
) -> Tuple[torch.Tensor, ...]:
    """
    Perform weighted operations on input tensors using distance-based weights. A diagonal 
    matrix is created from the normalized weights and used to weight the inputs.
    
    Notes:
        - Weights are calculated as the inverse of the distances.
        - Weights are normalized to ensure they sum to 1.

    Args:
        sparse_ori (torch.Tensor): Sparse original map.
        pred_ori (torch.Tensor): Predicted map.
        dists (torch.Tensor): Distances used for weight calculation.

    Returns:
        Tuple: Containing two tensors:
            - sparse_weighted: The weighted version of the sparse original map.
            - pred_weighted: The weighted version of the predicted map.
    """
    
    weights = 1 / dists
    wsum = weights.sum(dim=1, keepdim=True)
    weights = weights / wsum
    W = torch.diag_embed(weights)
    
    pred_weighted = W @ pred_ori
    sparse_weighted = W @ sparse_ori.unsqueeze(-1)
    return sparse_weighted, pred_weighted
def calc_scale_shift(k_sparse_targets, k_pred_targets, currk_dists=None, knn=False):
    # 为预测目标添加一个小的随机扰动,避免数值不稳定性
    k_pred_targets += torch.rand(*k_pred_targets.shape, device=k_pred_targets.device) * 1e-5
    # 构建最小二乘法的输入矩阵X,包含预测目标和全1向量
    X = torch.stack([k_pred_targets, torch.ones_like(k_pred_targets, device=k_pred_targets.device)], dim=2)
    
    # 对KNN点进行加权处理
    if knn > 0: k_sparse_targets, X = perform_weighted(k_sparse_targets, X, currk_dists)
    # 如果预测目标有多个样本,则扩展稀疏目标的维度
    elif k_pred_targets.shape[0] > 1: k_sparse_targets = k_sparse_targets.unsqueeze(-1)
    
    # 使用最小二乘法求解线性方程组,得到缩放和平移参数
    solution = torch.linalg.lstsq(X, k_sparse_targets)
    # 提取缩放和平移参数,并去除多余的维度
    scale, shift = solution[0][:, 0].squeeze(), solution[0][:, 1].squeeze()
    
    return scale, shift
def knn_aligns(sparse_disparities, pred_disparities, sparse_masks, complete_masks, K) -> Tuple[torch.Tensor, ...]:
    """
    Perform K-Nearest Neighbors (KNN) alignment on sparse and predicted disparities.

    Args:
        sparse_disparities (torch.Tensor): Disparities for sparse map points.
        pred_disparities (torch.Tensor): Predicted disparities for sparse map points.
        sparse_masks (torch.Tensor): Indicating which points in the sparse map are valid.
        complete_masks (torch.Tensor): Indicating which points in the map to be completed.
        K (int): The number of nearest neighbors to find for each map point.

    Returns:
        Tuple: Containing three tensors:
            - dists: The Euclidean distances from each sparse point to its K nearest neighbors.
            - k_sparse_targets: Disparities of the K nearest neighbors from the sparse data.
            - k_pred_targets: Disparities of the K nearest neighbors from the predicted data.
    """
    
    # Coordinates are processed to ensure compatibility with the KNN function.
    batch_sparse = torch.nonzero(sparse_masks, as_tuple=False)[..., [0, 2, 1]].float() # [N, 3] (b, x, y)
    batch_complete = torch.nonzero(complete_masks, as_tuple=False)[..., [0, 2, 1]].float() # [M, 3] (b, x, y)
    
    batch_x, batch_y = batch_sparse[:, 0].contiguous(), batch_complete[:, 0].contiguous()
    x, y = batch_sparse[:, -2:].contiguous(), batch_complete[:, -2:].contiguous()
    
    # Use `torch_cluster.knn` to find K nearest neighbors.
    knn_map = torch_cluster.knn(x=x, y=y, k=K, batch_x=batch_x, batch_y=batch_y) # [2, M * K]
    knn_indices = knn_map[1, :].view(-1, K)
    
    k_sparse_targets = sparse_disparities[sparse_masks][knn_indices]
    k_pred_targets = pred_disparities[sparse_masks][knn_indices]
    
    knn_coords = x[knn_indices]
    expanded_complete_points = y.unsqueeze(dim=1).repeat(1, K, 1)
    dists = torch.norm(expanded_complete_points - knn_coords, dim=2)
    
    return dists, k_sparse_targets, k_pred_targets
def kss_completer(sparse_disparities, pred_disparities, complete_masks, sparse_masks, K=5) -> torch.Tensor:
    """
    Perform K-Nearest Neighbors (KNN) interpolation to complete sparse disparities.Use a batch-oriented 
    implementation of KNN interpolation to complete the sparse disparities. We leverages "torch_cluster.knn" 
    for acceleration and GPU memory efficiency.

    Args:
        sparse_disparities (torch.Tensor): Disparities for sparse map.
        pred_disparities (torch.Tensor): Dredicted disparities for sparse map points.
        complete_masks (torch.Tensor): Indicating which points in the complete map are valid.
        sparse_masks (torch.Tensor): Indicating which points in the sparse map are valid.
        K (int): The number of nearest neighbors to use for interpolation. Defaults to 5.

    Returns:
        The completed disparities, interpolated from the nearest neighbors.
    """
    
    # Use `knn_aligns` to find the K nearest neighbors and calculate distances.
    # 调用knn_aligns函数获取K近邻的距离、稀疏目标值和预测目标值
    bottomk_dists, k_sparse_targets, k_pred_targets = knn_aligns(
        sparse_disparities=sparse_disparities,
        pred_disparities=pred_disparities,
        sparse_masks=sparse_masks, K=K, 
        complete_masks=complete_masks
    )
    
    # 创建一个与sparse_disparities相同形状的全零张量用于存储缩放后的预测结果
    scaled_preds = torch.zeros_like(sparse_disparities, device=sparse_disparities.device, dtype=torch.float32)
    # 计算缩放和平移参数
    scale, shift = calc_scale_shift(
        k_sparse_targets=k_sparse_targets, k_pred_targets=k_pred_targets, 
        currk_dists=bottomk_dists, knn=True
    )
    # 对需要完成的区域应用缩放和平移变换
    complete_masks_depth = pred_disparities[complete_masks] * scale + shift
    # scaled_preds[complete_masks] = torch.clamp(complete_masks_depth, min=0)
    scaled_preds[complete_masks] = complete_masks_depth
    # 将原始稀疏视差值填充到稀疏点位置
    scaled_preds[sparse_masks] = sparse_disparities[sparse_masks]
    return scaled_preds,scale,shift
def recover_metric_depth(pred, gt, mask0):
    mask = (gt > 1e-8)
    if mask0 is not None and mask0.sum() > 0:
        mask0 = mask0 > 0
        mask = mask & mask0
        
    gt_mask = gt[mask]
    pred_mask = pred[mask]
    weight = 1.0 / gt_mask
    try:
        a, b = np.polyfit(x=pred_mask, y=gt_mask, w=weight, deg=1)
    except:
        a, b = 1.0, 0.0
        print(f"num of valid preds:{(pred > 1e-8).sum()}, num of valid gts:{(gt > 1e-8).sum()}")

    if a > 0:
        pred_metric = a * pred + b
    else:
        pred_mean = np.mean(pred_mask)
        gt_mean = np.mean(gt_mask)
        pred_metric = pred * (gt_mean / pred_mean)

    return pred_metric, a, b

def save_lidar(seq_save_dir):
    track_info, track_camera_visible, trajectory = load_track(seq_save_dir)
    extrinsics, intrinsics = load_calibration(seq_save_dir)
    print(f'Processing scene {seq_save_dir}...')
    print(f'Saving to {seq_save_dir}')

    os.makedirs(seq_save_dir, exist_ok=True)
    
    image_dir = os.path.join(seq_save_dir, 'images')
    lidar_depth_dir = os.path.join(seq_save_dir, 'lidar/depth')
    num_frames = len(sorted([os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith('.jpg')]))
    
    moge_dir = os.path.join(seq_save_dir, 'moge')
    os.makedirs(moge_dir, exist_ok=True)
    moge_dir_background = os.path.join(moge_dir, 'background')
    os.makedirs(moge_dir_background, exist_ok=True)
    moge_dir_actor = os.path.join(moge_dir, 'actor')
    os.makedirs(moge_dir_actor, exist_ok=True)
    lidar_dir_actor = os.path.join(seq_save_dir, 'lidar/actor')
    assert os.path.exists(lidar_dir_actor)
    
    pointcloud_actor = dict()
    # breakpoint()
    for track_id, traj in trajectory.items():
        dynamic = not traj['stationary']
        if dynamic and traj['label'] != 'sign':
            os.makedirs(os.path.join(moge_dir_actor, track_id), exist_ok=True)
            pointcloud_actor[track_id] = dict()
            pointcloud_actor[track_id]['xyz'] = []
            pointcloud_actor[track_id]['rgb'] = []
            pointcloud_actor[track_id]['mask'] = []
    
    for frame_id in tqdm(range(num_frames)):
        image_path = os.path.join(image_dir, f'{frame_id:06d}_0.jpg')
        lidar_depth_path = os.path.join(lidar_depth_dir, f'{frame_id:06d}_0.npz')
        lidar_depth = np.load(lidar_depth_path)
        lidar_depth_mask = lidar_depth['mask'].astype(np.bool_)
        lidar_depth_value = lidar_depth['value'].astype(np.float32)

        intrinsic = intrinsics[0]
        extrinsic = extrinsics[0]
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        
        # breakpoint()
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) / 255.
        image = torch.tensor(image).float().cuda().permute(2, 0, 1).unsqueeze(0) # (b, 3, h, w)
        _, _, orig_h, orig_w = image.shape
        area = orig_h * orig_w
        expected_area = 700 * 700
        expected_height, expected_width = int(orig_h * (expected_area / area) ** 0.5), int(orig_w * (expected_area / area) ** 0.5)
        image = torch.nn.functional.interpolate(image, (expected_height, expected_width), mode="bicubic", align_corners=False, antialias=True)        
        image = torch.clamp(image, 0, 1)

        fov_x = 2 * np.arctan(0.5 * orig_w / fx) / np.pi * 180

        output = moge_model.infer(image, fov_x=fov_x)
        # 从模型输出中获取预测的点云、深度图、内参和掩码
        pred_points, pred_depth, pred_intrinsics, pred_mask = output['points'], output['depth'], output['intrinsics'], output['mask']        
        # 将预测深度图插值到原始图像尺寸
        pred_depth_original = torch.nn.functional.interpolate(pred_depth.unsqueeze(1), (orig_h, orig_w), mode='bilinear', align_corners=False, antialias=False).squeeze(1)
        # 将深度图转换为numpy数组
        pred_depth_original = pred_depth_original.squeeze(0).cpu().numpy() # (h, w)
        # 找出深度图中的无效值(nan或inf)
        pred_depth_invalid = np.isnan(pred_depth_original) | np.isinf(pred_depth_original) 
        # 获取有效深度值的掩码
        pred_depth_valid = np.logical_not(pred_depth_invalid)

        # 创建与预测深度图相同大小的零矩阵作为真实深度图
        gt_depth = np.zeros_like(pred_depth_original).astype(np.float32)
        # 将激光雷达深度值填充到真实深度图中
        gt_depth[lidar_depth_mask] = lidar_depth_value
        sparse_masks = torch.tensor([pred_depth_valid])
        sparse_disparities=torch.tensor([gt_depth])
        pred_disparities=torch.tensor([pred_depth_original])
        K=5
        # 恢复预测深度图的度量尺度
        complete_masks = torch.ones_like(sparse_masks).to(torch.bool)
        complete_masks[sparse_masks] = False
        
        # 使用KNN对齐缩放预测视差
        scaled_preds,a,b  = kss_completer(
            sparse_disparities=sparse_disparities,
            pred_disparities=pred_disparities,
            sparse_masks=sparse_masks, K=K,
            complete_masks=complete_masks,
        )
        pred_depth_aligned = scaled_preds.squeeze(0).cpu().numpy()
        # pred_depth_aligned, a, b = recover_metric_depth(pred_depth_original, gt_depth, pred_depth_valid)
        # 将预测点云转换为numpy数组并重塑为N×3的形状
        # breakpoint()
        a_ori = torch.ones_like(scaled_preds)
        b_ori = torch.zeros_like(scaled_preds)
        a_ori[complete_masks] = a
        b_ori[complete_masks] = b 
        a = torch.nn.functional.interpolate(a_ori.unsqueeze(0), (expected_height, expected_width), mode='bilinear', align_corners=False, antialias=False).squeeze(0)
        b = torch.nn.functional.interpolate(b_ori.unsqueeze(0), (expected_height, expected_width), mode='bilinear', align_corners=False, antialias=False).squeeze(0)
        xyzs = (pred_points * a.unsqueeze(-1).cuda()  + b.unsqueeze(-1).cuda()).squeeze(0).cpu().numpy().reshape(-1, 3)
        # 将图像转换为numpy数组并重塑为N×3的形状
        rgbs = image.squeeze(0).permute(1, 2, 0).cpu().numpy().reshape(-1, 3)
        # 将预测掩码转换为一维numpy数组
        pred_mask = pred_mask.squeeze(0).cpu().numpy().reshape(-1)
        # 根据掩码筛选有效的点云坐标
        xyzs = xyzs[pred_mask]
        # 将点云从相机坐标系转换到车辆坐标系
        xyzs = transform_points_numpy(xyzs, extrinsic) # transform from camera space to vehicle space
        
        # 根据掩码筛选有效的RGB值
        rgbs = rgbs[pred_mask]
        # 创建全1的布尔掩码数组
        masks = np.ones_like(xyzs[:, 0]).astype(np.bool_)
        # 创建一个布尔数组用于标记属于actor的点云
        actor_mask = np.zeros(xyzs.shape[0], dtype=np.bool_)
        # 获取当前帧的track信息
        track_info_frame = track_info[f'{frame_id:06d}']
        # 遍历当前帧中的每个track
        for track_id, track_info_actor in track_info_frame.items():
            # 如果track_id不在pointcloud_actor中则跳过
            if track_id not in pointcloud_actor.keys():
                continue
            
            # 构建激光雷达actor点云文件路径
            ply_actor_path_lidar =  os.path.join(lidar_dir_actor, track_id, f'{frame_id:06d}.ply')
            # 如果文件不存在则跳过
            if not os.path.exists(ply_actor_path_lidar):
                continue            
            
            # 获取激光雷达检测框信息
            lidar_box = track_info_actor['lidar_box']
            height = lidar_box['height']
            width = lidar_box['width']
            length = lidar_box['length']
            # 获取当前帧在轨迹中的索引
            pose_idx = trajectory[track_id]['frames'].index(f"{frame_id:06d}")
            # 获取车辆坐标系下的位姿
            pose_vehicle = trajectory[track_id]['poses_vehicle'][pose_idx]

            # 将点云坐标转换为齐次坐标
            xyzs_homo = np.concatenate([xyzs, np.ones_like(xyzs[..., :1])], axis=-1)
            # 将点云从车辆坐标系转换到actor坐标系
            xyzs_actor = xyzs_homo @ np.linalg.inv(pose_vehicle).T
            xyzs_actor = xyzs_actor[..., :3]
        
            # 计算边界框的八个角点
            bbox = np.array([[-length, -width, -height], [length, width, height]]) * 0.5
            corners3d = bbox_to_corner3d(bbox)
            # 判断点云是否在边界框内
            inbbox_mask = inbbox_points(xyzs_actor, corners3d)
            
            # 更新actor掩码
            actor_mask = np.logical_or(actor_mask, inbbox_mask)
            
            # 如果边界框内的点数大于10
            if np.sum(inbbox_mask) > 10:
                
                # 提取边界框内的点云、颜色和掩码
                xyzs_inbbox = xyzs_actor[inbbox_mask]
                rgbs_inbbox = rgbs[inbbox_mask]
                masks_inbbox = masks[inbbox_mask]
                
                # 将点云添加到对应actor的字典中
                pointcloud_actor[track_id]['xyz'].append(xyzs_inbbox)
                pointcloud_actor[track_id]['rgb'].append(rgbs_inbbox)
                pointcloud_actor[track_id]['mask'].append(masks_inbbox)
                
                # 扩展掩码维度并保存点云
                masks_inbbox = masks_inbbox[..., None]
                ply_actor_path = os.path.join(moge_dir_actor, track_id, f'{frame_id:06d}.ply')
                storePly(ply_actor_path, xyzs_inbbox, rgbs_inbbox, masks_inbbox)
  
        # 提取背景点云(不属于任何actor的点)
        xyzs_background = xyzs[~actor_mask]
        rgbs_background = rgbs[~actor_mask]
        masks_background = masks[~actor_mask]
        # 扩展背景掩码维度
        masks_background = masks_background[..., None]
        # 构建背景点云保存路径
        ply_background_path = os.path.join(moge_dir_background, f'{frame_id:06d}.ply')
        
        # 保存背景点云
        storePly(ply_background_path, xyzs_background, rgbs_background, masks_background)

    for track_id, pointcloud in pointcloud_actor.items():
        try:
            xyzs = np.concatenate(pointcloud['xyz'], axis=0)
            rgbs = np.concatenate(pointcloud['rgb'], axis=0)
            masks = np.concatenate(pointcloud['mask'], axis=0)
            masks = masks[..., None]
            ply_actor_path_full = os.path.join(moge_dir_actor, track_id, 'full.ply')
            storePly(ply_actor_path_full, xyzs, rgbs, masks)
        except:
            pass # No pcd

def check_existing(scene_dir):
    image_dir = os.path.join(scene_dir, 'images')
    num_frames = len(os.listdir(image_dir)) 
    moge_background_dir = os.path.join(scene_dir, 'moge/background')
    num_pcds = len(os.listdir(moge_background_dir))
    if num_frames == num_pcds:
        return True
    else:
        return False
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/iag_ad_01/ad/yuanweizhong/huzeyu/street_crafter/data')
    parser.add_argument('--skip_existing', action='store_true')
    
    args = parser.parse_args()
    data_dir = args.data_dir
    # scene_ids = sorted([x for x in os.listdir(data_dir)])
    # for scene_id in scene_ids:
    #     print(f'Processing scene {scene_id}...')
    #     scene_dir = os.path.join(data_dir, scene_id)
    #     if args.skip_existing and check_existing(scene_dir):
    #         print(f'moge pcd exists for {scene_id}, skipping...')
    #         continue
    save_lidar(data_dir)
        
if __name__ == '__main__':
    main()