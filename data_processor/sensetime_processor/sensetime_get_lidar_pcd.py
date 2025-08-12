import os
import numpy as np
import cv2
import math
import argparse
import sys
import open3d as o3d
from PIL import Image
from tqdm import tqdm


sys.path.append(os.getcwd())
from sensetime_helpers import load_calibration, load_track, get_object, load_centered_vehicle_poses, load_sensor_sync, camera_names_dict
from utils.img_utils import visualize_depth_numpy
from utils.box_utils import bbox_to_corner3d, inbbox_points
from utils.pcd_utils import storePly, fetchPly, storePlyXYZ, storePlyXYZRGB
from utils.base_utils import project_numpy

from typing import Dict, List, Optional, Tuple, TypedDict

np.set_printoptions(precision=4, suppress=True)
from copy import deepcopy

# only copied center_cemera_fov120 to the traget dir, so use a temp camera_name_dict
camera_names_dict_1 = {
    'center_camera_fov120': '0', 
    # 'center_camera_fov30': '1',
    # 'left_front_camera': '2', 
    # 'right_front_camera': '3',
    # 'rear_camera': '4',
    # 'left_rear_camera': '5',
    # 'right_rear_camera': '6',
}

def overlay_depth_on_image(image, depth, save_path, alpha=0.6):
    """
    将深度图叠加在 RGB 图像上。
    
    参数:
        image: 原始 RGB 图像，(H, W, 3)，uint8
        depth: 深度图，(H, W)，float32，单位为 meter，0 表示无效
        alpha: 叠加强度（0-1），越大越偏向深度图
        
    返回:
        叠加后的 RGB 图像，uint8
    """
    depth_vis = np.zeros_like(image)

    # 有效深度掩码（非零）
    valid_mask = depth > 0
    if np.sum(valid_mask) == 0:
        return image

    # 将深度值归一化到 0-255
    d_min = np.percentile(depth[valid_mask], 1)
    d_max = np.percentile(depth[valid_mask], 99)
    depth_clip = np.clip((depth - d_min) / (d_max - d_min), 0, 1)
    depth_uint8 = (depth_clip * 255).astype(np.uint8)

    # 映射为伪彩色
    color_map = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

    # 使用掩码进行叠加
    depth_vis[valid_mask] = color_map[valid_mask]
    blended = image.copy()
    blended[valid_mask] = cv2.addWeighted(image[valid_mask], 1 - alpha, depth_vis[valid_mask], alpha, 0)

    cv2.imwrite(save_path, blended)

def colorize_point_cloud_with_occlusion(
    points_car,                # (N, 3)
    images,                    # list of HWC uint8
    intrinsics,                # list of (3, 3)
    extrinsics,                # list of (4, 4)
):
    N = points_car.shape[0]
    colors = np.zeros((N, 3), dtype=np.float32)
    z_buffer = np.full(N, np.inf)
    mask = np.zeros(N, dtype=bool)
    depths = dict()

    points_hom = np.hstack([points_car, np.ones((N, 1))])  # (N, 4)

    for camera_name in camera_names_dict_1:
        view_image = images[camera_name]
        view_intrinsic = intrinsics[camera_name]
        T_vc = extrinsics[camera_name]
        T_cv = np.linalg.inv(T_vc)

        H, W, _ = view_image.shape
        H_mask = H
        if camera_name == 'center_camera_fov120':
            H_mask = int(H*0.66)
        depth = np.ones((H, W), dtype=np.float32) * np.finfo(np.float32).max  # 初始化为无穷大
        
        p_cam = (T_cv @ points_hom.T).T
        xyz_cam = p_cam[:,:3]
        z = xyz_cam[:,2]
        valid = z > 0

        uv = (view_intrinsic @ (xyz_cam[valid] / z[valid,None]).T).T
        u = uv[:, 0].astype(np.int32)
        v = uv[:, 1].astype(np.int32)

        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H_mask)
        indices = np.where(valid)[0][inside]
        u, v = u[inside], v[inside]
        z_sel = z[valid][inside]

        # 用最小深度填充
        flat_indices = v * W + u
        depth_flat = depth.reshape(-1)
        np.minimum.at(depth_flat, flat_indices, z_sel)
        depth = depth_flat.reshape(H, W)
        depth[depth >= np.finfo(np.float32).max - 1e-3] = 0  # 无效值设为0
        depths[camera_name] = depth

        # only update if this view sees point closer than previous
        closer = z_sel < z_buffer[indices]
        indices = indices[closer]
        u = u[closer]
        v = v[closer]
        z_buffer[indices] = z_sel[closer]

        sampled_colors = view_image[v, u] / 255.0
        colors[indices] = sampled_colors
        mask[indices] = True
        
    return colors, mask, depths

   

def save_lidar(root_dir, seq_path, seq_save_dir):
    track_info, track_camera_visible, trajectory = load_track(seq_save_dir)
    extrinsics, intrinsics = load_calibration(seq_save_dir)
    sensor_sync = load_sensor_sync(seq_save_dir)
    ego_frame_poses = load_centered_vehicle_poses(seq_save_dir)
    print(f'Processing sequence {seq_path}...')
    print(f'Saving to {seq_save_dir}')

    os.makedirs(seq_save_dir, exist_ok=True)
    
    image_dir = os.path.join(seq_save_dir, 'images')
    lidar_dir = os.path.join(seq_save_dir, 'lidar')
    os.makedirs(lidar_dir, exist_ok=True)
    lidar_dir_background = os.path.join(lidar_dir, 'background')
    os.makedirs(lidar_dir_background, exist_ok=True)
    lidar_dir_actor = os.path.join(lidar_dir, 'actor')
    os.makedirs(lidar_dir_actor, exist_ok=True)
    lidar_dir_depth = os.path.join(lidar_dir, 'depth')
    os.makedirs(lidar_dir_depth, exist_ok=True)

    pvb_lidar_dir = os.path.join(seq_path, 'slam_output/map_generation_results/pvb_vehicle')
    bgr_lidar_dir = os.path.join(seq_path, 'slam_output/map_generation_results/top_center_lidar_vehicle')

    pointcloud_actor = dict()
    for track_id, traj in trajectory.items():
        # dynamic = not traj['stationary'] # modified
        dynamic = True
        if dynamic and traj['label'] != 'sign':
            os.makedirs(os.path.join(lidar_dir_actor, track_id), exist_ok=True)
            pointcloud_actor[track_id] = dict()
            pointcloud_actor[track_id]['xyz'] = []
            pointcloud_actor[track_id]['rgb'] = []
            pointcloud_actor[track_id]['mask'] = []
    
    print("Processing LiDAR data...")
    bgr_full_points = []
    bgr_full_rgbs = []
    bgr_full_masks = []
    for frame_id in tqdm(list(sensor_sync.keys())):
        sync_sensor_path = sensor_sync[frame_id]
        lidar_ts = os.path.basename(sync_sensor_path['top_center_lidar']).split('.')[0]
        lidar_ts_modified = lidar_ts[:-6] + '0' * 6
        bgr_lidar_path = os.path.join(bgr_lidar_dir, lidar_ts_modified+'.pcd')

        bgr_lidar = o3d.io.read_point_cloud(bgr_lidar_path)
        bgr_points = np.asarray(bgr_lidar.points)

        images = dict()
        for camera_name in camera_names_dict_1:
            img_path = os.path.join(image_dir, frame_id+'_'+camera_names_dict_1[camera_name]+'.jpg')
            img = np.array(Image.open(img_path)) 
            images[camera_name] = img

        # color bgr point cloud
        bgr_rgb, bgr_mask, depths = colorize_point_cloud_with_occlusion(bgr_points, images, intrinsics, extrinsics)
        bgr_full_rgbs.append(bgr_rgb)
        bgr_full_masks.append(bgr_mask)

        ego_pose = ego_frame_poses[frame_id]
        ego_pose_R = ego_pose[:3,:3]
        ego_pose_t = ego_pose[:3,3:]
        bgr_points_trans = ego_pose_R @ bgr_points.T + ego_pose_t
        bgr_full_points.append(bgr_points_trans.T)
        storePly(os.path.join(lidar_dir_background, frame_id+'.ply'), bgr_points, bgr_rgb, bgr_mask[:,None])

        # save depth
        for camera_name in camera_names_dict_1:
            depth = depths[camera_name]
            depth_path = os.path.join(lidar_dir_depth, frame_id+'_'+camera_names_dict_1[camera_name]+'.npz')
            valid_mask = depth > 0
            valid_depth = depth[valid_mask]
            np.savez_compressed(depth_path, value=valid_depth, mask=valid_mask)
            # depth_viz_path = os.path.join(lidar_dir_depth, frame_id+'_'+camera_names_dict_1[camera_name]+'.png')
            # overlay_depth_on_image(images[camera_name], depth, depth_viz_path)

        # color pvb point cloud 
        pvb_lidar_path = os.path.join(pvb_lidar_dir, lidar_ts_modified+'.pcd')
        pvb_lidar = o3d.io.read_point_cloud(pvb_lidar_path)
        pvb_points = np.asarray(pvb_lidar.points)
        pvb_rgb, pvb_mask, pvb_depth = colorize_point_cloud_with_occlusion(pvb_points, images, intrinsics, extrinsics)

        track_info_frame = track_info[frame_id]
        actor_mask = np.zeros(pvb_points.shape[0], dtype=np.bool_)
        for track_id, track_info_actor in track_info_frame.items():
            if track_id not in pointcloud_actor.keys():
                continue
            
            lidar_box = track_info_actor['lidar_box']
            height = lidar_box['height']
            width = lidar_box['width']
            length = lidar_box['length']
            pose_idx = trajectory[track_id]['frames'].index(frame_id)
            pose_vehicle = trajectory[track_id]['poses_vehicle'][pose_idx]

            xyzs_homo = np.concatenate([pvb_points, np.ones_like(pvb_points[..., :1])], axis=-1)
            xyzs_actor = xyzs_homo @ np.linalg.inv(pose_vehicle).T
            xyzs_actor = xyzs_actor[..., :3]
            
            bbox = np.array([[-length, -width, -height], [length, width, height]]) * 0.5
            corners3d = bbox_to_corner3d(bbox)
            inbbox_mask = inbbox_points(xyzs_actor, corners3d)
            
            actor_mask = np.logical_or(actor_mask, inbbox_mask)
            
            xyzs_inbbox = xyzs_actor[inbbox_mask]
            rgbs_inbbox = pvb_rgb[inbbox_mask]
            masks_inbbox = pvb_mask[inbbox_mask]
            
            pointcloud_actor[track_id]['xyz'].append(xyzs_inbbox)
            pointcloud_actor[track_id]['rgb'].append(rgbs_inbbox)
            pointcloud_actor[track_id]['mask'].append(masks_inbbox)
            
            masks_inbbox = masks_inbbox[..., None]
            ply_actor_path = os.path.join(lidar_dir_actor, track_id, frame_id+'.ply')
            try:
                storePly(ply_actor_path, xyzs_inbbox, rgbs_inbbox, masks_inbbox)
            except:
                pass # No pcd


    bgr_full_points_path = os.path.join(lidar_dir_background, 'full.ply')
    bgr_full_points_np = np.concatenate(bgr_full_points, axis=0)
    bgr_full_rgb_np = np.concatenate(bgr_full_rgbs, axis=0)
    bgr_full_masks_np = np.concatenate(bgr_full_masks, axis=0)
    # down sample
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(bgr_full_points_np)
    pcd.colors = o3d.utility.Vector3dVector(bgr_full_rgb_np)
    voxel_size = 0.20  # 20 cm
    pcd_down = pcd.voxel_down_sample(voxel_size)
    points_down = np.asarray(pcd_down.points)
    colors_down = np.asarray(pcd_down.colors)
    # storePlyXYZ(bgr_full_points_path, points_down)
    storePlyXYZRGB(bgr_full_points_path, points_down, colors_down)

    for track_id, pointcloud in pointcloud_actor.items():
        xyzs = np.concatenate(pointcloud['xyz'], axis=0)
        rgbs = np.concatenate(pointcloud['rgb'], axis=0)
        masks = np.concatenate(pointcloud['mask'], axis=0)
        masks = masks[..., None]
        ply_actor_path_full = os.path.join(lidar_dir_actor, track_id, 'full.ply')
        
        try:
            storePly(ply_actor_path_full, xyzs, rgbs, masks)
        except:
            pass # No pcd

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/iag_ad_01/ad/yuanweizhong/datasets/senseauto/2024_09_08_07_53_23_pathway_pilotGtParser')
    parser.add_argument('--save_dir', type=str, default='/iag_ad_01/ad/yuanweizhong/huzeyu/street_crafter/data')
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()
    
    root_dir = args.root_dir
    save_dir = args.save_dir
                
    save_lidar(
        root_dir=root_dir,
        seq_path=root_dir,
        seq_save_dir=save_dir,
    )

    
if __name__ == '__main__':
    main()