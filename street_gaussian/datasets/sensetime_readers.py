from easyvolcap.utils.console_utils import *
from street_gaussian.utils.sensetime_utils import generate_dataparser_outputs
from street_gaussian.utils.graphics_utils import focal2fov
from street_gaussian.utils.data_utils import get_val_frames
from street_gaussian.datasets.base_readers import CameraInfo, SceneInfo, getNerfppNorm
from street_gaussian.utils.novel_view_utils import waymo_novel_view_cameras
from street_gaussian.config import cfg
from PIL import Image
import os
import numpy as np
import cv2
import sys
import shutil
sys.path.append(os.getcwd())


def readSensetimeInfo(path, images='images', split_train=-1, split_test=-1, **kwargs):
    selected_frames = cfg.data.get('selected_frames', None)
    if cfg.debug:
        selected_frames = [0, 0]

    if cfg.data.get('load_pcd_from', False) and (cfg.mode == 'train'):
        load_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, 'input_ply')
        save_dir = os.path.join(cfg.model_path, 'input_ply')
        os.system(f'rm -rf {save_dir}')
        shutil.copytree(load_dir, save_dir)

        colmap_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, 'colmap')
        save_dir = os.path.join(cfg.model_path, 'colmap')
        os.system(f'rm -rf {save_dir}')
        shutil.copytree(colmap_dir, save_dir)

    # dynamic mask
    dynamic_mask_dir = os.path.join(path, 'dynamic_mask')
    load_dynamic_mask = True

    # sky mask
    sky_mask_dir = os.path.join(path, 'sky_mask')
    load_sky_mask = (cfg.mode == 'train')

    # lidar depth
    lidar_depth_dir = os.path.join(path, 'lidar/depth')
    load_lidar_depth = (cfg.mode == 'train')

    output = generate_dataparser_outputs(
        datadir=path,
        selected_frames=selected_frames,
        cameras=cfg.data.get('cameras', [0, 1, 2]),
    )

    exts = output['exts']
    ixts = output['ixts']
    ego_cam_poses = output['ego_cam_poses']
    ego_frame_poses = output['ego_frame_poses']
    image_filenames = output['image_filenames']
    obj_info = output['obj_info']
    frames, cams, frames_idx = output['frames'], output['cams'], output['frames_idx']
    cams_timestamps = output['cams_timestamps']
    cams_tracklets = output['cams_tracklets']

    num_frames = output['num_frames']
    train_frames, test_frames = get_val_frames(
        num_frames,
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )

    scene_metadata = dict()
    scene_metadata['camera_tracklets'] = cams_tracklets
    scene_metadata['obj_meta'] = obj_info
    scene_metadata['num_images'] = len(exts)
    scene_metadata['num_cams'] = len(cfg.data.cameras)
    scene_metadata['num_frames'] = num_frames
    scene_metadata['ego_frame_poses'] = ego_frame_poses
    scene_metadata['camera_timestamps'] = dict()
    for cam_idx in cfg.data.get('cameras'):
        scene_metadata['camera_timestamps'][cam_idx] = sorted([timestamp for i, timestamp in enumerate(cams_timestamps) if cams[i] == cam_idx])
        # scene_metadata['camera_timestamps'][cam_idx]['train_timestamps'] = \
        #     sorted([timestamp for i, timestamp in enumerate(cams_timestamps) if frames_idx[i] in train_frames and cams[i] == cam_idx])
        # scene_metadata['camera_timestamps'][cam_idx]['test_timestamps'] = \
        #     sorted([timestamp for i, timestamp in enumerate(cams_timestamps) if frames_idx[i] in test_frames and cams[i] == cam_idx])

    # make camera infos: train, test, novel view cameras
    # 初始化相机信息列表
    cam_infos = []
    # 遍历所有相机外参,准备相机和图像数据
    for i in tqdm(range(len(exts)), desc='Preparing cameras and images'):
        # 准备相机姿态和图像
        ext = exts[i]  # 获取当前相机的外参矩阵
        ixt = ixts[i]  # 获取当前相机的内参矩阵
        ego_pose = ego_cam_poses[i] # 获取自车相机姿态
        image_path = image_filenames[i]  # 获取图像路径
        image_name = os.path.basename(image_path).split('.')[0]  # 获取图像名称(不含扩展名)
        image = Image.open(image_path)  # 打开图像文件

        # 获取图像尺寸和相机参数
        width, height = image.size  # 获取图像宽高
        fx, fy = ixt[0, 0], ixt[1, 1]  # 获取相机焦距

        # 计算视场角
        FovY = focal2fov(fy, height)  # 计算垂直视场角
        FovX = focal2fov(fx, width)   # 计算水平视场角

        # 计算相机姿态
        c2w = ego_pose @ ext  # 计算相机到世界坐标系的变换矩阵
        RT = np.linalg.inv(c2w)  # 计算世界到相机坐标系的变换矩阵
        R = RT[:3, :3].T  # 获取旋转矩阵的转置
        T = RT[:3, 3]  # 获取平移向量
        K = ixt.copy()  # 复制内参矩阵

        # 构建元数据字典
        metadata = dict()
        metadata['frame'] = frames[i]  # 帧序号
        metadata['cam'] = cams[i]  # 相机编号
        metadata['frame_idx'] = frames_idx[i]  # 帧索引
        metadata['ego_pose'] = ego_pose  # 自车姿态
        metadata['extrinsic'] = ext  # 外参矩阵
        metadata['timestamp'] = cams_timestamps[i]  # 时间戳
        metadata['is_novel_view'] = False  # 是否为新视角
        guidance_dir = os.path.join(cfg.source_path, 'lidar', f'color_render')  # 引导数据目录
        metadata['guidance_rgb_path'] = os.path.join(guidance_dir, f'{str(frames[i]).zfill(6)}_{cams[i]}.png')  # RGB引导图路径
        metadata['guidance_mask_path'] = os.path.join(guidance_dir, f'{str(frames[i]).zfill(6)}_{cams[i]}_mask.png')  # 掩码引导图路径

        # 初始化引导数据字典
        guidance = dict()

        # 加载动态物体掩码
        if load_dynamic_mask:
            dynamic_mask_path = os.path.join(dynamic_mask_dir, f'{image_name}.jpg')  # 动态掩码路径
            obj_bound = (cv2.imread(dynamic_mask_path)[..., 0]) > 0.  # 读取并二值化动态物体边界
            guidance['obj_bound'] = Image.fromarray(obj_bound)  # 转换为PIL图像格式

        # 加载激光雷达深度图
        if load_lidar_depth:
            depth_path = os.path.join(lidar_depth_dir, f'{image_name}.npz')  # 深度图路径
            depth = np.load(depth_path)  # 加载深度数据
            mask = depth['mask'].astype(np.bool_)  # 深度掩码
            value = depth['value'].astype(np.float32)  # 深度值
            depth = np.zeros_like(mask).astype(np.float32)  # 创建深度图
            depth[mask] = value  # 填充有效深度值
            guidance['lidar_depth'] = depth  # 保存深度图

        # 加载天空掩码
        if load_sky_mask:
            sky_mask_path = os.path.join(sky_mask_dir, f'{image_name}.png')  # 天空掩码路径
            sky_mask = (cv2.imread(sky_mask_path)[..., 0]) > 0.  # 读取并二值化天空掩码
            guidance['sky_mask'] = Image.fromarray(sky_mask)  # 转换为PIL图像格式

        # 创建相机信息对象
        mask = None
        cam_info = CameraInfo(
            uid=i, R=R, T=T, FovY=FovY, FovX=FovX, K=K,
            image=image, image_path=image_path, image_name=image_name,
            width=width, height=height,
            metadata=metadata,
            guidance=guidance,
        )
        cam_infos.append(cam_info)  # 添加到相机信息列表

    # 分离训练集和测试集相机信息
    train_cam_infos = [cam_info for cam_info in cam_infos if cam_info.metadata['frame_idx'] in train_frames]  # 训练集相机
    test_cam_infos = [cam_info for cam_info in cam_infos if cam_info.metadata['frame_idx'] in test_frames]  # 测试集相机
    for cam_info in train_cam_infos:
        cam_info.metadata['is_val'] = False  # 标记训练集
    for cam_info in test_cam_infos:
        cam_info.metadata['is_val'] = True  # 标记测试集

    # 生成新视角相机
    print('making novel view cameras')
    novel_view_cam_infos = waymo_novel_view_cameras(cam_infos, ego_frame_poses, obj_info, cams_tracklets)

    # 获取场景范围
    # 1. 使用默认的nerf++设置
    if cfg.mode == 'novel_view':
        nerf_normalization = getNerfppNorm(novel_view_cam_infos)  # 使用新视角相机计算归一化参数
    else:
        nerf_normalization = getNerfppNorm(train_cam_infos)  # 使用训练集相机计算归一化参数

    # 2. 确保场景半径不会太小(至少为10)
    nerf_normalization['radius'] = max(nerf_normalization['radius'], 10)

    # 3. 如果配置中设置了场景范围,则使用配置值
    if cfg.data.get('extent', False):
        nerf_normalization['radius'] = cfg.data.extent

    # 4. 将场景半径写回配置
    cfg.data.extent = float(nerf_normalization['radius'])

    # 5. 将场景中心和半径写入场景元数据
    scene_metadata['scene_center'] = nerf_normalization['center']  # 保存场景中心
    scene_metadata['scene_radius'] = nerf_normalization['radius']  # 保存场景半径
    print(f'Scene extent: {nerf_normalization["radius"]}')  # 打印场景范围

    # 创建场景信息对象
    scene_info = SceneInfo(
        train_cameras=train_cam_infos,  # 训练集相机
        test_cameras=test_cam_infos,    # 测试集相机
        metadata=scene_metadata,         # 场景元数据
        novel_view_cameras=novel_view_cam_infos,  # 新视角相机
    )

    return scene_info
