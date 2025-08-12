import glob
import numpy as np
import os
import pickle
import cv2
import json
from collections import defaultdict
from typing import Dict, List, Literal, Tuple, Type
def image_filename_to_cam(x): return int(x.split('.')[0][-1])
def image_filename_to_frame(x): return int(x.split('.')[0][:6])
def file_path_to_frame_name(x): return x.split('.')[0][:6]


image_heights = [2160, 1280, 1280, 886, 886]
image_widths = [3840, 1920, 1920, 1920, 1920]

_camera2label = {
    'center_camera_fov120': '0', 
    'center_camera_fov30': '1',
    'left_front_camera': '2', 
    'right_front_camera': '3',
    'rear_camera': '4',
    'left_rear_camera': '5',
    'right_rear_camera': '6',
}

_label2camera = {
    '0':'center_camera_fov120', 
    '1':'center_camera_fov30',
    '2':'left_front_camera', 
    '3':'right_front_camera',
    '4':'rear_camera',
    '5':'left_rear_camera',
    '6':'right_rear_camera',
}

camera_names_dict = {
    'center_camera_fov120': '0', 
    # 'center_camera_fov30': '1',
    'left_front_camera': '2', 
    'right_front_camera': '3',
    # 'rear_camera': '4',
    # 'left_rear_camera': '5',
    # 'right_rear_camera': '6',
}

waymo_track2label = {"vehicle": 0, "pedestrian": 1, "cyclist": 2, "sign": 3, "misc": -1}
LANE_SHIFT_SIGN: Dict[str, Literal[-1, 1]] = defaultdict(lambda: -1)
LANE_SHIFT_SIGN.update(
    {
    "173": 1,
    "176": 1,
    "159": -1,
    "140": -1,
    "121": -1,
    "101": 1,
    "096": -1,
    "090": -1,
    "079": -1,
    "067": 1, 
    "062": -1,
    "051": -1,
    "049": -1,
    "035": -1,
    "027": -1,
    "020": -1,
    }
)

def get_object(object_list, name):
    """ Search for an object by name in an object list. """

    object_list = [obj for obj in object_list if obj.name == name]
    return object_list[0]

def load_sensor_sync(seq_save_dir):
    sensor_sync_path = os.path.join(seq_save_dir, 'sync_file.json')
    with open(sensor_sync_path, 'r') as f:
        sensor_sync = json.load(f)
    return sensor_sync

def load_track(seq_save_dir):
    track_dir = os.path.join(seq_save_dir, 'track')
    assert os.path.exists(track_dir), f"Track directory {track_dir} does not exist."

    track_info_path = os.path.join(track_dir, 'track_info.pkl')
    with open(track_info_path, 'rb') as f:
        track_info = pickle.load(f)

    track_camera_visible_path = os.path.join(track_dir, 'track_camera_visible.pkl')
    with open(track_camera_visible_path, 'rb') as f:
        track_camera_visible = pickle.load(f)

    trajectory_path = os.path.join(track_dir, 'trajectory.pkl')
    with open(trajectory_path, 'rb') as f:
        trajectory = pickle.load(f)

    return track_info, track_camera_visible, trajectory

# load ego pose and camera calibration(extrinsic and intrinsic)

def load_centered_vehicle_poses(datadir):
    ego_pose_dir = os.path.join(datadir, 'ego_pose')

    ego_frame_poses = []
    ego_frame_centered_poses = {}
 
    ego_pose_paths = sorted(os.listdir(ego_pose_dir))
    for ego_pose_path in ego_pose_paths:

        # frame pose
        if '_' not in ego_pose_path:
            ego_frame_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_frame_poses.append(ego_frame_pose)
            ego_frame_centered_poses[ego_pose_path.split('.')[0]] = ego_frame_pose

    # center ego pose
    ego_frame_poses = np.array(ego_frame_poses)
    center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)

    for item in ego_frame_centered_poses:
        ego_frame_centered_poses[item][:3, 3] -= center_point  # [4, 4]
   
    return ego_frame_centered_poses


def load_ego_poses(datadir):
    ego_pose_dir = os.path.join(datadir, 'ego_pose')

    ego_frame_poses = []
    ego_cam_poses = [[] for i in range(7)]
    ego_pose_paths = sorted(os.listdir(ego_pose_dir))
    for ego_pose_path in ego_pose_paths:

        # frame pose
        if '_' not in ego_pose_path:
            ego_frame_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_frame_poses.append(ego_frame_pose)
        else:
            cam = image_filename_to_cam(ego_pose_path)
            ego_cam_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_cam_poses[cam].append(ego_cam_pose)

    # center ego pose
    ego_frame_poses = np.array(ego_frame_poses)
    center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
    ego_frame_poses[:, :3, 3] -= center_point  # [num_frames, 4, 4]

    ego_cam_poses = [np.array(ego_cam_poses[i]) for i in range(7)]
    ego_cam_poses = np.array(ego_cam_poses)
    ego_cam_poses[:, :, :3, 3] -= center_point  # [5, num_frames, 4, 4]
    return ego_frame_poses, ego_cam_poses


def load_calibration(datadir):
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    assert os.path.exists(extrinsics_dir), f"{extrinsics_dir} does not exist"
    intrinsics_dir = os.path.join(datadir, 'intrinsics')
    assert os.path.exists(intrinsics_dir), f"{intrinsics_dir} does not exist"

    intrinsics = {}
    extrinsics = {}

    for camera_name in camera_names_dict:
        intrinsic_path = os.path.join(intrinsics_dir, camera_names_dict[camera_name]+".txt")
        intrinsic = np.loadtxt(intrinsic_path)
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics[camera_name] = intrinsic
        extrinsic_path = os.path.join(extrinsics_dir, camera_names_dict[camera_name]+".txt")
        cam_to_ego = np.loadtxt(extrinsic_path)
        extrinsics[camera_name] = cam_to_ego

    return extrinsics, intrinsics


# load ego pose and camera calibration(extrinsic and intrinsic)
def load_camera_info(datadir):
    ego_pose_dir = os.path.join(datadir, 'ego_pose')
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    intrinsics_dir = os.path.join(datadir, 'intrinsics')

    intrinsics = []
    extrinsics = []
    for i in range(5):
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir, f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)

    for i in range(5):
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir, f"{i}.txt"))
        extrinsics.append(cam_to_ego)

    ego_frame_poses = []
    ego_cam_poses = [[] for i in range(5)]
    ego_pose_paths = sorted(os.listdir(ego_pose_dir))
    for ego_pose_path in ego_pose_paths:

        # frame pose
        if '_' not in ego_pose_path:
            ego_frame_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_frame_poses.append(ego_frame_pose)
        else:
            cam = image_filename_to_cam(ego_pose_path)
            ego_cam_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_cam_poses[cam].append(ego_cam_pose)

    # center ego pose
    ego_frame_poses = np.array(ego_frame_poses)
    center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
    ego_frame_poses[:, :3, 3] -= center_point  # [num_frames, 4, 4]

    ego_cam_poses = [np.array(ego_cam_poses[i]) for i in range(5)]
    ego_cam_poses = np.array(ego_cam_poses)
    ego_cam_poses[:, :, :3, 3] -= center_point  # [5, num_frames, 4, 4]
    return intrinsics, extrinsics, ego_frame_poses, ego_cam_poses


opencv2camera = np.array([[0., 0., 1., 0.],
                          [-1., 0., 0., 0.],
                          [0., -1., 0., 0.],
                          [0., 0., 0., 1.]])


def get_extrinsic(camera_calibration):
    camera_extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4, 4)  # camera to vehicle
    extrinsic = np.matmul(camera_extrinsic, opencv2camera)  # [forward, left, up] to [right, down, forward]
    return extrinsic


def get_intrinsic(camera_calibration):
    camera_intrinsic = camera_calibration.intrinsic
    fx = camera_intrinsic[0]
    fy = camera_intrinsic[1]
    cx = camera_intrinsic[2]
    cy = camera_intrinsic[3]
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return intrinsic


def project_label_to_image(dim, obj_pose, calibration):
    from utils.base_utils import project_numpy
    from utils.box_utils import bbox_to_corner3d, get_bound_2d_mask
    bbox_l, bbox_w, bbox_h = dim
    bbox = np.array([[-bbox_l, -bbox_w, -bbox_h], [bbox_l, bbox_w, bbox_h]]) * 0.5
    points = bbox_to_corner3d(bbox)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points_vehicle = points @ obj_pose.T  # 3D bounding box in vehicle frame
    extrinsic = calibration.extrinsic
    intrinsic = calibration.intrinsic
    width, height = calibration.width, calibration.height
    points_uv, valid = project_numpy(
        xyz=points_vehicle[..., :3],
        K=intrinsic,
        RT=np.linalg.inv(extrinsic),
        H=height, W=width
    )
    return points_uv, valid


def project_label_to_mask(dim, obj_pose, calibration):
    from utils.box_utils import bbox_to_corner3d, get_bound_2d_mask
    bbox_l, bbox_w, bbox_h = dim
    bbox = np.array([[-bbox_l, -bbox_w, -bbox_h], [bbox_l, bbox_w, bbox_h]]) * 0.5
    points = bbox_to_corner3d(bbox)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points_vehicle = points @ obj_pose.T  # 3D bounding box in vehicle frame

    extrinsic = calibration.extrinsic
    intrinsic = calibration.intrinsic
    width, height = calibration.width, calibration.height

    mask = get_bound_2d_mask(
        corners_3d=points_vehicle[..., :3],
        K=intrinsic,
        pose=np.linalg.inv(extrinsic),
        H=height, W=width
    )

    return mask

def draw_3d_box_on_img(vertices, img, color=(255, 128, 128), thickness=1):
    # Draw the edges of the 3D bounding box
    for k in [0, 1]:
        for l in [0, 1]:
            for idx1, idx2 in [((0, k, l), (1, k, l)), ((k, 0, l), (k, 1, l)), ((k, l, 0), (k, l, 1))]:
                cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), color, thickness)

    # Draw a cross on the front face to identify front & back.
    for idx1, idx2 in [((1, 0, 0), (1, 1, 1)), ((1, 1, 0), (1, 0, 1))]:
        cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), color, thickness)

def get_lane_shift_direction(ego_frame_poses, frame, frame_name_list):
    assert frame >= 0 and frame < len(ego_frame_poses)
    if frame == 0:
        prev_frame_name = frame_name_list[0]
        curr_frame_name = frame_name_list[1]
        ego_pose_delta = ego_frame_poses[curr_frame_name][:3, 3] - ego_frame_poses[prev_frame_name][:3, 3]
    else:
        prev_frame_name = frame_name_list[frame - 1]
        curr_frame_name = frame_name_list[frame]
        ego_pose_delta = ego_frame_poses[curr_frame_name][:3, 3] - ego_frame_poses[prev_frame_name][:3, 3]

    ego_pose_delta = ego_pose_delta[:2]  # x, y
    ego_pose_delta /= np.linalg.norm(ego_pose_delta)
    direction = np.array([ego_pose_delta[1], -ego_pose_delta[0], 0])  # y, -x
    return direction
