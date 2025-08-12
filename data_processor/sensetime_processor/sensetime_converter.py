import pickle
import io
import os
import numpy as np
import cv2
import imageio
import sys
import math
import argparse
import json
import pickle
from PIL import Image
from tqdm import tqdm

sys.path.append(os.getcwd())

from sensetime_helpers import load_track, load_ego_poses, load_centered_vehicle_poses, get_object, \
    project_label_to_image, project_label_to_mask, draw_3d_box_on_img, opencv2camera, camera_names_dict
from pose_helper import parse_txt_to_poses, get_pose_at_time

# from petrel_client.client import Client
from download_interface import download_file, upload_file

# petrel_conf_path = '/iag_ad_01/ad/lijianfeng/petreloss_local.conf'
# backend = Client(conf_path=petrel_conf_path)
print('petrel backend init done')

class Calibration:
    def __init__(self, extrinsic, intrinsic, width, height):
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.width = width
        self.height = height

def parse_seq_rawdata(process_list, seq_path, seq_save_dir, skip_existing):
    print(f'Processing sequence {seq_path}...')
    print(f'Saving to {seq_save_dir}')
    os.makedirs(seq_save_dir, exist_ok=True)

    s3_prefix_path = os.path.join(seq_path, 's3_prefix.txt')
    with open(s3_prefix_path, 'r') as f:
        s3_prefix = f.readline()
        s3_prefix = s3_prefix.strip()

    # camera calibration 
    intrinsic_save_dir = os.path.join(seq_save_dir, 'intrinsics')
    extrinsic_save_dir = os.path.join(seq_save_dir, 'extrinsics')
    extrinsic_map = dict()
    intrinsic_map = dict()
    calibration_map = dict()
    if 'calib' not in process_list:
        print("Skipping calibration processing...")
    elif os.path.exists(intrinsic_save_dir) and os.path.exists(extrinsic_save_dir) and skip_existing:
        print('Calibration already exists, skipping...')
    else:
        os.makedirs(intrinsic_save_dir, exist_ok=True)
        os.makedirs(extrinsic_save_dir, exist_ok=True)
        print("Processing camera calibration...")
        
        for camera_name in camera_names_dict:
            img_width = 0
            img_height = 0
            cam_extrinsic_path = os.path.join(seq_path, 'calib', camera_name, camera_name+'-to-car_center-extrinsic.json')
            with open(cam_extrinsic_path, 'r') as cam_extrinsic_file:
                cam_extrinsic = json.load(cam_extrinsic_file)
                for key in cam_extrinsic:
                    extrinsic = np.array(cam_extrinsic[key]['param']['sensor_calib']['data'])
                    extrinsic_map[camera_name] = extrinsic
                    np.savetxt(os.path.join(extrinsic_save_dir, camera_names_dict[camera_name]+".txt"), extrinsic)

            cam_intrinsic_path = os.path.join(seq_path, 'calib', camera_name, camera_name+'-intrinsic.json')
            with open(cam_intrinsic_path) as cam_intrinsic_file:
                cam_intrinsic = json.load(cam_intrinsic_file)
                for key in cam_intrinsic:
                    intrinsic = np.array(cam_intrinsic[key]['param']['cam_K_new']['data'])
                    img_width = cam_intrinsic[key]['param']['img_dist_w']
                    img_height = cam_intrinsic[key]['param']['img_dist_h']
                    intrinsic_map[camera_name] = intrinsic
                    intrinsic_flat = [intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2], 0,0,0,0,0]
                    with open(os.path.join(intrinsic_save_dir, camera_names_dict[camera_name]+".txt"), 'w') as f:
                        for val in intrinsic_flat:
                            f.write(f'{val}\n')

            calibration = Calibration(extrinsic_map[camera_name], intrinsic_map[camera_name], img_width, img_height)
            calibration_map[camera_name] = calibration

    sync_files = dict()
    # frame timestamps
    timestamp = dict()
    timestamp['FRAME'] = dict()
    for camera_name in camera_names_dict:
        timestamp[camera_name] = dict()
    sensor_sync_file_path = os.path.join(seq_path, 'sensor_temporal_alignment.json')
    with open(sensor_sync_file_path, 'r') as sensor_sync_file:
        sensor_sync = json.load(sensor_sync_file)

        for frame_id, sensor_frame in tqdm(enumerate(sensor_sync)):
            sync_file = dict()
            for key in sensor_frame:
                timestamp['FRAME'][str(frame_id).zfill(6)] = float(key)/1e6
                lidar_file_rel_path = sensor_frame[key]['top_center_lidar']
                sync_file['top_center_lidar'] = lidar_file_rel_path

                for camera_name in camera_names_dict:
                    cam_rel_path = sensor_frame[key][camera_name]
                    cam_ts = float(os.path.basename(cam_rel_path).split('.')[0])/1e6
                    cam_file_name = str(frame_id).zfill(6)+'_'+camera_names_dict[camera_name]
                    sync_file[camera_name] = cam_rel_path
                    timestamp[camera_name][cam_file_name] = cam_ts

            sync_files[str(frame_id).zfill(6)] = sync_file
            

    # ego pose
    ego_vehicle_poses = dict()
    ego_camera_poses = dict()
    pose_file_path = os.path.join(seq_path, 'slam_output/map_generation_results/backend-pose-vehicle.txt')
    vehicle_poses = parse_txt_to_poses(pose_file_path)
    ego_pose_save_dir = os.path.join(seq_save_dir, 'ego_pose')
    os.makedirs(ego_pose_save_dir, exist_ok=True)
    print("Processing ego pose...")
    for frame_name in tqdm(list(sync_files.keys())):
        frame_ts = timestamp['FRAME'][frame_name]
        lidar_pose = get_pose_at_time(vehicle_poses, frame_ts)
        # np.savetxt(os.path.join(ego_pose_save_dir, frame_name+".txt"), lidar_pose)
        ego_vehicle_poses[frame_name] = lidar_pose
        for camera_name in camera_names_dict:
            camera_file_path = sync_files[frame_name][camera_name]
            camera_file_name = os.path.basename(camera_file_path).split('.')[0]
            camera_ts = float(camera_file_name)/1e6
            camera_pose = get_pose_at_time(vehicle_poses, camera_ts) @ extrinsic_map[camera_name]
            # np.savetxt(os.path.join(ego_pose_save_dir, frame_name+"_"+camera_names_dict[camera_name]+".txt"), camera_pose)
            ego_camera_poses[frame_name+"_"+camera_names_dict[camera_name]] = camera_pose
    # ego_vehicle_poses = load_centered_vehicle_poses(seq_save_dir)


    # select key frame
    key_frame_trans_thrsh = 0.3
    first_key_frame_name = list(sync_files.keys())[0]
    last_vehicle_pose = ego_vehicle_poses[first_key_frame_name]
    key_frame_sync_files = dict()
    key_frame_sync_files[first_key_frame_name] = sync_files[first_key_frame_name]
    for frame_name in sync_files.keys():
        curr_pose = ego_vehicle_poses[frame_name]
        delta_trans = np.linalg.norm(curr_pose[:3,3] - last_vehicle_pose[:3,3])
        if delta_trans > key_frame_trans_thrsh:
            key_frame_sync_files[frame_name] = sync_files[frame_name]
            last_vehicle_pose = curr_pose
    

    # update key frame and timestamps and ego poses
    frame_name_list = list(key_frame_sync_files.keys())
    frame_name_list.sort()
    updated_sync_files = dict()
    updated_ego_vehicle_poses = dict()
    updated_ego_camera_poses = dict()
    updated_timestamps = dict()
    updated_timestamps['FRAME'] = dict()
    for camera_name in camera_names_dict:
        updated_timestamps[camera_name] = dict()
    for frame_index, frame_name in enumerate(frame_name_list):
        new_frame_name = str(frame_index).zfill(6)
        updated_sync_files[new_frame_name] = key_frame_sync_files[frame_name]
        updated_ego_vehicle_poses[new_frame_name] = ego_vehicle_poses[frame_name]
        updated_timestamps['FRAME'][new_frame_name] = timestamp['FRAME'][frame_name]
        np.savetxt(os.path.join(ego_pose_save_dir, new_frame_name+".txt"), updated_ego_vehicle_poses[new_frame_name])

        for camera_name in camera_names_dict:
            pc_name = frame_name+'_'+camera_names_dict[camera_name]
            new_pc_name = new_frame_name+'_'+camera_names_dict[camera_name]
            updated_ego_camera_poses[new_pc_name] = ego_camera_poses[pc_name]
            updated_timestamps[camera_name][new_pc_name] = timestamp[camera_name][pc_name]
            np.savetxt(os.path.join(ego_pose_save_dir, new_pc_name+".txt"), updated_ego_camera_poses[new_pc_name])

    sync_file_save_path = os.path.join(seq_save_dir, "sync_file.json")
    with open(sync_file_save_path, 'w') as f:
        json.dump(updated_sync_files, f, indent=1) 
    
    timestamp_save_path = os.path.join(seq_save_dir, "timestamps.json")
    with open(timestamp_save_path, 'w') as f:
        json.dump(updated_timestamps, f, indent=1)


    # # images
    image_save_dir = os.path.join(seq_save_dir, 'images')
    if 'image' not in process_list:
        print("Skipping image processing...")
    elif os.path.exists(image_save_dir) and skip_existing:
        print('Images already exist, skipping...')
    else:
        os.makedirs(image_save_dir, exist_ok=True)      
        print("Processing image data...")

        camera_timestamp = dict()
        for frame_name in tqdm(list(updated_sync_files.keys())):
            for camera_name in camera_names_dict:
                camera_file_path = updated_sync_files[frame_name][camera_name]
                camera_file_name = os.path.basename(camera_file_path).split('.')[0]
                camera_ts = float(camera_file_name)/1e6
                
                s3_path = 'ad_system_common_hs:'+os.path.join(s3_prefix, camera_file_path)
                img_save_path = os.path.join(image_save_dir, frame_name+'_'+camera_names_dict[camera_name]+'.jpg')
                if not os.path.exists(img_save_path):
                    download_file(s3_path, img_save_path, backend, strict=True)

                camera_timestamp[frame_name+'_'+camera_names_dict[camera_name]] = camera_ts
                updated_sync_files[frame_name][camera_name] = img_save_path
        
        camera_timestamp_save_path = os.path.join(seq_save_dir, 'images', 'timestamps.json')
        with open(camera_timestamp_save_path, 'w') as f:
            json.dump(camera_timestamp, f, indent=1)

    # trajectory
    track_dir = os.path.join(seq_save_dir, "track")
    if 'track' not in process_list:
        print("Skipping tracking data processing...")
    elif os.path.exists(track_dir) and skip_existing:
        print('Tracking data already exists, skipping...')
    else:
        os.makedirs(track_dir, exist_ok=True)
        print("Processing tracking data...")

        track_info = dict() # 以每个frame的一个bbox为单位 frame_id, track_id 记录LiDAR-synced和Camera_synced bboxes
        track_camera_visible = dict() # 以每个camera的一个bbox为单位 frame_id, camera_id, track_id 记录这个camera看到了哪些物体
        trajectory_info = dict() # 以每个track物体的一个bbox为单位 track_id, frame_id 记录LiDAR-synced boxes
        object_ids = dict() # 每个物体的track_id对应一个数字 （track_id, object_id）之后streetgaussian训练时用的是object_id
        track_vis_imgs = []

        lidar_obj_path = os.path.join(seq_path, 'gt_labels', 'car_center-to-car_center#object.pkl')
        lidar_obj_map = dict()
        with open(lidar_obj_path, 'rb') as lidar_obj_file:
            lidar_obj = pickle.load(lidar_obj_file)
            for frame_objs in lidar_obj['frames']:
                file_name = os.path.basename(frame_objs['filename'])
                lidar_obj_map[file_name] = frame_objs['objects']

        
        for frame_name in tqdm(list(updated_sync_files.keys())):
            frame_ts = updated_timestamps['FRAME'][frame_name]
            lidar_file_name = os.path.basename(updated_sync_files[frame_name]['top_center_lidar'])
            
            track_info_cur_frame = dict()
            track_camera_visible_cur_frame = dict()
            images = dict()

            for camera_name in camera_names_dict.keys():
                img_path = updated_sync_files[frame_name][camera_name]
                img = np.array(Image.open(img_path)) 
                images[camera_name] = img
                track_camera_visible_cur_frame[camera_name] = []


            frame_lidar_objs = lidar_obj_map[lidar_file_name]
            for lidar_obj in frame_lidar_objs:
                if 'VEHICLE' in lidar_obj['label']:
                    obj_class = "vehicle"
                elif 'PEDESTRIAN' in lidar_obj['label']:
                    obj_class = "pedestrian"
                elif 'SIGN' in lidar_obj['label']:
                    obj_class = "sign"
                elif 'BIKE' in lidar_obj['label']:
                    obj_class = "cyclist"
                else:
                    obj_class = "misc"

                # speed_x = 0.0 if lidar_obj['vel'] is None else lidar_obj['vel'][]
                speed = np.linalg.norm([0.0, 0.0])  
                label_id = str(lidar_obj['id'])

                # Add one label
                if label_id not in trajectory_info:
                    trajectory_info[label_id] = dict()
                    
                if label_id not in object_ids:
                    object_ids[label_id] = int(label_id)

                track_info_cur_frame[label_id] = dict()

                lidar_bbox3d = lidar_obj['bbox3d']

                # LiDAR-synced box
                lidar_synced_box = dict()
                lidar_synced_box['height'] = lidar_bbox3d[3]
                lidar_synced_box['width'] = lidar_bbox3d[4]
                lidar_synced_box['length'] = lidar_bbox3d[5]
                lidar_synced_box['center_x'] = lidar_bbox3d[0]
                lidar_synced_box['center_y'] = lidar_bbox3d[1]
                lidar_synced_box['center_z'] = lidar_bbox3d[2]
                lidar_synced_box['heading'] = lidar_bbox3d[8]
                lidar_synced_box['label'] = obj_class
                lidar_synced_box['speed'] = speed
                lidar_synced_box['timestamp'] = frame_ts
                if obj_class == 'vehicle':
                    lidar_synced_box['height'] = lidar_bbox3d[4]
                    lidar_synced_box['width'] = lidar_bbox3d[3]

                track_info_cur_frame[label_id]['lidar_box'] = lidar_synced_box                
                trajectory_info[label_id][frame_name] = lidar_synced_box
                track_info_cur_frame[label_id]['camera_box'] = None
            
                c = math.cos(lidar_synced_box['heading'])
                s = math.sin(lidar_synced_box['heading'])
                rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

                obj_pose_vehicle = np.eye(4)
                obj_pose_vehicle[:3, :3] = rotz_matrix
                obj_pose_vehicle[:3, 3] = np.array([lidar_synced_box['center_x'], lidar_synced_box['center_y'], lidar_synced_box['center_z']])
                
                camera_visible = []
                for camera_name in camera_names_dict.keys():
                    camera_calibration = calibration_map[camera_name]
                    vertices, valid = project_label_to_image(
                        dim=[lidar_synced_box['length'], lidar_synced_box['width'], lidar_synced_box['height']],
                        obj_pose=obj_pose_vehicle,
                        calibration=camera_calibration,
                    )
                
                    # if one corner of the 3D bounding box is on camera plane, we should consider it as visible
                    # partial visible for the case when not all corners can be observed
                    if valid.any():
                        camera_visible.append(camera_names_dict[camera_name])
                        track_camera_visible_cur_frame[camera_name].append(label_id)
                    if valid.all() and camera_name in ['center_camera_fov120', 'left_front_camera', 'right_front_camera']:
                        vertices = vertices.reshape(2, 2, 2, 2).astype(np.int32)
                        draw_3d_box_on_img(vertices, images[camera_name])
                    
                    
            track_info[frame_name] = track_info_cur_frame
            track_camera_visible[frame_name] = track_camera_visible_cur_frame
                
            track_vis_img = np.concatenate([
                images['left_front_camera'], 
                images['center_camera_fov120'], 
                images['right_front_camera']], axis=1)
            track_vis_imgs.append(track_vis_img)

        # reset information for trajectory 
        # poses, stationary, symmetric, deformable
        for label_id in trajectory_info.keys():
            new_trajectory = dict()
            
            trajectory = trajectory_info[label_id]
            trajectory = dict(sorted(trajectory.items(), key=lambda item: item[0]))
            
            dims = []
            frames = []
            timestamps = []
            poses_vehicle = []
            poses_world = []
            speeds = []
            
            for frame_name, bbox in trajectory.items():
                label = bbox['label']
                dims.append([bbox['height'], bbox['width'], bbox['length']])
                frames.append(frame_name)
                timestamps.append(bbox['timestamp'])
                speeds.append(bbox['speed'])
                pose_vehicle = np.eye(4)
                pose_vehicle[:3, :3] = np.array([
                    [math.cos(bbox['heading']), -math.sin(bbox['heading']), 0], 
                    [math.sin(bbox['heading']), math.cos(bbox['heading']), 0], 
                    [0, 0, 1]
                ])
                pose_vehicle[:3, 3] = np.array([bbox['center_x'], bbox['center_y'], bbox['center_z']])
                
                ego_pose = ego_vehicle_poses[frame_name]
                pose_world = np.matmul(ego_pose, pose_vehicle)
                
                poses_vehicle.append(pose_vehicle.astype(np.float32))
                poses_world.append(pose_world.astype(np.float32))
            
            # if label_id == '-ItvfksmEcYtVEcOjjRESg':
            #     __import__('ipdb').set_trace()
            
            dims = np.array(dims).astype(np.float32)
            dim = np.max(dims, axis=0)
            poses_vehicle = np.array(poses_vehicle).astype(np.float32)
            poses_world = np.array(poses_world).astype(np.float32)
            actor_world_postions = poses_world[:, :3, 3]
            
            # if label == 'sign':
            #     __import__('ipdb').set_trace()
            
            distance = np.linalg.norm(actor_world_postions[0] - actor_world_postions[-1])
            dynamic = np.any(np.std(actor_world_postions, axis=0) > 0.5) or distance > 2
            
            new_trajectory['label'] = label
            new_trajectory['height'], new_trajectory['width'], new_trajectory['length'] = dim[0], dim[1], dim[2]
            new_trajectory['poses_vehicle'] = poses_vehicle
            new_trajectory['timestamps'] = timestamps
            new_trajectory['frames'] = frames
            new_trajectory['speeds'] = speeds 
            new_trajectory['symmetric'] = (label != 'pedestrian')
            new_trajectory['deformable'] = (label == 'pedestrian')
            new_trajectory['stationary'] = not dynamic
            
            trajectory_info[label_id] = new_trajectory
            
            # print(new_trajectory['label'], new_trajectory['stationary'])


        # save visualization        
        imageio.mimwrite(os.path.join(track_dir, "track_vis.mp4"), track_vis_imgs, fps=24)
        
        # save track info
        with open(os.path.join(track_dir, "track_info.pkl"), 'wb') as f:
            pickle.dump(track_info, f)
            
        # save track camera visible
        with open(os.path.join(track_dir, "track_camera_visible.pkl"), 'wb') as f:
            pickle.dump(track_camera_visible, f)
            
        # save trajectory
        with open(os.path.join(track_dir, "trajectory.pkl"), 'wb') as f:
            pickle.dump(trajectory_info, f)
        
        with open(os.path.join(track_dir, "track_ids.json"), 'w') as f:
            json.dump(object_ids, f, indent=2)


        dynamic_mask_dir = os.path.join(seq_save_dir, 'dynamic_mask')
        if 'dynamic' not in process_list:
            print("Skipping dynamic mask processing...")
        elif os.path.exists(dynamic_mask_dir) and skip_existing:
            print('Dynamic mask already exists, skipping...')
        else:
            os.makedirs(dynamic_mask_dir, exist_ok=True)
            print("Processing dynamic mask...")
            track_info, track_camera_visible, trajectory = load_track(seq_save_dir)
            for frame, track_info_frame in track_info.items():
                track_camera_visible_cur_frame = track_camera_visible[frame]
                for camera_name in camera_names_dict:
                    calibration = calibration_map[camera_name]
                    dynamic_mask_name = frame + '_' + camera_names_dict[camera_name] + '.jpg'
                    dynamic_mask = np.zeros((calibration.height, calibration.width), dtype=np.uint8).astype(np.bool_)
                    deformable_mask_name = frame + '_' + camera_names_dict[camera_name] + '_deformable.jpg'
                    deformable_mask = np.zeros((calibration.height, calibration.width), dtype=np.uint8).astype(np.bool_)

                    track_ids = track_camera_visible_cur_frame[camera_name]

                    for track_id in track_ids:
                        object_tracklet = trajectory[track_id]
                        if object_tracklet['stationary']:
                            continue
                        pose_idx = trajectory[track_id]['frames'].index(frame)
                        pose_vehicle = trajectory[track_id]['poses_vehicle'][pose_idx]
                        height, width, length = trajectory[track_id]['height'], trajectory[track_id]['width'], trajectory[track_id]['length']
                        box_mask = project_label_to_mask(
                            dim=[length, width, height],
                            obj_pose=pose_vehicle,
                            calibration=calibration
                        )

                        dynamic_mask = np.logical_or(dynamic_mask, box_mask)
                        if trajectory[track_id]['deformable']:
                            deformable_mask = np.logical_or(deformable_mask, box_mask)

                    dynamic_mask_path = os.path.join(dynamic_mask_dir, dynamic_mask_name)
                    cv2.imwrite(dynamic_mask_path, dynamic_mask.astype(np.uint8) * 255)
        
                # deformable_mask_path = os.path.join(dynamic_mask_dir, deformable_mask_name)
                # cv2.imwrite(deformable_mask_path, deformable_mask.astype(np.uint8) * 255)

        print("Processing tracking data done...")

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_list', type=str, nargs='+', default=['pose', 'calib', 'image', 'track', 'track_old', 'lidar', 'dynamic'])
    # parser.add_argument('--process_list', type=str, nargs='+', default=['pose', 'calib', 'image', 'track'])
    parser.add_argument('--root_dir', type=str, default='/nas/home/yanyunzhi/waymo/training')
    parser.add_argument('--save_dir', type=str, default='./test_data/')
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()
    
    process_list = args.process_list
    root_dir = args.root_dir
    save_dir = args.save_dir
    
    parse_seq_rawdata(
        process_list=process_list,
        seq_path=root_dir,
        seq_save_dir=save_dir,
        skip_existing=args.skip_existing,
    )
        
if __name__ == '__main__':
    main()
    