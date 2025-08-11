import argparse
from email.mime import image
import os
import shutil
import json
from turtle import left

import numpy as np
from pyparsing import annotations
from scipy.spatial.transform import Rotation
from sympy import root
from tqdm import tqdm


#一共有那么多个摄像头数据，不知道要哪五个
camera_names = ['center_camera_fov30',
                'center_camera_fov30',
                'center_camera_fov30', 
                'center_camera_fov30',
                'center_camera_fov30']

#从示例的waymo集里timestamp翻出来的五个名字，但是交集是空集
# camera_names = ['center_camera_fov30',
#             'left_camera_fov195',
#             'left_front_camera', 
#             'right_camera_fov195',
#             'right_front_camera']


#结果的位数待处理
def gen_timestamp(root_dir,save_dir):
    """
    从根目录下的sensor_temporal_alignment.json提取时间戳
    存放至timestamp.json
    """
    timestamp_save_path=os.path.join(save_dir,'timestamp.json')
    timestamp = {
    'FRAME': {},
    'FRONT_LEFT': {},
    'FRONT_RIGHT': {},
    'SIDE_LEFT': {},
    'SIDE_RIGHT': {},
    'FRONT': {}
    }

    try:
        with open(os.path.join(root_dir,'sensor_temporal_alignment.json'),'r',encoding='utf-8') as f:
            data=json.load(f)
            titles=[]
            frame_id_short=0

            for item in data:
                for frame_id in item.keys():
                    titles.append(frame_id)
                    timestamp['FRAME'][str(frame_id_short).zfill(6)]=int(frame_id) /1e9
                    frame_id_short+=1

            frame_id_short=0
            for frame in data:
                # timestamp['FRAME'][str(frame_id).zfill(6)]=frame.get('frame_id','')

                left_front_value=frame.get(titles[frame_id_short],{}).get('left_front_camera','')
                timestamp['FRONT_LEFT'][str(frame_id_short).zfill(6)]=int(os.path.basename(left_front_value) .split('.')[0] ) / 1e9
                right_front_value=frame.get(titles[frame_id_short],{}).get('right_front_camera','')
                timestamp['FRONT_RIGHT'][str(frame_id_short).zfill(6)]=int(os.path.basename(right_front_value).split('.')[0] ) / 1e9
                left_value=frame.get(titles[frame_id_short],{}).get('left_camera_fov195','')
                timestamp['SIDE_LEFT'][str(frame_id_short).zfill(6)]=int(os.path.basename(left_value).split('.')[0] ) / 1e9
                right_value=frame.get(titles[frame_id_short],{}).get('right_camera_fov195','')
                timestamp['SIDE_RIGHT'][str(frame_id_short).zfill(6)]=int(os.path.basename(right_value).split('.')[0] ) / 1e9
                front_value=frame.get(titles[frame_id_short],{}).get('center_camera_fov30','')
                timestamp['FRONT'][str(frame_id_short).zfill(6)]=int(os.path.basename(front_value).split('.')[0]) / 1e9

                frame_id_short += 1

        with open(timestamp_save_path, 'w', encoding='utf-8') as f:
            json.dump(timestamp, f, indent=1)


    except FileNotFoundError:
        print("sensor_temporal_alignment.json not found")
        return


def process_ego_pose(root_dir,save_dir,json_files):
    """
    获取相机的姿态矩阵
    """
    ego_save_dir=os.path.join(save_dir,'ego_pose')
    if not os.path.exists(ego_save_dir):
        os.makedirs(ego_save_dir, exist_ok=True)
    else:
        os.makedirs(ego_save_dir, exist_ok=True)

    frame_id=0

    for json_file in json_files:
        try:
            with open(os.path.join(root_dir, 
                                'format_output',
                                'annotations', 
                                'NV_lane3d', 
                                json_file),
                                'r'
                                ,encoding='utf-8') as file:
                data=json.load(file)

            ego_translation=data.get("ego2global_translation",[])
            ego_rotation_quaternion=data.get("ego2global_rotation",[])
            ego_velocity=data.get("ego_velocity",[])

            rotation_matrix = Rotation.from_quat(ego_rotation_quaternion).as_matrix()
            pose_matrix=np.eye(4)
            pose_matrix[:3, :3] = rotation_matrix
            pose_matrix[:3, 3] = ego_translation

        except FileNotFoundError:
            print(f"{json_file} not found")

        ego_file_save_path=os.path.join(ego_save_dir,f"{frame_id:06d}.txt")

        with open(ego_file_save_path,'w',encoding='utf-8') as f:
            for row in pose_matrix:
                f.write(" ".join(map(str, row)) + "\n")   #待改为以科学计数法保存
        frame_id += 1


def process_camera_calibration(root_dir, save_dir):
    """
    提取相机内参与外参
    Args:
        root_dir (str): pilotGTParser数据集的根目录
        save_dir (str): 处理后数据保存的目录
    """
    json_files=sorted([x for x in os.listdir(os.path.join(root_dir, 'format_output', 'annotations', 'NV_lane3d')) if x.endswith('.json')])

    extrinsics_save_dir=os.path.join(save_dir,'extrinsics')
    intrinsics_save_dir=os.path.join(save_dir,'intrinsics')

    intrinsics=list()
    extrinscis=list()

    try:
        with open(os.path.join(root_dir, 
                            'format_output',
                            'annotations', 
                            'NV_lane3d', 
                            json_files[0]),
                            'r'
                            ,encoding='utf-8') as file:
            data=json.load(file)

            for camera_name in camera_names:
                cam_data=data.get("cams",{}).get(camera_name,{})

                if cam_data:
                    intrinsics.append(cam_data.get("cam_intrinsic",[]))
                    extrinscis.append(cam_data.get("extrinsic",[]))
        
        print(intrinsics[0][0][1])
        print(extrinscis[1][1][2])

    except FileNotFoundError:
        print(f"{json_files[0]}not found")
    except json.JSONDecodeError:
        print("json decode ereror")

    for i in range(5):
        intrinsics_file_save_dir = os.path.join(intrinsics_save_dir, f"{i}.txt")
        extrinscis_file_save_dir = os.path.join(extrinsics_save_dir, f"{i}.txt")

        if os.path.exists(intrinsics_file_save_dir) and os.path.exists(extrinscis_file_save_dir):
            continue
        else:
            os.makedirs(intrinsics_save_dir, exist_ok=True)
            os.makedirs(extrinsics_save_dir, exist_ok=True)

        try:
            with open(intrinsics_file_save_dir, 'w') as f:
                for row in intrinsics[i]:
                    f.write("\n".join(map(str, row)))           #顺序看起来很奇怪

            with open(extrinscis_file_save_dir, 'w') as f:
                for row in extrinscis[i]:
                    f.write(" ".join(map(str, row)) + "\n")  

        except FileNotFoundError:
            print(f"File {intrinsics_file_save_dir} not found")

    process_ego_pose(root_dir,save_dir,json_files)



def process_image(root_dir, save_dir,skip_existing):
    """
    到camera目录底下找五个摄像头的图像,重命名为waymo的格式并复制到保存目录下
    Args:
        root_dir (str): pilotGTParser数据集的根目录
        save_dir (str): 处理后数据保存的目录
    Returns:
        temporary none
    """
    image_save_dir = os.path.join(save_dir, 'images')

    if os.path.exists(image_save_dir) and skip_existing:
        print('Images already exist, skipping...')
    else:
        os.makedirs(image_save_dir, exist_ok=True)      
        print("Processing image data...")

    print(f"Processing images from {root_dir} and saving to {save_dir}")

    # image_dir=os.path.join(root_dir,'camera')

    # camera0_images=set([x for x in os.listdir(os.path.join(image_dir, camera_names[0])) if x.endswith('.jpg')])
    # camera1_images=set([x for x in os.listdir(os.path.join(image_dir, camera_names[1])) if x.endswith('.jpg')])
    # camera2_images=set([x for x in os.listdir(os.path.join(image_dir, camera_names[2])) if x.endswith('.jpg')])
    # camera3_images=set([x for x in os.listdir(os.path.join(image_dir, camera_names[3])) if x.endswith('.jpg')])
    # camera4_images=set([x for x in os.listdir(os.path.join(image_dir, camera_names[4])) if x.endswith('.jpg')])
    
    # all_image_names =sorted(camera0_images & camera1_images & camera2_images & camera3_images & camera4_images)

    #不同帧相机都是一样的，取第一帧读取内外参
    # process_camera_calibration(root_dir, save_dir)
    
    image_id=000000

    sensor_sync_file_path = os.path.join(root_dir, 'sensor_temporal_alignment.json')
    with open(sensor_sync_file_path, 'r') as sensor_sync_file:
        sensor_sync = json.load(sensor_sync_file)
        for frame_id, sensor_frame in tqdm(enumerate(sensor_sync)):
            for key in sensor_frame:
                if "top_center_lidar" in sensor_frame[key]:
                    image_path = os.path.join(root_dir,sensor_frame[key]['center_camera_fov120'])
                    new_image_name = f"{image_id:06d}_{0}.jpg"

                    if not os.path.exists(image_path):
                        print(f"Source image {image_path} not found")
                        continue
                    dst_path = os.path.join(image_save_dir, new_image_name)
                    
                    shutil.copy(image_path, dst_path)
                image_id += 1
    print("image copy done")


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/iag_ad_01/ad/yuanweizhong/datasets/senseauto/2024_09_08_07_53_23_pathway_pilotGtParser')
    parser.add_argument('--save_dir', type=str, default='/iag_ad_01/ad/yuanweizhong/huzeyu/street_crafter/data')
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()

    root_dir = args.root_dir
    save_dir = args.save_dir
    
    process_image(root_dir, save_dir, 0) #skip_existing 功能待写
    # gen_timestamp(root_dir, save_dir)


if __name__ == "__main__":
    main()
