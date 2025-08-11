import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from datetime import datetime, timedelta, timezone

def parse_custom_timestamp(ts_str):
    # 补全毫秒为微秒
    if len(ts_str.split("-")) == 7:
        ts_str = ts_str + "000"
    dt_local = datetime.strptime(ts_str, "%Y-%m-%d-%H-%M-%S-%f")

    # 设置为东八区时区
    tz_east8 = timezone(timedelta(hours=8))
    dt_east8 = dt_local.replace(tzinfo=tz_east8)

    # 转为 UTC 时间戳（单位：秒）
    timestamp_utc = float(dt_east8.timestamp()) * 1e3
    return timestamp_utc


def parse_txt_to_poses(file_path):
    results = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 13:
                continue  # 跳过非法行

            timestamp = float(parse_custom_timestamp(parts[0]))
            numbers = list(map(float, parts[1:]))

            # 构造 3x4 矩阵
            matrix_3x4 = np.array(numbers).reshape(3, 4)

            # 扩展为 4x4 齐次矩阵
            transform = np.eye(4)
            transform[:3, :4] = matrix_3x4

            results.append((timestamp, transform))

    return results


def interpolate_pose(t1, pose1, t2, pose2, t_target):
    alpha = (t_target - t1) / (t2 - t1)
    trans1 = pose1[:3, 3]
    trans2 = pose2[:3, 3]
    interp_trans = (1 - alpha) * trans1 + alpha * trans2

    # 旋转插值使用 Slerp
    rot1 = Rotation.from_matrix(pose1[:3, :3])
    rot2 = Rotation.from_matrix(pose2[:3, :3])
    slerp = Slerp([t1, t2], Rotation.concatenate([rot1, rot2]))
    interp_rot = slerp([t_target]).as_matrix()[0]

    result = np.eye(4)
    result[:3, :3] = interp_rot
    result[:3, 3] = interp_trans
    return result


def extrapolate_rotation(rot1, rot2, t1, t2, t_target):
    delta_t = t2 - t1
    if delta_t == 0:
        return rot1.as_matrix()

    # 计算相对旋转
    R_rel = rot1.inv() * rot2
    # 旋转向量（axis-angle）
    rvec = R_rel.as_rotvec()

    # 计算时间比例
    alpha = (t_target - t1) / delta_t

    # 外推旋转向量
    extrap_rvec = rvec * alpha

    # 计算外推旋转
    extrap_rot = rot1 * Rotation.from_rotvec(extrap_rvec)
    return extrap_rot.as_matrix()


def extrapolate_pose(t1, pose1, t2, pose2, t_target):
    trans1 = pose1[:3, 3]
    trans2 = pose2[:3, 3]
    delta_t = t2 - t1
    velocity = (trans2 - trans1) / delta_t
    extrap_trans = trans1 + velocity * (t_target - t1)

    rot1 = Rotation.from_matrix(pose1[:3, :3])
    rot2 = Rotation.from_matrix(pose2[:3, :3])

    extrap_rot = extrapolate_rotation(rot1, rot2, t1, t2, t_target)

    result = np.eye(4)
    result[:3, :3] = extrap_rot
    result[:3, 3] = extrap_trans
    return result

def get_pose_at_time(poses, t_target):
    """
    输入：
        poses: List[Tuple[float, 4x4 np.array]]，时间戳升序排列
        t_target: float，要查询的时间戳
    输出：
        t_target 时刻的 4x4 位姿矩阵（插值或外推）
    """

    if not poses or len(poses) < 2:
        raise ValueError("需要至少两帧数据")

    # 目标时间在轨迹之前 → 前向外推
    if t_target <= poses[0][0]:
        return extrapolate_pose(poses[0][0], poses[0][1], poses[1][0], poses[1][1], t_target)

    # 目标时间在轨迹之后 → 后向外推
    if t_target >= poses[-1][0]:
        return extrapolate_pose(poses[-2][0], poses[-2][1], poses[-1][0], poses[-1][1], t_target)

    # 中间 → 找到相邻两帧进行插值
    for i in range(len(poses) - 1):
        t1, T1 = poses[i]
        t2, T2 = poses[i + 1]
        if t1 <= t_target <= t2:
            return interpolate_pose(t1, T1, t2, T2, t_target)

    raise RuntimeError("无法插值或外推该时间戳")
