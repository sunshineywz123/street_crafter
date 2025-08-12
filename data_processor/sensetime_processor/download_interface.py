import concurrent.futures
import os
import sys

from loguru import logger

def download_file(remote_path, save_path, backend, strict=True, retry=5):
    """ download file """
    # if not backend.exists(remote_path):
    #     if strict == True:
    #         logger.error(f"{remote_path} not exists")
    #         sys.exit(-1)
    #     else:
    #         logger.warning(f"{remote_path} not exists")
    #         return
    
    # 创建保存路径的目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 初始化下载成功标志
    success_flag = False
    # 尝试下载指定次数
    for i in range(0, retry):
        # 以二进制写模式打开文件
        with open(save_path, "wb") as fp:
            # 从后端获取数据并写入文件
            fp.write(backend.get(remote_path))
        # 检查文件是否存在且大小大于0
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            # 设置下载成功标志
            success_flag = True
            break
        # 记录下载失败警告日志
        logger.warning(f"download failed: {remote_path} to {save_path}, retry times: {i}")
    
    # 如果下载失败
    if not success_flag:
        # 如果是严格模式
        if strict == True:
            # 记录错误日志并退出
            logger.error(f"download failed: {remote_path} to {save_path}")
            sys.exit(-1)
        else:
            # 记录警告日志
            logger.warning(f"download failed: {remote_path} to {save_path}")
            # 如果文件存在但大小为0
            if os.path.exists(save_path) and os.path.getsize(save_path) == 0:
                # 记录警告并删除空文件
                logger.warning(f"size is 0, remove {save_path}")
                os.remove(save_path)
    else:
        # 记录下载成功日志
        logger.info(f"download {remote_path} to {save_path}")

def upload_file(local_path, remote_path, backend, strict=True, retry=5):
    """ upload file """
    if not os.path.exists(local_path):
        if strict == True:
            logger.error(f"{local_path} not exists")
            sys.exit(-1)
        else:
            logger.warning(f"{local_path} not exists")
            return
    
    success_flag = False
    for i in range(0, retry):
        backend.put(remote_path, open(local_path, "rb"))
        if backend.exists(remote_path):
            success_flag = True
            break
        logger.warning(f"upload failed: {local_path} to {remote_path}, retry times: {i}")
    
    if not success_flag:
        if strict == True:
            logger.error(f"upload failed: {local_path} to {remote_path}")
            sys.exit(-1)
        else:
            logger.warning(f"upload failed: {local_path} to {remote_path}")
    else:
        logger.info(f"upload {local_path} to {remote_path}")


