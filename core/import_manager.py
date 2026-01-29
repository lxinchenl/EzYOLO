# -*- coding: utf-8 -*-
"""
导入管理器
处理图像导入、视频抽帧、标注导入等功能
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from datetime import datetime
import cv2
from PIL import Image
import numpy as np

from models.database import db


class ImportManager:
    """导入管理器类"""
    
    # 支持的图像格式
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    
    # 支持的视频格式
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    
    def __init__(self, project_id: int):
        """
        初始化导入管理器
        
        Args:
            project_id: 项目ID
        """
        self.project_id = project_id
        self.project = db.get_project(project_id)
        if not self.project:
            raise ValueError(f"项目 {project_id} 不存在")
        
        # 确保项目存储目录存在
        self.storage_path = Path(self.project['storage_path'])
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 创建images子目录
        self.images_path = self.storage_path / "images"
        self.images_path.mkdir(exist_ok=True)
    
    def import_folder(self, folder_path: str, 
                     progress_callback: Callable[[int, str], None] = None) -> Tuple[int, int]:
        """
        导入文件夹中的所有图像
        
        Args:
            folder_path: 文件夹路径
            progress_callback: 进度回调函数，参数为(进度百分比, 状态信息)
            
        Returns:
            (成功导入数量, 跳过数量)
        """
        folder_path = Path(folder_path)
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"无效的文件夹路径: {folder_path}")
        
        # 获取所有图像文件
        image_files = []
        for ext in self.SUPPORTED_IMAGE_FORMATS:
            image_files.extend(folder_path.rglob(f"*{ext}"))
            image_files.extend(folder_path.rglob(f"*{ext.upper()}"))
        
        # 去重并排序
        image_files = sorted(set(image_files))
        
        if not image_files:
            return 0, 0
        
        total = len(image_files)
        imported = 0
        skipped = 0
        
        for i, image_file in enumerate(image_files):
            try:
                # 更新进度
                progress = int((i / total) * 100)
                if progress_callback:
                    progress_callback(progress, f"正在导入: {image_file.name}")
                
                # 导入单张图像
                result = self.import_single_image(str(image_file))
                if result:
                    imported += 1
                else:
                    skipped += 1
                    
            except Exception as e:
                print(f"导入图像失败 {image_file}: {e}")
                skipped += 1
        
        # 完成进度
        if progress_callback:
            progress_callback(100, f"导入完成: 成功 {imported}, 跳过 {skipped}")
        
        return imported, skipped
    
    def import_images(self, file_paths: List[str],
                     progress_callback: Callable[[int, str], None] = None) -> Tuple[int, int]:
        """
        导入多张图像
        
        Args:
            file_paths: 图像文件路径列表
            progress_callback: 进度回调函数
            
        Returns:
            (成功导入数量, 跳过数量)
        """
        total = len(file_paths)
        imported = 0
        skipped = 0
        
        for i, file_path in enumerate(file_paths):
            try:
                # 更新进度
                progress = int((i / total) * 100)
                if progress_callback:
                    progress_callback(progress, f"正在导入: {Path(file_path).name}")
                
                # 导入单张图像
                result = self.import_single_image(file_path)
                if result:
                    imported += 1
                else:
                    skipped += 1
                    
            except Exception as e:
                print(f"导入图像失败 {file_path}: {e}")
                skipped += 1
        
        # 完成进度
        if progress_callback:
            progress_callback(100, f"导入完成: 成功 {imported}, 跳过 {skipped}")
        
        return imported, skipped
    
    def import_single_image(self, file_path: str) -> bool:
        """
        导入单张图像
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            是否成功导入
        """
        file_path = Path(file_path)
        
        # 检查文件是否存在
        if not file_path.exists():
            return False
        
        # 检查文件格式
        if file_path.suffix.lower() not in self.SUPPORTED_IMAGE_FORMATS:
            return False
        
        # 检查是否已存在（通过文件哈希）
        file_hash = self._calculate_file_hash(str(file_path))
        if self._check_duplicate(file_hash):
            print(f"图像已存在，跳过: {file_path.name}")
            return False
        
        try:
            # 读取图像信息
            image_info = self._get_image_info(str(file_path))
            if not image_info:
                return False
            
            # 生成目标文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            target_filename = f"{timestamp}_{file_path.name}"
            target_path = self.images_path / target_filename
            
            # 复制文件到项目目录
            shutil.copy2(str(file_path), str(target_path))
            
            # 添加到数据库
            db.add_image(
                project_id=self.project_id,
                filename=file_path.name,
                storage_path=str(target_path),
                width=image_info['width'],
                height=image_info['height'],
                size=image_info['size'],
                image_format=image_info['format'],
                original_path=str(file_path)
            )
            
            return True
            
        except Exception as e:
            print(f"导入图像失败 {file_path}: {e}")
            return False
    
    def import_video(self, video_path: str, frame_interval: int = 1,
                    progress_callback: Callable[[int, str], None] = None) -> Tuple[int, int]:
        """
        从视频中抽取帧导入
        
        Args:
            video_path: 视频文件路径
            frame_interval: 抽帧间隔（每隔多少帧抽取一帧）
            progress_callback: 进度回调函数
            
        Returns:
            (成功导入数量, 跳过数量)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise ValueError(f"视频文件不存在: {video_path}")
        
        if video_path.suffix.lower() not in self.SUPPORTED_VIDEO_FORMATS:
            raise ValueError(f"不支持的视频格式: {video_path.suffix}")
        
        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        imported = 0
        skipped = 0
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 按间隔抽帧
                if frame_count % frame_interval == 0:
                    try:
                        # 更新进度
                        progress = int((frame_count / total_frames) * 100)
                        if progress_callback:
                            progress_callback(progress, f"正在抽取帧 {frame_count}/{total_frames}")
                        
                        # 保存帧为图像
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        target_filename = f"{timestamp}_frame_{frame_count:06d}.jpg"
                        target_path = self.images_path / target_filename
                        
                        # 转换颜色空间并保存
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(str(target_path), frame)
                        
                        # 获取图像信息
                        height, width = frame.shape[:2]
                        size = target_path.stat().st_size
                        
                        # 添加到数据库
                        db.add_image(
                            project_id=self.project_id,
                            filename=target_filename,
                            storage_path=str(target_path),
                            width=width,
                            height=height,
                            size=size,
                            image_format='jpg',
                            original_path=str(video_path)
                        )
                        
                        imported += 1
                        
                    except Exception as e:
                        print(f"保存帧失败 {frame_count}: {e}")
                        skipped += 1
                
                frame_count += 1
                
        finally:
            cap.release()
        
        # 完成进度
        if progress_callback:
            progress_callback(100, f"视频导入完成: 成功 {imported}, 跳过 {skipped}")
        
        return imported, skipped
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        计算文件哈希值（用于去重）
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件哈希值
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _check_duplicate(self, file_hash: str) -> bool:
        """
        检查文件是否已存在
        
        Args:
            file_hash: 文件哈希值
            
        Returns:
            是否已存在
        """
        # TODO: 实现基于哈希的去重检查
        # 目前简单检查文件名是否已存在
        return False
    
    def _get_image_info(self, file_path: str) -> Optional[Dict]:
        """
        获取图像信息
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            图像信息字典，失败返回None
        """
        try:
            # 使用PIL获取图像信息
            with Image.open(file_path) as img:
                width, height = img.size
                image_format = img.format.lower() if img.format else 'unknown'
            
            # 获取文件大小
            size = Path(file_path).stat().st_size
            
            return {
                'width': width,
                'height': height,
                'size': size,
                'format': image_format
            }
            
        except Exception as e:
            print(f"获取图像信息失败 {file_path}: {e}")
            return None
    
    def get_project_images(self) -> List[Dict]:
        """
        获取项目中的所有图像
        
        Returns:
            图像列表
        """
        return db.get_project_images(self.project_id)
    
    def delete_image(self, image_id: int) -> bool:
        """
        删除图像
        
        Args:
            image_id: 图像ID
            
        Returns:
            是否成功删除
        """
        try:
            # 获取图像信息
            images = db.get_project_images(self.project_id)
            image_info = None
            for img in images:
                if img['id'] == image_id:
                    image_info = img
                    break
            
            if not image_info:
                return False
            
            # 删除文件
            storage_path = image_info.get('storage_path')
            if storage_path and Path(storage_path).exists():
                Path(storage_path).unlink()
            
            # TODO: 从数据库删除记录
            # 需要在database.py中添加delete_image方法
            
            return True
            
        except Exception as e:
            print(f"删除图像失败 {image_id}: {e}")
            return False
