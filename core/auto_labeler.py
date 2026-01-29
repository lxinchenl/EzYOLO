# -*- coding: utf-8 -*-
"""
自动打标签核心逻辑模块
负责使用YOLO模型自动生成标注
"""

import os
import time
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal, QObject

from core.model_manager import model_manager
from models.database import db


class AutoLabeler:
    """自动打标签器"""
    
    def __init__(self, model_path: str, model_manager):
        self.model_path = model_path
        self.model_manager = model_manager
        self.current_model = None
        self.model_info = {}
        self.class_mappings = {}
    
    def load_model(self, config: Dict) -> bool:
        """加载模型
        
        Args:
            config: 模型配置
            
        Returns:
            是否加载成功
        """
        try:
            model_version = config['model_version']
            model_size = config['model_size']
            model_source = config['model_source']
            custom_model_path = config.get('custom_model_path', '')
            
            # 加载模型
            if model_source == 'official':
                self.current_model = model_manager.load_model(
                    model_version, model_size, 'detect'
                )
            else:
                self.current_model = model_manager.load_custom_model(custom_model_path)
            
            if self.current_model:
                self.model_info = model_manager.get_model_info(self.current_model)
                self.class_mappings = config.get('class_mappings', {})
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def process_single_image(self, image_path: str, image_id: int, config: Dict) -> List[Dict]:
        """处理单张图像
        
        Args:
            image_path: 图像路径
            image_id: 图像ID
            config: 推理配置
            
        Returns:
            生成的标注列表
        """
        if not self.current_model:
            return []
        
        # 推理参数
        conf_threshold = config.get('conf_threshold', 0.5)
        iou_threshold = config.get('iou_threshold', 0.45)
        
        # 进行推理
        result = model_manager.infer(
            self.current_model, image_path, conf_threshold, iou_threshold
        )
        
        if not result:
            return []
        
        # 生成标注
        annotations = self._generate_annotations(result, image_id, config)
        return annotations
    
    def _generate_annotations(self, result: object, image_id: int, config: Dict) -> List[Dict]:
        """根据推理结果生成标注
        
        Args:
            result: 推理结果
            image_id: 图像ID
            config: 配置
            
        Returns:
            标注列表
        """
        annotations = []
        
        # 遍历检测结果
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # 计算宽度和高度
            width = x2 - x1
            height = y2 - y1
            
            # 应用类别映射
            mapped_class_id = self._map_class_id(class_id)
            
            # 创建标注
            annotation = {
                'image_id': image_id,
                'type': 'bbox',
                'class_id': mapped_class_id,
                'confidence': conf,
                'data': {
                    'x': x1,
                    'y': y1,
                    'width': width,
                    'height': height
                }
            }
            annotations.append(annotation)
        
        return annotations
    
    def _map_class_id(self, class_id: int) -> int:
        """映射类别ID
        
        Args:
            class_id: 模型输出的类别ID
            
        Returns:
            映射后的类别ID
        """
        # 检查是否有手动映射
        if class_id in self.class_mappings:
            return self.class_mappings[class_id]
        
        # 如果没有映射，直接返回原始ID
        # 这里可以添加逻辑，处理超出范围的情况
        return class_id
    
    def process_class_id(self, class_id: int, project_classes: list, enable_mapping: bool = False) -> int:
        """处理类别ID，根据映射设置和项目类别列表
        
        Args:
            class_id: 模型输出的类别ID
            project_classes: 项目类别列表
            enable_mapping: 是否启用类别映射
            
        Returns:
            处理后的类别ID
        """
        if enable_mapping:
            # 使用映射
            return self._map_class_id(class_id)
        else:
            # 不使用映射，直接对应或创建新类别
            # 检查类别ID是否在项目类别范围内
            if project_classes:
                max_class_id = max([cls['id'] for cls in project_classes])
                if class_id > max_class_id:
                    # 如果超出范围，返回原始ID（后续会创建新类别）
                    return class_id
            # 如果在范围内，直接返回
            return class_id
    
    def save_annotations(self, annotations: List[Dict], image_id: int, overwrite: bool = False):
        """保存标注
        
        Args:
            annotations: 标注列表
            image_id: 图像ID
            overwrite: 是否覆盖原标注
        """
        if overwrite:
            # 删除原标注
            db.delete_image_annotations(image_id)
        
        # 保存新标注
        for annotation in annotations:
            # 获取项目ID
            image_info = self.get_image_info(image_id)
            if not image_info:
                continue
            
            project_id = image_info.get('project_id', 0)
            
            # 获取类别名称
            class_id = annotation.get('class_id', 0)
            class_name = f"class_{class_id}"
            
            # 保存标注
            db.add_annotation(
                image_id,
                project_id,
                class_id,
                class_name,
                annotation.get('type', 'bbox'),
                annotation.get('data', {})
            )
    
    def get_image_info(self, image_id: int) -> Optional[Dict]:
        """获取图像信息
        
        Args:
            image_id: 图像ID
            
        Returns:
            图像信息
        """
        return db.get_image(image_id)
    
    def get_unlabeled_images(self, project_id: int) -> List[Dict]:
        """获取未标注的图像
        
        Args:
            project_id: 项目ID
            
        Returns:
            未标注图像列表
        """
        images = db.get_project_images(project_id)
        unlabeled = []
        
        for image in images:
            annotations = db.get_image_annotations(image['id'])
            if not annotations:
                unlabeled.append(image)
        
        return unlabeled
    
    def get_all_images(self, project_id: int) -> List[Dict]:
        """获取所有图像
        
        Args:
            project_id: 项目ID
            
        Returns:
            图像列表
        """
        return db.get_project_images(project_id)
    
    def process_image(self, image_path: str, conf_threshold: float, iou_threshold: float, class_mapping: dict) -> List[Dict]:
        """处理单张图像
        
        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            class_mapping: 类别映射
            
        Returns:
            生成的标注列表
        """
        try:
            # 加载模型
            if not self.current_model:
                # 根据model_path判断是官方模型还是自定义模型
                if os.path.exists(self.model_path):
                    # 自定义模型
                    self.current_model = self.model_manager.load_custom_model(self.model_path)
                else:
                    # 官方模型（模型名称）
                    # 尝试从pretrained目录加载
                    # 这里需要解析model_path来获取版本和大小
                    # 假设model_path是类似"yolov8n"这样的格式
                    match = re.match(r'(yolov)(\d+)([a-z])', self.model_path)
                    if match:
                        prefix = match.group(1)
                        version_num = match.group(2)
                        size = match.group(3)
                        # 转换版本格式，例如yolov8 -> YOLOv8
                        version = f"YOLOv{version_num}"
                        # 尝试通过model_manager加载
                        self.current_model = self.model_manager.load_model(version, size)
                    if not self.current_model:
                        # 如果model_manager加载失败，尝试直接加载但设置正确的下载路径
                        from ultralytics import YOLO
                        
                        # 设置环境变量，指定模型下载路径
                        pretrained_dir = os.path.join(os.path.dirname(__file__), '..', 'pretrained')
                        os.makedirs(pretrained_dir, exist_ok=True)
                        
                        original_hub_dir = os.environ.get('YOLO_HUB_DIR')
                        os.environ['YOLO_HUB_DIR'] = pretrained_dir
                        
                        try:
                            self.current_model = YOLO(self.model_path)
                        finally:
                            # 恢复原始环境变量
                            if original_hub_dir:
                                os.environ['YOLO_HUB_DIR'] = original_hub_dir
                            else:
                                del os.environ['YOLO_HUB_DIR']
            
            if not self.current_model:
                return []
            
            # 进行推理
            result = self.model_manager.infer(
                self.current_model, image_path, conf_threshold, iou_threshold
            )
            
            if not result:
                return []
            
            # 生成标注
            annotations = []
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # 计算宽度和高度
                width = x2 - x1
                height = y2 - y1
                
                # 应用类别映射
                mapped_class_id = class_mapping.get(class_id, class_id)
                
                # 创建标注
                annotation = {
                    'type': 'bbox',
                    'class_id': mapped_class_id,
                    'data': {
                        'x': x1,
                        'y': y1,
                        'width': width,
                        'height': height
                    }
                }
                annotations.append(annotation)
            
            return annotations
        except Exception as e:
            print(f"Error processing image: {e}")
            return []
    
    def unload_model(self):
        """卸载模型"""
        model_manager.unload_all_models()
        self.current_model = None
        self.model_info = {}


class BatchLabelingThread(QThread):
    """批量标注线程"""
    
    # 信号定义
    progress_updated = pyqtSignal(int, int, int)  # 已处理数量, 总数量, 已标注数量
    image_processed = pyqtSignal(str, int)  # 图像路径, 生成的标注数量
    batch_completed = pyqtSignal(bool, str)  # 是否成功, 消息
    
    def __init__(self, auto_labeler: AutoLabeler, images: List[Dict], config: Dict):
        super().__init__()
        self.auto_labeler = auto_labeler
        self.images = images
        self.config = config
        self._is_running = True
        self._is_paused = False
    
    def run(self):
        """运行批量标注"""
        total = len(self.images)
        processed = 0
        labeled = 0
        
        try:
            for i, image in enumerate(self.images):
                if not self._is_running:
                    break
                
                # 检查是否暂停
                while self._is_paused:
                    time.sleep(0.1)
                    if not self._is_running:
                        break
                
                if not self._is_running:
                    break
                
                # 处理图像
                image_path = image.get('storage_path', '')
                image_id = image.get('id', 0)
                
                if image_path and os.path.exists(image_path):
                    annotations = self.auto_labeler.process_single_image(
                        image_path, image_id, self.config
                    )
                    
                    if annotations:
                        # 保存标注
                        overwrite = self.config.get('overwrite_labels', False)
                        self.auto_labeler.save_annotations(
                            annotations, image_id, overwrite
                        )
                        labeled += len(annotations)
                        
                    # 发送信号
                    processed += 1
                    self.progress_updated.emit(processed, total, labeled)
                    self.image_processed.emit(image_path, len(annotations))
                
                # 避免CPU占用过高
                time.sleep(0.01)
            
            # 完成
            if self._is_running:
                self.batch_completed.emit(True, f"批量标注完成，处理了 {processed}/{total} 张图像，生成了 {labeled} 个标注")
            else:
                self.batch_completed.emit(False, f"批量标注被停止，已处理 {processed}/{total} 张图像")
        except Exception as e:
            self.batch_completed.emit(False, f"批量标注出错: {str(e)}")
    
    def pause(self):
        """暂停"""
        self._is_paused = True
    
    def resume(self):
        """恢复"""
        self._is_paused = False
    
    def stop(self):
        """停止"""
        self._is_running = False
        self._is_paused = False


class BatchLabelingManager(QObject):
    """批量标注管理器"""
    
    # 信号定义
    progress_updated = pyqtSignal(int, int, int, str)  # 进度, 当前, 总数, 图像名称
    batch_completed = pyqtSignal(bool, str, int)  # 成功, 消息, 处理数量
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_thread = None
        self.auto_labeler = None
        self.model_manager = None
        self.images = []
    
    def start_batch_processing(self, model_path: str, images: list, conf_threshold: float, iou_threshold: float, class_mapping: dict, model_manager):
        """开始批量处理
        
        Args:
            model_path: 模型路径
            images: 图像列表
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            class_mapping: 类别映射
            model_manager: 模型管理器实例
        """
        try:
            self.model_manager = model_manager
            self.images = images  # 保存图像列表
            
            # 创建自动标注器
            self.auto_labeler = AutoLabeler(model_path, model_manager)
            
            # 配置
            config = {
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'class_mappings': class_mapping,
                'overwrite_labels': True
            }
            
            # 加载模型
            import re
            match = re.match(r'(yolov)(\d+)([a-z])', model_path)
            if match:
                version_num = match.group(2)
                size = match.group(3)
                version = f"YOLOv{version_num}"
                load_config = {
                    'model_version': version,
                    'model_size': size,
                    'model_source': 'official',
                    'class_mappings': class_mapping
                }
                self.auto_labeler.load_model(load_config)
            else:
                # 自定义模型
                if os.path.exists(model_path):
                    # 直接加载模型
                    if not self.auto_labeler.current_model:
                        self.auto_labeler.current_model = model_manager.load_custom_model(model_path)
            
            # 创建并启动线程
            self.current_thread = BatchLabelingThread(
                self.auto_labeler, images, config
            )
            
            # 连接信号
            self.current_thread.progress_updated.connect(self.on_progress_updated)
            self.current_thread.batch_completed.connect(self.on_batch_completed)
            
            # 启动线程
            self.current_thread.start()
        except Exception as e:
            print(f"Error starting batch processing: {e}")
    
    def on_progress_updated(self, processed: int, total: int, labeled: int):
        """进度更新回调"""
        # 发送进度信号
        if hasattr(self, 'progress_updated'):
            current_image = self.images[processed-1] if processed <= len(self.images) else {}
            image_name = os.path.basename(current_image.get('storage_path', ''))
            progress = int((processed / total) * 100)
            self.progress_updated.emit(progress, processed, total, image_name)
    
    def on_batch_completed(self, success: bool, message: str):
        """批量完成回调"""
        # 发送完成信号
        if hasattr(self, 'batch_completed'):
            processed_count = len(self.images)
            self.batch_completed.emit(success, message, processed_count)
    
    def pause(self):
        """暂停"""
        if self.current_thread:
            self.current_thread.pause()
    
    def resume(self):
        """恢复"""
        if self.current_thread:
            self.current_thread.resume()
    
    def stop(self):
        """停止"""
        if self.current_thread:
            self.current_thread.stop()
            self.current_thread.wait()
    
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self.current_thread and self.current_thread.isRunning()
    
    def cleanup(self):
        """清理资源"""
        self.stop()
        self.auto_labeler.unload_model()
