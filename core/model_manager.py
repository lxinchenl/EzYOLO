# -*- coding: utf-8 -*-
"""
模型管理模块
负责YOLO模型的加载、管理和推理
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Ultralytics YOLO 导入
try:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# 真实的Ultralytics模型配置（从train_page.py获取）
ULTRALYTICS_MODELS = {
    "YOLOv3": {
        "sizes": ["n", "u"],
        "tasks": ["detect"],
        "prefix": "yolov3",
    },
    "YOLOv5": {
        "sizes": ["nu", "su", "mu", "lu", "xu"],
        "tasks": ["detect"],
        "prefix": "yolov5",
    },
    "YOLOv8": {
        "sizes": ["n", "s", "m", "l", "x"],
        "tasks": ["detect", "classify", "obb", "pose", "segment", "world"],
        "prefix": "yolov8",
    },
    "YOLOv9": {
        "sizes": ["t", "s", "m", "c", "e"],
        "tasks": ["detect"],
        "prefix": "yolov9",
    },
    "YOLOv10": {
        "sizes": ["n", "s", "m", "b", "l", "x"],
        "tasks": ["detect"],
        "prefix": "yolov10",
    },
    "YOLOv11": {
        "sizes": ["n", "s", "m", "l", "x"],
        "tasks": ["detect", "classify", "obb", "pose", "segment"],
        "prefix": "yolo11",
    },
    "YOLOv12": {
        "sizes": ["n", "s", "m", "l", "x"],
        "tasks": ["detect"],
        "prefix": "yolo12",
    },
    "YOLOv26": {
        "sizes": ["n", "s", "m", "l", "x"],
        "tasks": ["detect", "classify", "obb", "pose", "segment"],
        "prefix": "yolo26",
    },
}


class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        self.models = {}
        # 使用基于应用根目录的相对路径
        app_root = Path(__file__).parent.parent  # 向上两级到sfyolo根目录
        self.pretrained_dir = app_root / "pretrained"
        self.pretrained_dir.mkdir(exist_ok=True)
    
    def get_model_path(self, version: str, size: str, task: str = "detect") -> Path:
        """获取模型路径
        
        Args:
            version: 模型版本，如 "YOLOv8"
            size: 模型大小，如 "n"
            task: 任务类型，如 "detect"
            
        Returns:
            模型文件路径
        """
        if version not in ULTRALYTICS_MODELS:
            raise ValueError(f"Unknown model version: {version}")
        
        model_info = ULTRALYTICS_MODELS[version]
        prefix = model_info["prefix"]
        
        # 构建模型文件名
        if task == "detect":
            model_name = f"{prefix}{size}.pt"
        else:
            model_name = f"{prefix}{size}-{task}.pt"
        
        model_path = self.pretrained_dir / model_name
        return model_path
    
    def load_model(self, version: str, size: str, task: str = "detect") -> Optional[YOLO]:
        """加载官方预训练模型
        
        Args:
            version: 模型版本
            size: 模型大小
            task: 任务类型
            
        Returns:
            加载的YOLO模型实例，或None如果加载失败
        """
        if not YOLO_AVAILABLE:
            return None
        
        model_path = self.get_model_path(version, size, task)
        model_key = f"{version}_{size}_{task}"
        
        # 检查模型是否已加载
        if model_key in self.models:
            return self.models[model_key]
        
        try:
            # 尝试从本地加载
            if model_path.exists():
                model = YOLO(str(model_path))
            else:
                # 本地不存在，尝试在线下载
                # 构建模型名称
                if task == "detect":
                    model_name = f"{version.lower()}{size}"
                else:
                    model_name = f"{version.lower()}{size}-{task}"
                
                # 设置环境变量，指定模型下载路径
                import os
                original_hub_dir = os.environ.get('YOLO_HUB_DIR')
                os.environ['YOLO_HUB_DIR'] = str(self.pretrained_dir)
                
                try:
                    model = YOLO(model_name)
                finally:
                    # 恢复原始环境变量
                    if original_hub_dir:
                        os.environ['YOLO_HUB_DIR'] = original_hub_dir
                    else:
                        del os.environ['YOLO_HUB_DIR']
                
                # 确保模型文件存在
                if not model_path.exists():
                    # 如果模型文件仍然不存在，尝试复制
                    import shutil
                    for root, dirs, files in os.walk(self.pretrained_dir):
                        for file in files:
                            if file == model_path.name:
                                src_path = os.path.join(root, file)
                                shutil.copy2(src_path, model_path)
                                break
            
            self.models[model_key] = model
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def load_custom_model(self, model_path: str) -> Optional[YOLO]:
        """加载自定义模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            加载的YOLO模型实例，或None如果加载失败
        """
        if not YOLO_AVAILABLE:
            return None
        
        model_key = f"custom_{model_path}"
        
        # 检查模型是否已加载
        if model_key in self.models:
            return self.models[model_key]
        
        try:
            model = YOLO(model_path)
            self.models[model_key] = model
            return model
        except Exception as e:
            print(f"Error loading custom model: {e}")
            return None
    
    def get_model_info(self, model: YOLO) -> Dict:
        """获取模型信息
        
        Args:
            model: YOLO模型实例
            
        Returns:
            包含模型信息的字典
        """
        info = {
            "model_type": "YOLO",
            "task": model.task,
            "names": model.names if hasattr(model, 'names') else {},
            "nc": len(model.names) if hasattr(model, 'names') else 0,
        }
        return info
    
    def infer(self, model: YOLO, image_path: str, conf: float = 0.5, iou: float = 0.45) -> Optional[Results]:
        """使用模型进行推理
        
        Args:
            model: YOLO模型实例
            image_path: 图像路径
            conf: 置信度阈值
            iou: IOU阈值
            
        Returns:
            推理结果，或None如果推理失败
        """
        try:
            results = model.predict(
                source=image_path,
                conf=conf,
                iou=iou,
                verbose=False
            )
            return results[0] if results else None
        except Exception as e:
            print(f"Error during inference: {e}")
            return None
    
    def unload_model(self, model_key: str):
        """卸载模型
        
        Args:
            model_key: 模型键
        """
        if model_key in self.models:
            del self.models[model_key]
    
    def unload_all_models(self):
        """卸载所有模型"""
        self.models.clear()
    
    def get_available_models(self) -> List[Dict]:
        """获取可用的模型列表
        
        Returns:
            可用模型列表
        """
        available_models = []
        
        for version, info in ULTRALYTICS_MODELS.items():
            for size in info["sizes"]:
                for task in info["tasks"]:
                    model_path = self.get_model_path(version, size, task)
                    available = model_path.exists()
                    available_models.append({
                        "version": version,
                        "size": size,
                        "task": task,
                        "path": str(model_path),
                        "available": available
                    })
        
        return available_models


# 全局模型管理器实例
model_manager = ModelManager()


if __name__ == "__main__":
    # 测试模型加载
    manager = ModelManager()
    
    # 测试加载YOLOv8n模型
    model = manager.load_model("YOLOv8", "n")
    if model:
        print("Successfully loaded YOLOv8n model")
        info = manager.get_model_info(model)
        print(f"Model info: {info}")
    else:
        print("Failed to load YOLOv8n model")
