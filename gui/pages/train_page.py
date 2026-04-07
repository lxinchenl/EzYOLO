# -*- coding: utf-8 -*-


from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
    QCheckBox, QSlider, QProgressBar, QTextEdit, QSplitter,
    QTabWidget, QFileDialog, QMessageBox, QScrollArea, QFrame,
    QInputDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QSettings
import os
import json
import shutil
from typing import Dict, List, Optional

from gui.styles import COLORS
from models.database import db

# Matplotlib 导入
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# 真实的Ultralytics模型配置（从model_info.py获取）
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
        "tasks": ["detect", "classify", "pose", "segment", "world"],
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
        "tasks": ["detect", "classify", "pose", "segment"],
        "prefix": "yolo11",
    },
    "YOLOv12": {
        "sizes": ["n", "s", "m", "l", "x"],
        "tasks": ["detect"],
        "prefix": "yolo12",
    },
    "YOLOv26": {
        "sizes": ["n", "s", "m", "l", "x"],
        "tasks": ["detect", "classify", "pose", "segment"],
        "prefix": "yolo26",
    },
}

# 任务类型显示名称
TASK_NAMES = {
    "detect": "目标检测",
    "segment": "实例分割",
    "classify": "图像分类",
    "pose": "姿态估计",
    "world": "开放词汇检测",
}

# 型号显示名称
SIZE_NAMES = {
    "n": "nano (超轻量)",
    "s": "small (轻量)",
    "m": "medium (中等)",
    "l": "large (大)",
    "x": "xlarge (超大)",
    "nu": "nano-u (超轻量新版)",
    "su": "small-u (轻量新版)",
    "mu": "medium-u (中等新版)",
    "lu": "large-u (大新版)",
    "xu": "xlarge-u (超大新版)",
    "tiny": "tiny (超轻量)",
    "t": "tiny (超轻量)",
    "c": "compact (紧凑)",
    "e": "extended (扩展)",
    "b": "balanced (平衡)",
    "u": "ultra (超大)",
}

NO_TEMPLATE_OPTION = "不使用模板"


class NoWheelSpinBox(QSpinBox):
    """未聚焦时忽略滚轮，交给外层滚动区域处理。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """未聚焦时忽略滚轮，交给外层滚动区域处理。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoWheelComboBox(QComboBox):
    """未聚焦时忽略滚轮，交给外层滚动区域处理。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoWheelSlider(QSlider):
    """未聚焦时忽略滚轮，交给外层滚动区域处理。"""

    def __init__(self, orientation, *args, **kwargs):
        super().__init__(orientation, *args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class TrainingThread(QThread):
    """训练后台线程"""
    
    epoch_started = pyqtSignal(int, int)
    epoch_finished = pyqtSignal(int, dict)
    batch_progress = pyqtSignal(int, int)
    training_finished = pyqtSignal(bool, str)
    log_message = pyqtSignal(str)
    metrics_updated = pyqtSignal(dict)  # 新增：指标更新信号
    
    def __init__(self, config: dict, project_id: int = None):
        super().__init__()
        self.config = config
        self.project_id = project_id
        self._is_running = False
        self._is_paused = False
        self.metrics_history = []  # 存储训练指标历史
    
    def run(self):
        """运行训练"""
        self._is_running = True
        
        try:
            # 尝试使用Ultralytics进行真实训练
            self.log_message.emit("开始训练...")
            
            try:
                from ultralytics import YOLO
                self.log_message.emit("✓ 检测到Ultralytics库，开始训练...")
                self.run_real_training()
            except ImportError as e:
                self.log_message.emit(f"✗ 未检测到Ultralytics库: {e}")
                self.log_message.emit("请安装: pip install ultralytics")
                self.training_finished.emit(False, "未安装Ultralytics库")
                
        except Exception as e:
            self.log_message.emit(f"训练出错: {str(e)}")
            import traceback
            self.log_message.emit(traceback.format_exc())
            self.training_finished.emit(False, f"训练出错: {str(e)}")
    
    def run_real_training(self):
        """运行真实YOLO训练"""
        from ultralytics import YOLO
        
        # 构建模型名称
        model_prefix = self.config['model_prefix']
        model_size = self.config['model_size']
        task = self.config['task']
        
        # 任务类型到后缀的映射
        task_suffix_map = {
            "segment": "seg",
            "classify": "cls",
            "pose": "pose",
            "world": "world"
        }
        
        # 构建模型文件名
        if task == 'detect':
            model_name = f"{model_prefix}{model_size}.pt"
        else:
            suffix = task_suffix_map.get(task, task)
            model_name = f"{model_prefix}{model_size}-{suffix}.pt"
        
        # 从pretrained目录加载（使用相对路径）
        import os
        from pathlib import Path
        app_root = Path(__file__).parent.parent.parent  # 向上三级到EzYOLO根目录
        pretrained_dir = app_root / "pretrained"
        model_path = os.path.join(pretrained_dir, model_name)
        
        # 如果本地不存在，则使用模型名称（会自动下载）
        if os.path.exists(model_path):
            self.log_message.emit(f"加载本地模型: {model_path}")
            load_path = model_path
        else:
            self.log_message.emit(f"本地模型不存在: {model_path}")
            self.log_message.emit(f"尝试在线下载: {model_name}")
            load_path = model_name
        
        # 加载预训练模型
        try:
            model = YOLO(load_path)
            self.log_message.emit(f"✓ 模型加载成功")
        except Exception as e:
            self.log_message.emit(f"✗ 模型加载失败: {e}")
            self.training_finished.emit(False, f"模型加载失败: {e}")
            return
        
        # 准备数据配置
        data_yaml = self.prepare_data_yaml()
        if not data_yaml:
            self.log_message.emit("✗ 错误：无法准备数据集")
            self.training_finished.emit(False, "数据集准备失败")
            return
        
        # 训练参数
        epochs = self.config['epochs']
        batch = self.config['batch_size']
        imgsz = self.config['img_size']
        lr = self.config['lr']
        optimizer = self.config['optimizer'].lower()
        
        # 设备选择逻辑
        device = self.config['device']
        if device == '自动选择':
            # 检测是否有可用GPU
            import torch
            if torch.cuda.is_available():
                device = '0'  # 使用第一个GPU
                self.log_message.emit("✓ 检测到可用GPU，使用CUDA:0")
            else:
                device = 'cpu'
                self.log_message.emit("✗ 未检测到可用GPU，使用CPU")
        elif 'CUDA' in device:
            # 手动选择了CUDA设备
            device = device.split(':')[1]  # 提取设备编号
        else:
            # 使用CPU
            device = 'cpu'
        
        self.log_message.emit(f"开始训练 {epochs} epochs...")
        self.log_message.emit(f"  - Batch size: {batch}")
        self.log_message.emit(f"  - Image size: {imgsz}")
        self.log_message.emit(f"  - Learning rate: {lr}")
        self.log_message.emit(f"  - Optimizer: {optimizer}")
        self.log_message.emit(f"  - Device: {'CUDA:' + device if device != 'cpu' else 'cpu'}")
        
        try:
            # 定义训练回调函数
            def on_train_epoch_start(trainer):
                """每个epoch开始时调用"""
                epoch = trainer.epoch + 1
                total_epochs = trainer.epochs
                self.epoch_started.emit(epoch, total_epochs)
                self.log_message.emit(f"\n[Epoch {epoch}/{total_epochs}] 开始训练...")
            
            def on_train_epoch_end(trainer):
                """每个epoch结束时调用"""
                epoch = trainer.epoch + 1
                
                # 获取训练指标
                metrics = {}
                if hasattr(trainer, 'loss_items'):
                    # 确保将张量转换为标量
                    def to_scalar(value):
                        import torch
                        if isinstance(value, torch.Tensor):
                            return value.item()
                        return value
                    
                    metrics['box_loss'] = to_scalar(trainer.loss_items[0]) if len(trainer.loss_items) > 0 else 0
                    metrics['cls_loss'] = to_scalar(trainer.loss_items[1]) if len(trainer.loss_items) > 1 else 0
                    metrics['dfl_loss'] = to_scalar(trainer.loss_items[2]) if len(trainer.loss_items) > 2 else 0
                
                # 获取验证指标 - 使用results_dict属性
                if hasattr(trainer, 'validator') and trainer.validator:
                    val_metrics = trainer.validator.metrics
                    if val_metrics and hasattr(val_metrics, 'results_dict'):
                        results_dict = val_metrics.results_dict
                        # 确保将张量转换为标量
                        def to_scalar(value):
                            import torch
                            if isinstance(value, torch.Tensor):
                                return value.item()
                            return value
                        metrics['map50'] = to_scalar(results_dict.get('metrics/mAP50(B)', 0))
                        metrics['map50_95'] = to_scalar(results_dict.get('metrics/mAP50-95(B)', 0))
                
                metrics['epoch'] = epoch
                self.metrics_history.append(metrics)
                self.epoch_finished.emit(epoch, metrics)
                self.metrics_updated.emit(metrics)
                
                # 输出训练信息
                loss_str = f"box_loss: {metrics.get('box_loss', 0):.4f}, cls_loss: {metrics.get('cls_loss', 0):.4f}"
                if metrics.get('dfl_loss'):
                    loss_str += f", dfl_loss: {metrics['dfl_loss']:.4f}"
                self.log_message.emit(f"  训练损失 - {loss_str}")
                
                if metrics.get('map50'):
                    self.log_message.emit(f"  验证指标 - mAP50: {metrics['map50']:.4f}, mAP50-95: {metrics.get('map50_95', 0):.4f}")
            
            def on_fit_epoch_end(trainer):
                """每个fit epoch结束时调用（包含验证）"""
                pass
            
            # 注册回调
            model.add_callback('on_train_epoch_start', on_train_epoch_start)
            model.add_callback('on_train_epoch_end', on_train_epoch_end)
            
            # 开始训练
            self.log_message.emit("=" * 60)
            self.log_message.emit("开始训练...")
            self.log_message.emit("=" * 60)
            
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                lr0=lr,
                optimizer=optimizer,
                device=device,
                workers=self.config.get('workers', 4),
                verbose=True,
                project='runs',
                name=f'train/exp_{self.project_id}' if self.project_id else 'train/exp',
                exist_ok=True,
                mosaic=self.config.get('mosaic', True),
                mixup=self.config.get('mixup', 0.0),
                hsv_h=self.config.get('hsv_strength', 50) / 100.0 if self.config.get('hsv', False) else 0.0,
                hsv_s=self.config.get('hsv_strength', 50) / 100.0 if self.config.get('hsv', False) else 0.0,
                hsv_v=self.config.get('hsv_strength', 50) / 100.0 if self.config.get('hsv', False) else 0.0,
                fliplr=0.5 if self.config.get('flip', True) else 0.0,
                degrees=10.0 if self.config.get('rotate', False) else 0.0,
            )
            
            # 获取训练结果
            final_map50 = results.results_dict.get('metrics/mAP50(B)', 0)
            final_map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0)
            
            self.log_message.emit("=" * 60)
            self.log_message.emit(f"✓ 训练完成！")
            self.log_message.emit(f"  - mAP50: {final_map50:.4f}")
            self.log_message.emit(f"  - mAP50-95: {final_map50_95:.4f}")
            self.log_message.emit("=" * 60)
            
            if self._is_running:
                self.training_finished.emit(True, f"训练完成！mAP50: {final_map50:.4f}")
            else:
                self.training_finished.emit(False, "训练已停止")
                
        except Exception as e:
            self.log_message.emit(f"✗ 训练过程出错: {e}")
            import traceback
            self.log_message.emit(traceback.format_exc())
            self.training_finished.emit(False, f"训练出错: {e}")
    
    def prepare_data_yaml(self) -> str:
        """准备YOLO数据配置文件"""
        try:
            if not self.project_id:
                self.log_message.emit("✗ 错误：未选择项目")
                return None
            
            # 获取项目信息
            project = db.get_project(self.project_id)
            if not project:
                self.log_message.emit("✗ 错误：无法获取项目信息")
                return None
            
            # 创建数据集目录（使用基于应用根目录的相对路径）
            import os
            import shutil
            from pathlib import Path
            app_root = Path(__file__).parent.parent.parent  # 向上三级到EzYOLO根目录
            dataset_dir = app_root / f"datasets/project_{self.project_id}"
            
            # 清空原有训练数据目录
            if os.path.exists(dataset_dir):
                self.log_message.emit(f"清空原有训练数据目录: {dataset_dir}")
                shutil.rmtree(dataset_dir)
            
            # 重新创建目录结构
            os.makedirs(dataset_dir, exist_ok=True)
            
            # 创建images和labels目录
            for split in ['train', 'val', 'test']:
                os.makedirs(f"{dataset_dir}/images/{split}", exist_ok=True)
                os.makedirs(f"{dataset_dir}/labels/{split}", exist_ok=True)
            
            # 获取项目图片
            images = db.get_project_images(self.project_id)
            if not images:
                self.log_message.emit("✗ 错误：项目中没有图片")
                return None
            
            self.log_message.emit(f"准备数据集: 共 {len(images)} 张图片")
            
            # 划分数据集
            import random
            random.seed(42)  # 保证可重复
            random.shuffle(images)
            
            total = len(images)
            
            # 获取分割比例
            train_ratio = self.config.get('train_split', 80)
            val_ratio = self.config.get('val_split', 10)
            test_ratio = self.config.get('test_split', 10)
            
            # 计算数量，确保每个集合至少有一张图片
            train_num = max(1, int(total * train_ratio / 100))
            val_num = max(1, int(total * val_ratio / 100))
            test_num = max(0, total - train_num - val_num)
            
            # 如果图片太少，调整分配
            if total < 3:
                train_num = total
                val_num = 0
                test_num = 0
            elif train_num + val_num > total:
                train_num = total - 1
                val_num = 1
                test_num = 0
            
            train_images = images[:train_num]
            val_images = images[train_num:train_num + val_num]
            test_images = images[train_num + val_num:]
            
            self.log_message.emit(f"  - 训练集: {len(train_images)} 张, 验证集: {len(val_images)} 张, 测试集: {len(test_images)} 张")
            
            # 复制图片和标注
            copied_count = {'train': 0, 'val': 0, 'test': 0}
            for split, img_list in [('train', train_images), ('val', val_images), ('test', test_images)]:
                for img in img_list:
                    # 复制图片
                    src_img = img.get('storage_path', '')
                    filename = img.get('filename', '')
                    if src_img and os.path.exists(src_img):
                        dst_img = f"{dataset_dir}/images/{split}/{filename}"
                        try:
                            shutil.copy2(src_img, dst_img)
                            copied_count[split] += 1
                        except Exception:
                            pass
                    
                    # 复制标注
                    annotations = db.get_image_annotations(img['id'])
                    if annotations:
                        label_file = f"{dataset_dir}/labels/{split}/{os.path.splitext(img['filename'])[0]}.txt"
                        try:
                            with open(label_file, 'w') as f:
                                for ann in annotations:
                                    ann_type = ann.get('type', 'unknown')
                                    data = ann.get('data', {})
                                    if not data:
                                        continue
                                    
                                    # 转换为YOLO格式 - 直接从img字典获取图片尺寸
                                    img_w = img.get('width', 640)
                                    img_h = img.get('height', 480)
                                    class_id = ann.get('class_id', 0)
                                    
                                    # 根据标注类型生成不同格式的标注
                                    if ann_type == 'bbox':
                                        # 边界框标注
                                        x = data.get('x', 0)
                                        y = data.get('y', 0)
                                        w = data.get('width', 0)
                                        h = data.get('height', 0)
                                        
                                        x_center = (x + w / 2) / img_w
                                        y_center = (y + h / 2) / img_h
                                        width = w / img_w
                                        height = h / img_h
                                        
                                        line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                                        f.write(line)
                                    elif ann_type == 'polygon':
                                        # 多边形标注（用于segment任务）
                                        points = data.get('points', [])
                                        if points:
                                            # 写入类别ID
                                            line = f"{class_id} "
                                            # 写入多边形点
                                            for point in points:
                                                px = point.get('x', 0) / img_w
                                                py = point.get('y', 0) / img_h
                                                line += f"{px:.6f} {py:.6f} "
                                            line += "\n"
                                            f.write(line)
                                    elif ann_type == 'keypoint':
                                        # 关键点标注（用于pose任务）
                                        x = data.get('x', 0)
                                        y = data.get('y', 0)
                                        w = data.get('width', 0)
                                        h = data.get('height', 0)
                                        keypoints = data.get('keypoints', [])
                                        
                                        x_center = (x + w / 2) / img_w
                                        y_center = (y + h / 2) / img_h
                                        width = w / img_w
                                        height = h / img_h
                                        
                                        # 写入边界框
                                        line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} "
                                        
                                        # 写入关键点
                                        for kp in keypoints:
                                            kp_x = kp.get('x', 0) / img_w
                                            kp_y = kp.get('y', 0) / img_h
                                            kp_v = kp.get('v', 1)
                                            line += f"{kp_x:.6f} {kp_y:.6f} {kp_v} "
                                        line += "\n"
                                        f.write(line)
                                    elif ann_type == 'obb':
                                        # 旋转框标注（用于obb任务）
                                        # 检查是否有八个角点的坐标
                                        if 'points' in data and len(data['points']) == 4:
                                            # 直接使用八个角点格式（x1 y1 x2 y2 x3 y3 x4 y4）
                                            line = f"{class_id} "
                                            for point in data['points']:
                                                px = point.get('x', 0) / img_w
                                                py = point.get('y', 0) / img_h
                                                line += f"{px:.6f} {py:.6f} "
                                            line += "\n"
                                            f.write(line)
                                        else:
                                            # 如果没有八个角点坐标，使用默认的xywhr格式
                                            x = data.get('x', 0)
                                            y = data.get('y', 0)
                                            w = data.get('width', 0)
                                            h = data.get('height', 0)
                                            angle = data.get('angle', 0)
                                            
                                            x_center = (x + w / 2) / img_w
                                            y_center = (y + h / 2) / img_h
                                            width = w / img_w
                                            height = h / img_h
                                            
                                            line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {angle:.6f}\n"
                                            f.write(line)
                        except Exception:
                            pass
            
            self.log_message.emit(f"  - 复制完成: 训练集 {copied_count['train']} 张, 验证集 {copied_count['val']} 张, 测试集 {copied_count['test']} 张")
            
            # 创建data.yaml
            if isinstance(project, dict):
                classes_json = project.get('classes', '[]')
            else:
                classes_json = project.classes if hasattr(project, 'classes') else '[]'
            
            classes = json.loads(classes_json) if classes_json else []
            class_names = [c['name'] for c in classes]
            
            yaml_content = f"""path: {os.path.abspath(dataset_dir)}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
            
            yaml_path = f"{dataset_dir}/data.yaml"
            with open(yaml_path, 'w') as f:
                f.write(yaml_content)
            
            self.log_message.emit(f"✓ 数据集准备完成: {yaml_path}")
            return yaml_path
            
        except Exception as e:
            self.log_message.emit(f"✗ 准备数据集出错: {e}")
            import traceback
            self.log_message.emit(traceback.format_exc())
            return None
    
    def pause(self):
        """暂停训练"""
        self._is_paused = True
        self.log_message.emit("训练已暂停")
    
    def resume(self):
        """恢复训练"""
        self._is_paused = False
        self.log_message.emit("训练已恢复")
    
    def stop(self):
        """停止训练"""
        self._is_running = False
        self._is_paused = False


class TrainPage(QWidget):
    """训练页面"""
    
    def __init__(self):
        super().__init__()
        self.current_project_id = None
        self.training_thread = None
        self.training_history = []
        self.settings = QSettings("EzYOLO", "Settings")
        self.training_templates = {}
        
        self.init_ui()
        self.load_training_templates()
    
    def set_project(self, project_id: int):
        """设置当前项目"""
        self.current_project_id = project_id
        if project_id:
            project = db.get_project(project_id)
            if project:
                project_name = project.get('name', 'Unknown') if isinstance(project, dict) else project.name
                print(f"[TrainPage] 已切换到项目: {project_name} (ID: {project_id})")
        else:
            print("[TrainPage] 项目已取消选择")
    
    def init_ui(self):
        """初始化界面"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        
        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：配置面板
        left_panel = self.create_config_panel()
        splitter.addWidget(left_panel)
        
        # 右侧：监控面板
        right_panel = self.create_monitor_panel()
        splitter.addWidget(right_panel)
        
        # 设置分割比例
        splitter.setSizes([400, 800])
        
        main_layout.addWidget(splitter)
    
    def _init_model_lists(self):
        """初始化型号和任务列表"""
        version = self.model_version.currentText()
        
        if not version:
            # 如果没有获取到版本，使用第一个可用版本
            version = sorted(ULTRALYTICS_MODELS.keys())[0]
            self.model_version.setCurrentText(version)
        
        if version in ULTRALYTICS_MODELS:
            model_info = ULTRALYTICS_MODELS[version]
            
            # 初始化型号列表
            self.model_size.clear()
            for size in model_info['sizes']:
                display_name = SIZE_NAMES.get(size, size)
                self.model_size.addItem(display_name, size)
            
            # 初始化任务列表
            self.task_type.clear()
            for task in model_info['tasks']:
                display_name = TASK_NAMES.get(task, task)
                self.task_type.addItem(display_name, task)
    
    def create_config_panel(self) -> QWidget:
        """创建配置面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        
        # 标题
        title = QLabel("训练配置")
        title.setObjectName("title")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        template_bar = self.create_template_bar()
        layout.addWidget(template_bar)
        
        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(16)
        
        # 模型选择组
        model_group = self.create_model_group()
        scroll_layout.addWidget(model_group)
        
        # 训练参数组
        params_group = self.create_params_group()
        scroll_layout.addWidget(params_group)
        
        # 数据增强组
        augment_group = self.create_augment_group()
        scroll_layout.addWidget(augment_group)
        
        # 数据集划分组
        split_group = self.create_split_group()
        scroll_layout.addWidget(split_group)
        
        # 控制按钮组
        control_group = self.create_control_group()
        scroll_layout.addWidget(control_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        return panel

    def create_template_bar(self) -> QWidget:
        """创建训练模板工具条。"""
        bar = QFrame()
        bar.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['panel']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
            }}
        """)
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        top_row.addWidget(QLabel("训练模板:"))

        self.template_combo = NoWheelComboBox()
        self.template_combo.currentTextChanged.connect(self.on_template_selection_changed)
        top_row.addWidget(self.template_combo, 1)
        layout.addLayout(top_row)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        self.btn_apply_template = QPushButton("套用模板")
        self.btn_apply_template.clicked.connect(self.apply_selected_template)
        button_row.addWidget(self.btn_apply_template)

        self.btn_save_template = QPushButton("保存为模板")
        self.btn_save_template.clicked.connect(self.save_current_as_template)
        button_row.addWidget(self.btn_save_template)

        self.btn_update_template = QPushButton("更新模板")
        self.btn_update_template.clicked.connect(self.update_selected_template)
        button_row.addWidget(self.btn_update_template)

        self.btn_delete_template = QPushButton("删除模板")
        self.btn_delete_template.clicked.connect(self.delete_selected_template)
        button_row.addWidget(self.btn_delete_template)

        layout.addLayout(button_row)
        return bar
    
    def get_group_style(self) -> str:
        """获取分组框样式"""
        return f"""
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """
    
    def create_model_group(self) -> QGroupBox:
        """创建模型选择组"""
        group = QGroupBox("模型选择")
        group.setStyleSheet(self.get_group_style())
        
        layout = QFormLayout(group)
        layout.setSpacing(10)
        
        # YOLO版本选择
        self.model_version = NoWheelComboBox()
        self.model_version.addItems(sorted(ULTRALYTICS_MODELS.keys()))
        self.model_version.currentTextChanged.connect(self.on_version_changed)
        layout.addRow("版本:", self.model_version)
        
        # 模型型号选择
        self.model_size = NoWheelComboBox()
        layout.addRow("型号:", self.model_size)
        
        # 任务类型
        self.task_type = NoWheelComboBox()
        layout.addRow("任务:", self.task_type)
        
        # 立即初始化型号和任务列表
        self._init_model_lists()
        
        return group
    
    def on_version_changed(self, version: str):
        """版本改变时更新型号和任务"""
        if not version or version not in ULTRALYTICS_MODELS:
            return
        
        model_info = ULTRALYTICS_MODELS[version]
        
        # 更新型号列表
        try:
            self.model_size.clear()
            for size in model_info['sizes']:
                display_name = SIZE_NAMES.get(size, size)
                self.model_size.addItem(display_name, size)
        except RuntimeError:
            return
        
        # 更新任务列表
        try:
            self.task_type.clear()
            for task in model_info['tasks']:
                display_name = TASK_NAMES.get(task, task)
                self.task_type.addItem(display_name, task)
        except RuntimeError:
            return
    
    def create_params_group(self) -> QGroupBox:
        """创建训练参数组"""
        group = QGroupBox("训练参数")
        group.setStyleSheet(self.get_group_style())
        
        layout = QFormLayout(group)
        layout.setSpacing(10)
        
        # Epochs
        self.epochs = NoWheelSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(100)
        layout.addRow("Epochs:", self.epochs)
        
        # Batch Size
        self.batch_size = NoWheelSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(16)
        layout.addRow("Batch Size:", self.batch_size)
        
        # Image Size
        self.img_size = NoWheelSpinBox()
        self.img_size.setRange(320, 1280)
        self.img_size.setValue(640)
        self.img_size.setSingleStep(32)
        layout.addRow("Image Size:", self.img_size)
        
        # Learning Rate
        self.lr = NoWheelDoubleSpinBox()
        self.lr.setRange(0.0001, 0.1)
        self.lr.setValue(0.01)
        self.lr.setDecimals(4)
        self.lr.setSingleStep(0.001)
        layout.addRow("Learning Rate:", self.lr)
        
        # Optimizer
        self.optimizer = NoWheelComboBox()
        self.optimizer.addItems(["SGD", "Adam", "AdamW", "LION"])
        layout.addRow("Optimizer:", self.optimizer)
        
        # Device
        self.device = NoWheelComboBox()
        self.device.addItems(["自动选择", "CPU", "CUDA:0", "CUDA:1", "CUDA:2", "CUDA:3"])
        layout.addRow("Device:", self.device)
        
        # Workers
        self.workers = NoWheelSpinBox()
        self.workers.setRange(0, 32)
        self.workers.setValue(4)
        self.workers.setSingleStep(1)
        layout.addRow("Workers:", self.workers)
        
        return group
    
    def create_augment_group(self) -> QGroupBox:
        """创建数据增强组"""
        group = QGroupBox("数据增强")
        group.setStyleSheet(self.get_group_style())
        
        layout = QFormLayout(group)
        layout.setSpacing(10)
        
        # Mosaic
        self.mosaic = QCheckBox("启用 Mosaic 增强")
        self.mosaic.setChecked(True)
        layout.addRow(self.mosaic)
        
        # MixUp
        self.mixup = QCheckBox("启用 MixUp 增强")
        layout.addRow(self.mixup)
        
        # 随机翻转
        self.flip = QCheckBox("启用随机水平翻转")
        self.flip.setChecked(True)
        layout.addRow(self.flip)
        
        # 随机旋转
        self.rotate = QCheckBox("启用随机旋转")
        layout.addRow(self.rotate)
        
        # HSV增强
        hsv_layout = QHBoxLayout()
        self.hsv = QCheckBox("HSV增强")
        self.hsv.setChecked(True)
        hsv_layout.addWidget(self.hsv)
        self.hsv_strength = NoWheelSlider(Qt.Orientation.Horizontal)
        self.hsv_strength.setRange(0, 100)
        self.hsv_strength.setValue(50)
        hsv_layout.addWidget(self.hsv_strength)
        layout.addRow(hsv_layout)
        
        return group
    
    def create_split_group(self) -> QGroupBox:
        """创建数据集划分组"""
        group = QGroupBox("数据集划分")
        group.setStyleSheet(self.get_group_style())
        
        layout = QFormLayout(group)
        layout.setSpacing(10)
        
        # 训练集比例
        self.train_split = NoWheelSlider(Qt.Orientation.Horizontal)
        self.train_split.setRange(50, 95)
        self.train_split.setValue(80)
        self.train_split.valueChanged.connect(self.on_split_changed)
        self.train_label = QLabel("80%")
        split_layout = QHBoxLayout()
        split_layout.addWidget(self.train_split)
        split_layout.addWidget(self.train_label)
        layout.addRow("训练集:", split_layout)
        
        # 验证集比例
        self.val_split = NoWheelSlider(Qt.Orientation.Horizontal)
        self.val_split.setRange(5, 30)
        self.val_split.setValue(10)
        self.val_split.valueChanged.connect(self.on_split_changed)
        self.val_label = QLabel("10%")
        split_layout = QHBoxLayout()
        split_layout.addWidget(self.val_split)
        split_layout.addWidget(self.val_label)
        layout.addRow("验证集:", split_layout)
        
        # 测试集比例
        self.test_split = NoWheelSlider(Qt.Orientation.Horizontal)
        self.test_split.setRange(0, 20)
        self.test_split.setValue(10)
        self.test_split.valueChanged.connect(self.on_split_changed)
        self.test_label = QLabel("10%")
        split_layout = QHBoxLayout()
        split_layout.addWidget(self.test_split)
        split_layout.addWidget(self.test_label)
        layout.addRow("测试集:", split_layout)
        
        # 总和提示
        self.split_warning = QLabel("")
        self.split_warning.setStyleSheet(f"color: {COLORS['error']}; font-size: 12px;")
        layout.addRow(self.split_warning)
        
        return group
    
    def on_split_changed(self):
        """数据集划分改变"""
        train = self.train_split.value()
        val = self.val_split.value()
        test = self.test_split.value()
        
        self.train_label.setText(f"{train}%")
        self.val_label.setText(f"{val}%")
        self.test_label.setText(f"{test}%")
        
        total = train + val + test
        if total != 100:
            self.split_warning.setText(f"⚠️ 总和为 {total}%，应为 100%")
        else:
            self.split_warning.setText("")
    
    def create_control_group(self) -> QGroupBox:
        """创建控制按钮组"""
        group = QGroupBox("训练控制")
        group.setStyleSheet(self.get_group_style())
        
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS['sidebar']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                text-align: center;
                color: white;
                height: 20px;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        # 按钮布局
        btn_layout = QHBoxLayout()
        
        # 开始按钮
        self.btn_start = QPushButton("▶ 开始训练")
        self.btn_start.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                font-weight: bold;
                padding: 12px 30px;
                font-size: 14px;
            }}
        """)
        self.btn_start.clicked.connect(self.start_training)
        btn_layout.addWidget(self.btn_start)
        
        # 居中布局
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addLayout(btn_layout)
        
        return group

    def load_training_templates(self):
        """加载全局训练模板。"""
        raw_templates = self.settings.value("training_templates", "{}")
        try:
            templates = json.loads(raw_templates) if raw_templates else {}
        except (TypeError, json.JSONDecodeError):
            templates = {}

        self.training_templates = templates if isinstance(templates, dict) else {}
        selected_name = str(self.settings.value("training_selected_template", NO_TEMPLATE_OPTION))

        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        self.template_combo.addItem(NO_TEMPLATE_OPTION)
        for template_name in sorted(self.training_templates.keys()):
            self.template_combo.addItem(template_name)

        if selected_name in self.training_templates:
            self.template_combo.setCurrentText(selected_name)
        else:
            self.template_combo.setCurrentText(NO_TEMPLATE_OPTION)
        self.template_combo.blockSignals(False)
        self.refresh_template_ui_state()
        self.save_training_templates()

    def save_training_templates(self):
        """保存全局训练模板。"""
        self.settings.setValue(
            "training_templates",
            json.dumps(self.training_templates, ensure_ascii=False)
        )
        current_name = self.template_combo.currentText() if hasattr(self, "template_combo") else NO_TEMPLATE_OPTION
        self.settings.setValue(
            "training_selected_template",
            current_name if current_name in self.training_templates else NO_TEMPLATE_OPTION
        )

    def on_template_selection_changed(self, _text: str):
        """模板选择变化。"""
        self.refresh_template_ui_state()
        self.save_training_templates()

    def refresh_template_ui_state(self):
        """刷新模板相关按钮状态。"""
        current_name = self.template_combo.currentText() if hasattr(self, "template_combo") else NO_TEMPLATE_OPTION
        has_template = current_name in self.training_templates
        self.btn_apply_template.setEnabled(has_template)
        self.btn_update_template.setEnabled(has_template)
        self.btn_delete_template.setEnabled(has_template)

    def collect_training_form_config(self) -> dict:
        """收集当前表单配置。"""
        return {
            'version': self.model_version.currentText(),
            'model_size': self.model_size.currentData(),
            'task': self.task_type.currentData(),
            'epochs': self.epochs.value(),
            'batch_size': self.batch_size.value(),
            'img_size': self.img_size.value(),
            'lr': self.lr.value(),
            'optimizer': self.optimizer.currentText(),
            'device': self.device.currentText(),
            'workers': self.workers.value(),
            'mosaic': self.mosaic.isChecked(),
            'mixup': self.mixup.isChecked(),
            'flip': self.flip.isChecked(),
            'rotate': self.rotate.isChecked(),
            'hsv': self.hsv.isChecked(),
            'hsv_strength': self.hsv_strength.value(),
            'train_split': self.train_split.value(),
            'val_split': self.val_split.value(),
            'test_split': self.test_split.value(),
        }

    def _set_combo_by_text(self, combo: QComboBox, value: str) -> bool:
        index = combo.findText(value)
        if index < 0:
            return False
        combo.setCurrentIndex(index)
        return True

    def _set_combo_by_data(self, combo: QComboBox, value: str) -> bool:
        index = combo.findData(value)
        if index < 0:
            return False
        combo.setCurrentIndex(index)
        return True

    def apply_training_form_config(self, config: dict) -> bool:
        """将配置写回到训练页面表单。"""
        version = config.get('version')
        if version not in ULTRALYTICS_MODELS:
            QMessageBox.warning(self, "模板无效", f"模板中的模型版本不可用: {version}")
            return False

        if not self._set_combo_by_text(self.model_version, version):
            QMessageBox.warning(self, "模板无效", f"无法切换到模板中的模型版本: {version}")
            return False
        self.on_version_changed(version)

        model_size = config.get('model_size')
        task = config.get('task')
        if not self._set_combo_by_data(self.model_size, model_size):
            QMessageBox.warning(self, "模板无效", f"模板中的模型型号不可用: {model_size}")
            return False
        if not self._set_combo_by_data(self.task_type, task):
            QMessageBox.warning(self, "模板无效", f"模板中的任务类型不可用: {task}")
            return False

        self.epochs.setValue(int(config.get('epochs', self.epochs.value())))
        self.batch_size.setValue(int(config.get('batch_size', self.batch_size.value())))
        self.img_size.setValue(int(config.get('img_size', self.img_size.value())))
        self.lr.setValue(float(config.get('lr', self.lr.value())))
        self._set_combo_by_text(self.optimizer, str(config.get('optimizer', self.optimizer.currentText())))
        self._set_combo_by_text(self.device, str(config.get('device', self.device.currentText())))
        self.workers.setValue(int(config.get('workers', self.workers.value())))

        self.mosaic.setChecked(bool(config.get('mosaic', self.mosaic.isChecked())))
        self.mixup.setChecked(bool(config.get('mixup', self.mixup.isChecked())))
        self.flip.setChecked(bool(config.get('flip', self.flip.isChecked())))
        self.rotate.setChecked(bool(config.get('rotate', self.rotate.isChecked())))
        self.hsv.setChecked(bool(config.get('hsv', self.hsv.isChecked())))
        self.hsv_strength.setValue(int(config.get('hsv_strength', self.hsv_strength.value())))

        self.train_split.setValue(int(config.get('train_split', self.train_split.value())))
        self.val_split.setValue(int(config.get('val_split', self.val_split.value())))
        self.test_split.setValue(int(config.get('test_split', self.test_split.value())))
        self.on_split_changed()
        return True

    def save_current_as_template(self):
        """将当前页面配置保存为模板。"""
        template_name, ok = QInputDialog.getText(self, "保存训练模板", "请输入模板名称:")
        template_name = template_name.strip()
        if not ok or not template_name:
            return
        if template_name == NO_TEMPLATE_OPTION:
            QMessageBox.warning(self, "模板名称无效", f"“{NO_TEMPLATE_OPTION}”是保留名称，请换一个模板名")
            return

        if template_name in self.training_templates:
            reply = QMessageBox.question(
                self,
                "覆盖模板",
                f"模板“{template_name}”已存在，是否覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.training_templates[template_name] = self.collect_training_form_config()
        self.save_training_templates()
        self.load_training_templates()
        self.template_combo.setCurrentText(template_name)

    def update_selected_template(self):
        """用当前页面参数覆盖当前模板。"""
        template_name = self.template_combo.currentText()
        if template_name not in self.training_templates:
            return

        reply = QMessageBox.question(
            self,
            "更新模板",
            f"确认用当前页面参数覆盖模板“{template_name}”？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.training_templates[template_name] = self.collect_training_form_config()
        self.save_training_templates()

    def delete_selected_template(self):
        """删除当前选中的模板。"""
        template_name = self.template_combo.currentText()
        if template_name not in self.training_templates:
            return

        reply = QMessageBox.question(
            self,
            "删除模板",
            f"确认删除模板“{template_name}”？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.training_templates.pop(template_name, None)
        self.save_training_templates()
        self.load_training_templates()

    def apply_selected_template(self):
        """立即套用当前选中的模板。"""
        template_name = self.template_combo.currentText()
        template_config = self.training_templates.get(template_name)
        if not template_config:
            return
        if self.apply_training_form_config(template_config):
            self.template_combo.setCurrentText(template_name)

    def maybe_confirm_template_before_training(self) -> Optional[dict]:
        """训练前确认是否使用当前模板。"""
        template_name = self.template_combo.currentText()
        if template_name not in self.training_templates:
            return self.collect_training_form_config()

        message_box = QMessageBox(self)
        message_box.setIcon(QMessageBox.Icon.Question)
        message_box.setWindowTitle("使用训练模板")
        message_box.setText(f"当前已选择训练模板“{template_name}”，本次训练是否使用该模板？")
        use_template_button = message_box.addButton("使用模板", QMessageBox.ButtonRole.AcceptRole)
        use_current_button = message_box.addButton("使用当前页面参数", QMessageBox.ButtonRole.DestructiveRole)
        cancel_button = message_box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        message_box.setDefaultButton(use_template_button)
        message_box.exec()

        clicked = message_box.clickedButton()
        if clicked == cancel_button:
            return None
        if clicked == use_current_button:
            return self.collect_training_form_config()

        template_config = self.training_templates.get(template_name)
        if not template_config:
            return self.collect_training_form_config()
        if not self.apply_training_form_config(template_config):
            return None
        return self.collect_training_form_config()

    def build_training_runtime_config(self, form_config: dict) -> Optional[dict]:
        """将表单配置转换为训练线程使用的配置。"""
        version = form_config.get('version')
        model_size = form_config.get('model_size')
        task = form_config.get('task')

        if not version or version not in ULTRALYTICS_MODELS or not model_size or not task:
            return None

        return {
            'version': version,
            'model_prefix': ULTRALYTICS_MODELS[version]['prefix'],
            'model_size': model_size,
            'task': task,
            'epochs': form_config['epochs'],
            'batch_size': form_config['batch_size'],
            'img_size': form_config['img_size'],
            'lr': form_config['lr'],
            'optimizer': form_config['optimizer'],
            'device': form_config['device'],
            'workers': form_config['workers'],
            'mosaic': form_config['mosaic'],
            'mixup': 0.1 if form_config['mixup'] else 0.0,
            'flip': form_config['flip'],
            'rotate': form_config['rotate'],
            'hsv': form_config['hsv'],
            'hsv_strength': form_config['hsv_strength'],
            'train_split': form_config['train_split'],
            'val_split': form_config['val_split'],
            'test_split': form_config['test_split'],
        }
    
    def create_monitor_panel(self) -> QWidget:
        """创建监控面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        
        # 标题
        title = QLabel("训练监控")
        title.setObjectName("title")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        
        # 标签页
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                background-color: {COLORS['panel']};
            }}
            QTabBar::tab {{
                background-color: {COLORS['sidebar']};
                color: {COLORS['text_secondary']};
                padding: 8px 16px;
                margin-right: 4px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {COLORS['primary']};
                color: white;
            }}
        """)
        
        # 损失曲线标签页
        loss_tab = self.create_loss_tab()
        tabs.addTab(loss_tab, "📉 损失曲线")
        
        # mAP曲线标签页
        map_tab = self.create_map_tab()
        tabs.addTab(map_tab, "📈 mAP曲线")
        
        # 日志标签页
        log_tab = self.create_log_tab()
        tabs.addTab(log_tab, "📝 训练日志")
        
        layout.addWidget(tabs)
        
        return panel
    
    def create_loss_tab(self) -> QWidget:
        """创建损失曲线标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 创建matplotlib图表
        self.loss_figure = Figure(figsize=(8, 6), dpi=100)
        self.loss_figure.patch.set_facecolor(COLORS['sidebar'])
        self.loss_canvas = FigureCanvas(self.loss_figure)
        layout.addWidget(self.loss_canvas)
        
        # 初始化损失曲线
        self.loss_ax = self.loss_figure.add_subplot(111)
        self.loss_ax.set_facecolor(COLORS['sidebar'])
        self.loss_ax.set_title('Training Loss', color=COLORS['text_primary'], fontsize=12)
        self.loss_ax.set_xlabel('Epoch', color=COLORS['text_primary'])
        self.loss_ax.set_ylabel('Loss', color=COLORS['text_primary'])
        self.loss_ax.tick_params(colors=COLORS['text_primary'])
        self.loss_ax.grid(True, alpha=0.3)
        
        # 初始化空曲线
        self.loss_lines = {
            'box': self.loss_ax.plot([], [], 'b-', label='Box Loss', linewidth=2)[0],
            'cls': self.loss_ax.plot([], [], 'r-', label='Cls Loss', linewidth=2)[0],
            'dfl': self.loss_ax.plot([], [], 'g-', label='DFL Loss', linewidth=2)[0],
        }
        # 设置标签颜色为白色
        self.loss_ax.legend(loc='upper right', facecolor=COLORS['sidebar'], edgecolor=COLORS['border'], labelcolor='white')
        # 设置坐标轴文字颜色为白色
        self.loss_ax.xaxis.label.set_color('white')
        self.loss_ax.yaxis.label.set_color('white')
        # 设置刻度文字颜色为白色
        self.loss_ax.tick_params(axis='x', colors='white')
        self.loss_ax.tick_params(axis='y', colors='white')
        
        return tab
    
    def create_map_tab(self) -> QWidget:
        """创建mAP曲线标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 创建matplotlib图表
        self.map_figure = Figure(figsize=(8, 6), dpi=100)
        self.map_figure.patch.set_facecolor(COLORS['sidebar'])
        self.map_canvas = FigureCanvas(self.map_figure)
        layout.addWidget(self.map_canvas)
        
        # 初始化mAP曲线
        self.map_ax = self.map_figure.add_subplot(111)
        self.map_ax.set_facecolor(COLORS['sidebar'])
        self.map_ax.set_title('Validation mAP', color=COLORS['text_primary'], fontsize=12)
        self.map_ax.set_xlabel('Epoch', color=COLORS['text_primary'])
        self.map_ax.set_ylabel('mAP', color=COLORS['text_primary'])
        self.map_ax.tick_params(colors=COLORS['text_primary'])
        self.map_ax.grid(True, alpha=0.3)
        self.map_ax.set_ylim(0, 1)
        
        # 初始化空曲线
        self.map_lines = {
            'map50': self.map_ax.plot([], [], 'b-', label='mAP50', linewidth=2)[0],
            'map50_95': self.map_ax.plot([], [], 'r-', label='mAP50-95', linewidth=2)[0],
        }
        # 设置标签颜色为白色
        self.map_ax.legend(loc='lower right', facecolor=COLORS['sidebar'], edgecolor=COLORS['border'], labelcolor='white')
        # 设置坐标轴文字颜色为白色
        self.map_ax.xaxis.label.set_color('white')
        self.map_ax.yaxis.label.set_color('white')
        # 设置刻度文字颜色为白色
        self.map_ax.tick_params(axis='x', colors='white')
        self.map_ax.tick_params(axis='y', colors='white')
        
        return tab
    
    def create_log_tab(self) -> QWidget:
        """创建日志标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 日志文本框
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['sidebar']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 10px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
            }}
        """)
        layout.addWidget(self.log_text)
        
        # 日志按钮
        btn_layout = QHBoxLayout()
        
        self.btn_clear_log = QPushButton("🗑 清空日志")
        self.btn_clear_log.clicked.connect(self.clear_log)
        btn_layout.addWidget(self.btn_clear_log)
        
        self.btn_save_log = QPushButton("💾 保存日志")
        self.btn_save_log.clicked.connect(self.save_log)
        btn_layout.addWidget(self.btn_save_log)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        return tab
    
    def start_training(self):
        """开始训练"""
        # 检查是否选择了项目
        if not self.current_project_id:
            QMessageBox.warning(self, "错误", "请先选择一个项目")
            return

        form_config = self.maybe_confirm_template_before_training()
        if form_config is None:
            return

        train = form_config['train_split']
        val = form_config['val_split']
        test = form_config['test_split']
        if train + val + test != 100:
            QMessageBox.warning(self, "配置错误", "数据集划分比例总和必须等于100%")
            return

        config = self.build_training_runtime_config(form_config)
        if not config:
            QMessageBox.warning(self, "错误", "请选择有效的模型版本和型号")
            return

        version = config['version']
        model_size = config['model_size']
        task = config['task']
        
        # 清空历史
        self.training_history = []
        self.log_text.clear()
        
        # 清空曲线数据
        self.clear_plots()
        
        # 创建训练线程
        self.training_thread = TrainingThread(config, self.current_project_id)
        self.training_thread.epoch_started.connect(self.on_epoch_started)
        self.training_thread.epoch_finished.connect(self.on_epoch_finished)
        self.training_thread.batch_progress.connect(self.on_batch_progress)
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.log_message.connect(self.on_log_message)
        self.training_thread.metrics_updated.connect(self.on_metrics_updated)
        
        # 更新UI状态
        self.btn_start.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(config['epochs'])
        self.progress_bar.setValue(0)
        self.status_label.setText("训练中...")
        self.status_label.setStyleSheet(f"color: {COLORS['primary']}; font-size: 12px;")
        
        # 启动训练
        self.training_thread.start()
        
        self.log_message("=" * 50)
        self.log_message("训练开始！")
        self.log_message(f"模型: {version} {model_size} ({task})")
        self.log_message(f"Epochs: {config['epochs']}, Batch: {config['batch_size']}")
        self.log_message("=" * 50)
    
    def pause_training(self):
        """暂停/恢复训练"""
        if self.training_thread:
            if self.btn_pause.text() == "⏸ 暂停":
                self.training_thread.pause()
                self.btn_pause.setText("▶ 继续")
                self.status_label.setText("已暂停")
                self.status_label.setStyleSheet(f"color: {COLORS['warning']}; font-size: 12px;")
            else:
                self.training_thread.resume()
                self.btn_pause.setText("⏸ 暂停")
                self.status_label.setText("训练中...")
                self.status_label.setStyleSheet(f"color: {COLORS['primary']}; font-size: 12px;")
    
    def stop_training(self):
        """停止训练"""
        if self.training_thread:
            self.training_thread.stop()
            self.training_thread.wait()
        
        self.reset_ui_state()
        self.log_message("训练已停止")
    
    def reset_ui_state(self):
        """重置UI状态"""
        self.btn_start.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("就绪")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
    
    def on_epoch_started(self, epoch: int, total: int):
        """Epoch开始"""
        self.progress_bar.setValue(epoch - 1)
        self.status_label.setText(f"训练中... Epoch {epoch}/{total}")
    
    def on_epoch_finished(self, epoch: int, metrics: dict):
        """Epoch完成"""
        # 确保所有指标值都是标量
        def to_scalar(value):
            import torch
            if isinstance(value, torch.Tensor):
                return value.item()
            return value
        
        # 转换所有指标值
        converted_metrics = {}
        for key, value in metrics.items():
            converted_metrics[key] = to_scalar(value)
        
        self.training_history.append({
            'epoch': epoch,
            'metrics': converted_metrics
        })
    
    def on_metrics_updated(self, metrics: dict):
        """指标更新 - 实时更新曲线"""
        self.update_plots()
    
    def clear_plots(self):
        """清空曲线图"""
        # 清空损失曲线
        for line in self.loss_lines.values():
            line.set_data([], [])
        self.loss_ax.set_xlim(0, 1)
        self.loss_ax.set_ylim(0, 1)
        self.loss_canvas.draw()
        
        # 清空mAP曲线
        for line in self.map_lines.values():
            line.set_data([], [])
        self.map_ax.set_xlim(0, 1)
        self.map_ax.set_ylim(0, 1)
        self.map_canvas.draw()
    
    def update_plots(self):
        """更新曲线图"""
        if not self.training_history:
            return
        
        epochs = [h['epoch'] for h in self.training_history]
        
        # 更新损失曲线
        box_losses = [h['metrics'].get('box_loss', 0) for h in self.training_history]
        cls_losses = [h['metrics'].get('cls_loss', 0) for h in self.training_history]
        dfl_losses = [h['metrics'].get('dfl_loss', 0) for h in self.training_history]
        
        self.loss_lines['box'].set_data(epochs, box_losses)
        self.loss_lines['cls'].set_data(epochs, cls_losses)
        if any(dfl_losses):
            self.loss_lines['dfl'].set_data(epochs, dfl_losses)
        
        self.loss_ax.set_xlim(0, max(epochs) + 1)
        all_losses = box_losses + cls_losses + dfl_losses
        if all_losses:
            self.loss_ax.set_ylim(0, max(all_losses) * 1.1)
        self.loss_canvas.draw()
        
        # 更新mAP曲线
        map50_values = [h['metrics'].get('map50', 0) for h in self.training_history]
        map50_95_values = [h['metrics'].get('map50_95', 0) for h in self.training_history]
        
        # 更新图表数据
        self.map_lines['map50'].set_data(epochs, map50_values)
        if any(map50_95_values):
            self.map_lines['map50_95'].set_data(epochs, map50_95_values)
        
        self.map_ax.set_xlim(0, max(epochs) + 1)
        self.map_ax.set_ylim(0, 1)
        self.map_canvas.draw()
    
    def on_batch_progress(self, current: int, total: int):
        """Batch进度"""
        pass
    
    def on_training_finished(self, success: bool, message: str):
        """训练完成"""
        self.reset_ui_state()
        
        if success:
            QMessageBox.information(self, "训练完成", message)
        else:
            QMessageBox.warning(self, "训练结束", message)
    
    def on_log_message(self, message: str):
        """日志消息"""
        self.log_message(message)
    
    def log_message(self, message: str):
        """添加日志"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_log(self):
        """清空日志"""
        self.log_text.clear()
    
    def save_log(self):
        """保存日志"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存日志", "training_log.txt",
            "文本文件 (*.txt);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                QMessageBox.information(self, "保存成功", f"日志已保存到:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存日志时出错:\n{str(e)}")
