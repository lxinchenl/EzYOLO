# -*- coding: utf-8 -*-
"""
测试页面 - 模型推理测试
支持图片、图片文件夹、视频的推理测试
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QGroupBox, QFormLayout,
    QProgressBar, QTextEdit, QSplitter, QFileDialog, QMessageBox,
    QScrollArea, QFrame, QSpinBox, QCheckBox, QSlider, QListWidget,
    QListWidgetItem, QStackedWidget, QTabWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from pathlib import Path
import os
import json
import shutil
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from gui.styles import COLORS
from models.database import db

# 从train_page导入模型配置
from gui.pages.train_page import ULTRALYTICS_MODELS, TASK_NAMES, SIZE_NAMES


class InferenceThread(QThread):
    """推理后台线程"""
    
    progress_updated = pyqtSignal(int, int)  # 当前进度, 总数
    inference_finished = pyqtSignal(bool, str)  # 是否成功, 消息
    log_message = pyqtSignal(str)
    result_ready = pyqtSignal(dict)  # 单张图片推理结果
    frame_ready = pyqtSignal(np.ndarray, dict)  # 视频帧和检测结果
    
    def __init__(self, model_path: str, config: dict, data_paths: List[str], 
                 project_classes: List[dict] = None, class_mapping: dict = None):
        super().__init__()
        self.model_path = model_path
        self.config = config
        self.data_paths = data_paths
        self.project_classes = project_classes or []
        self.class_mapping = class_mapping or {}  # 类别映射
        self._is_running = False
        self.model = None
        self.output_root = Path(__file__).parent.parent.parent / "outputs"
        self.image_output_dir = self.output_root / "test_images"
        self.video_output_dir = self.output_root / "test_videos"
        
    def run(self):
        """运行推理"""
        self._is_running = True
        
        try:
            from ultralytics import YOLO
            self.image_output_dir.mkdir(parents=True, exist_ok=True)
            self.video_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 加载模型
            self.log_message.emit(f"加载模型: {self.model_path}")
            
            # 检测模型类型
            model_type = "PyTorch"
            if self.model_path.endswith('.onnx'):
                model_type = "ONNX"
            elif self.model_path.endswith('.engine') or self.model_path.endswith('.trt'):
                model_type = "TensorRT"
            self.log_message.emit(f"模型类型: {model_type}")
            
            try:
                # 获取任务类型，用于ONNX/TensorRT模型
                task = self.config.get('task', 'detect')
                
                # 对于ONNX和TensorRT模型，需要传入task参数
                if model_type in ["ONNX", "TensorRT"]:
                    self.log_message.emit(f"使用任务类型: {task}")
                    self.model = YOLO(self.model_path, task=task)
                else:
                    self.model = YOLO(self.model_path)
                self.log_message.emit("✓ 模型加载成功")
            except Exception as e:
                self.log_message.emit(f"✗ 模型加载失败: {e}")
                # 针对不同模型类型给出更详细的错误提示
                if model_type == "ONNX":
                    self.log_message.emit("提示: ONNX模型需要安装onnxruntime (pip install onnxruntime)")
                elif model_type == "TensorRT":
                    self.log_message.emit("提示: TensorRT模型需要安装tensorrt并配置CUDA环境")
                self.inference_finished.emit(False, f"模型加载失败: {e}")
                return
            
            # 获取模型类别信息
            model_classes = self.model.names if hasattr(self.model, 'names') else {}
            self.log_message.emit(f"模型类别数: {len(model_classes)}")
            
            # 输出类别映射信息
            if self.class_mapping:
                self.log_message.emit(f"类别映射: {self.class_mapping}")
            
            # 推理参数
            conf = self.config.get('conf', 0.25)
            iou = self.config.get('iou', 0.45)
            imgsz = self.config.get('imgsz', 640)
            device = self.config.get('device', 'cpu')
            
            # 标准化设备值
            device = str(device).lower().strip()
            
            if device in ['自动选择', 'auto', '']:
                import torch
                # 更健壮的CUDA检测
                try:
                    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                        # 测试CUDA是否真正可用
                        torch.cuda.current_device()
                        device = '0'
                        self.log_message.emit("✓ 检测到可用GPU，使用CUDA:0")
                    else:
                        device = 'cpu'
                        self.log_message.emit("✗ 未检测到可用GPU，使用CPU")
                except Exception as cuda_e:
                    device = 'cpu'
                    self.log_message.emit(f"⚠ CUDA检测失败，使用CPU: {cuda_e}")
            elif device in ['cpu', 'CPU']:
                device = 'cpu'
                self.log_message.emit("✓ 使用CPU进行推理")
            elif device.startswith('cuda:') or device.startswith('CUDA:'):
                device = device.split(':')[1]
            elif device.isdigit():
                # 纯数字，认为是GPU ID
                device = device
                self.log_message.emit(f"✓ 使用CUDA:{device}")
            else:
                # 默认使用CPU
                device = 'cpu'
                self.log_message.emit(f"⚠ 未知设备设置 '{device}'，默认使用CPU")
            
            self.log_message.emit(f"推理参数: conf={conf}, iou={iou}, imgsz={imgsz}, device={device}")
            
            total = len(self.data_paths)
            success_count = 0
            
            for i, data_path in enumerate(self.data_paths):
                if not self._is_running:
                    break
                
                self.progress_updated.emit(i + 1, total)
                self.log_message.emit(f"[{i+1}/{total}] 处理: {os.path.basename(data_path)}")
                
                try:
                    # 检查是否为视频文件
                    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
                    is_video = any(data_path.lower().endswith(ext) for ext in video_extensions)
                    
                    if is_video:
                        # 视频推理
                        self.process_video(data_path, conf, iou, imgsz, device)
                    else:
                        # 图片推理
                        results = self.model(
                            data_path,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            device=device,
                            verbose=False
                        )
                        
                        # 处理结果
                        result_data = self.process_result(results[0], data_path)
                        self.result_ready.emit(result_data)
                    
                    success_count += 1
                    
                except Exception as e:
                    self.log_message.emit(f"✗ 处理失败 {data_path}: {e}")
            
            if self._is_running:
                self.inference_finished.emit(True, f"推理完成！成功: {success_count}/{total}")
            else:
                self.inference_finished.emit(False, "推理已停止")
                
        except ImportError as e:
            self.log_message.emit(f"✗ 未检测到Ultralytics库: {e}")
            self.inference_finished.emit(False, "未安装Ultralytics库")
        except Exception as e:
            self.log_message.emit(f"✗ 推理出错: {e}")
            import traceback
            self.log_message.emit(traceback.format_exc())
            self.inference_finished.emit(False, f"推理出错: {e}")
    
    def process_video(self, video_path: str, conf: float, iou: float, imgsz: int, device: str):
        """处理视频文件"""
        self.log_message.emit(f"开始处理视频: {os.path.basename(video_path)}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log_message.emit(f"✗ 无法打开视频: {video_path}")
            return
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0:
            fps = 25.0
        
        self.log_message.emit(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")

        # 创建输出视频（边推理边写盘，避免帧缓存导致内存暴涨）
        app_root = Path(__file__).parent.parent.parent
        output_video_path = self.video_output_dir / f"{Path(video_path).stem}_annotated.mp4"
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            self.log_message.emit(f"✗ 无法创建输出视频: {output_video_path}")
            return
        
        frame_count = 0
        processed_count = 0
        
        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每几帧处理一次（可以根据需要调整）
            if frame_count % 1 == 0:  # 处理每一帧
                try:
                    # 对单帧进行推理
                    results = self.model(
                        frame,
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        device=device,
                        verbose=False
                    )
                    
                    # 获取标注后的帧
                    annotated_frame = results[0].plot()
                    # 统一输出帧尺寸，避免播放器首段出现“逐步放大/缩小”的观感
                    if annotated_frame is not None:
                        ah, aw = annotated_frame.shape[:2]
                        if aw != width or ah != height:
                            annotated_frame = cv2.resize(
                                annotated_frame,
                                (width, height),
                                interpolation=cv2.INTER_LINEAR
                            )
                    if annotated_frame is not None:
                        writer.write(annotated_frame)
                    
                    # 处理检测结果
                    detections = []
                    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                        for box in results[0].boxes:
                            class_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                            # 应用类别映射
                            mapped_class_id = self.class_mapping.get(class_id, class_id)
                            
                            detection = {
                                'class_id': mapped_class_id,
                                'class_name': results[0].names.get(class_id, 'unknown'),
                                'confidence': float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf),
                                'bbox': box.xyxy[0].tolist() if hasattr(box.xyxy, 'tolist') else list(box.xyxy[0])
                            }
                            detections.append(detection)
                    
                    processed_count += 1
                    
                except Exception as e:
                    self.log_message.emit(f"✗ 处理帧 {frame_count} 失败: {e}")
            
            frame_count += 1
            
            # 每30帧更新一次进度
            if frame_count % 30 == 0:
                self.log_message.emit(f"  已处理 {frame_count}/{total_frames} 帧")
        
        cap.release()
        writer.release()
        
        # 发送视频处理完成信号
        result_data = {
            'source_path': video_path,
            'filename': os.path.basename(video_path),
            'is_video': True,
            'output_video': str(output_video_path),
            'total_frames': total_frames,
            'processed_frames': processed_count,
            'fps': fps,
            'detections': []  # 视频不存储所有检测结果
        }
        self.result_ready.emit(result_data)
        
        self.log_message.emit(f"✓ 视频处理完成: {processed_count} 帧")
    
    def process_result(self, result, source_path: str) -> dict:
        """处理单张推理结果"""
        result_data = {
            'source_path': source_path,
            'filename': os.path.basename(source_path),
            'detections': [],
            'masks': [],  # 添加masks字段用于分割模型
            'speed': {},
            'annotated_image_path': None
        }
        
        # 获取检测框
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                # 应用类别映射
                mapped_class_id = self.class_mapping.get(class_id, class_id)
                
                detection = {
                    'class_id': mapped_class_id,
                    'class_name': result.names.get(class_id, 'unknown'),
                    'confidence': float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf),
                    'bbox': box.xyxy[0].tolist() if hasattr(box.xyxy, 'tolist') else list(box.xyxy[0])
                }
                result_data['detections'].append(detection)
        
        # 获取分割masks（分割模型）
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks
            if hasattr(masks, 'xy') and masks.xy is not None:
                for i, mask_xy in enumerate(masks.xy):
                    # mask_xy 是多边形点列表 [(x1,y1), (x2,y2), ...]
                    if len(mask_xy) > 0:
                        # 获取对应的类别信息
                        class_id = 0
                        class_name = 'unknown'
                        confidence = 0.0
                        if hasattr(result, 'boxes') and result.boxes is not None and i < len(result.boxes):
                            box = result.boxes[i]
                            class_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                            class_name = result.names.get(class_id, 'unknown')
                            confidence = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                        
                        mask_data = {
                            'class_id': self.class_mapping.get(class_id, class_id),
                            'class_name': class_name,
                            'confidence': confidence,
                            'points': mask_xy.tolist() if hasattr(mask_xy, 'tolist') else list(mask_xy)
                        }
                        result_data['masks'].append(mask_data)
        
        # 获取推理速度
        if hasattr(result, 'speed'):
            result_data['speed'] = result.speed
        
        # 获取标注后的图像（Ultralytics会自动处理分割可视化）
        if hasattr(result, 'plot'):
            annotated_img = result.plot()
            if annotated_img is not None:
                output_path = self.image_output_dir / f"{Path(source_path).stem}_annotated.jpg"
                cv2.imwrite(str(output_path), annotated_img)
                result_data['annotated_image_path'] = str(output_path)
        
        return result_data
    
    def stop(self):
        """停止推理"""
        self._is_running = False


class ImageViewer(QWidget):
    """图像查看器组件"""
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_result = None
        self.scale_factor = 1.0
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(8)
        
        # 图像标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['sidebar']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
        """)
        self.image_label.setMinimumSize(400, 300)
        self.layout.addWidget(self.image_label, 0, Qt.AlignmentFlag.AlignCenter)
        
        # 信息标签
        self.info_label = QLabel("未加载图像")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        self.layout.addWidget(self.info_label)
        self._apply_aspect_ratio()
    
    def show_image(self, image_path: str):
        """显示图像"""
        if not os.path.exists(image_path):
            self.info_label.setText(f"图像不存在: {image_path}")
            return
        
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.info_label.setText(f"无法加载图像: {image_path}")
            return
        
        self.current_image = pixmap
        self._update_display()
        self.info_label.setText(f"{os.path.basename(image_path)} ({pixmap.width()}x{pixmap.height()})")
    
    def show_array(self, image_array: np.ndarray):
        """显示numpy数组图像"""
        if image_array is None:
            return
        
        # 转换BGR到RGB
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image_array
        
        # 转换为QPixmap
        height, width = rgb_image.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        self.current_image = pixmap
        self._update_display()
        self.info_label.setText(f"推理结果 ({width}x{height})")
    
    def _update_display(self):
        """更新显示"""
        if self.current_image is None:
            return
        
        # 缩放以适应窗口
        scaled_pixmap = self.current_image.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
    
    def resizeEvent(self, event):
        """窗口大小改变时重新缩放图像"""
        super().resizeEvent(event)
        self._apply_aspect_ratio()
        self._update_display()

    def _apply_aspect_ratio(self):
        """将图像显示区域固定为4:3比例。"""
        available_w = max(1, self.width())
        available_h = max(1, self.height() - self.info_label.sizeHint().height() - 8)
        target_w = min(available_w, int(available_h * 4 / 3))
        target_h = int(target_w * 3 / 4)
        if target_h > available_h:
            target_h = available_h
            target_w = int(target_h * 4 / 3)
        self.image_label.setFixedSize(max(1, target_w), max(1, target_h))


class VideoPlayer(QWidget):
    """视频播放器组件"""
    
    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.is_playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.video_frames = []  # 兼容旧逻辑，默认不再存完整视频帧
        self.current_frame_index = 0
        self.video_path = None
        self.video_cap = None
        self.video_fps = 30.0
        self.total_frames = 0
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(8)
        
        # 视频显示标签
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['sidebar']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
        """)
        self.video_label.setMinimumSize(400, 300)
        self.layout.addWidget(self.video_label, 0, Qt.AlignmentFlag.AlignCenter)
        
        # 控制按钮
        self.controls_widget = QWidget()
        controls_layout = QHBoxLayout(self.controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_play_pause = QPushButton("▶ 播放")
        self.btn_play_pause.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.btn_play_pause)
        
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.clicked.connect(self.stop)
        controls_layout.addWidget(self.btn_stop)
        
        controls_layout.addStretch()
        
        # 帧计数
        self.frame_label = QLabel("0 / 0")
        controls_layout.addWidget(self.frame_label)
        
        self.layout.addWidget(self.controls_widget)
        self._apply_aspect_ratio()
    
    def add_frame(self, frame: np.ndarray):
        """添加视频帧"""
        # 保留兼容能力（调试场景），正式流程不再依赖帧缓存
        self.video_frames.append(frame.copy())
        self.frame_label.setText(f"{self.current_frame_index + 1} / {len(self.video_frames)}")
        if len(self.video_frames) == 1 and not self.is_playing:
            self.display_frame(self.video_frames[0])

    def load_video(self, video_path: str):
        """加载视频文件进行播放（无帧缓存模式）。"""
        self.clear()
        self.video_path = video_path
        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            self.video_cap = None
            self.frame_label.setText("0 / 0")
            return

        self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps <= 0:
            self.video_fps = 30.0
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0
        self._seek_and_show_first_frame()
    
    def toggle_play(self):
        """切换播放/暂停"""
        if self.is_playing:
            self.pause()
        else:
            self.play()
    
    def play(self):
        """开始播放"""
        if self.video_cap is not None:
            self.is_playing = True
            self.btn_play_pause.setText("⏸ 暂停")
            interval = max(1, int(1000 / self.video_fps))
            self.timer.start(interval)
            return

        if not self.video_frames:
            return
        self.is_playing = True
        self.btn_play_pause.setText("⏸ 暂停")
        self.timer.start(33)  # 约30fps
    
    def pause(self):
        """暂停播放"""
        self.is_playing = False
        self.btn_play_pause.setText("▶ 播放")
        self.timer.stop()
    
    def stop(self):
        """停止播放"""
        self.is_playing = False
        self.btn_play_pause.setText("▶ 播放")
        self.timer.stop()
        if self.video_cap is not None:
            self.current_frame_index = 0
            self._seek_and_show_first_frame()
            return
        self.current_frame_index = 0
        if self.video_frames:
            self.display_frame(self.video_frames[0])
    
    def update_frame(self):
        """更新帧"""
        if self.video_cap is not None:
            ret, frame = self.video_cap.read()
            if not ret:
                self.stop()
                return
            self.display_frame(frame)
            self.current_frame_index += 1
            self.frame_label.setText(f"{self.current_frame_index} / {self.total_frames}")
            return

        if not self.video_frames:
            return
        
        if self.current_frame_index < len(self.video_frames):
            self.display_frame(self.video_frames[self.current_frame_index])
            self.current_frame_index += 1
            self.frame_label.setText(f"{self.current_frame_index} / {len(self.video_frames)}")
        else:
            # 播放结束
            self.stop()
    
    def display_frame(self, frame: np.ndarray):
        """显示单帧"""
        if frame is None:
            return
        self.current_frame = frame
        
        # 转换BGR到RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
        
        # 转换为QPixmap
        height, width = rgb_frame.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # 缩放以适应窗口
        target_size = self.video_label.contentsRect().size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            target_size = self.video_label.size()
        scaled_pixmap = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    def clear(self):
        """清空视频帧"""
        self.timer.stop()
        self.is_playing = False
        self.btn_play_pause.setText("▶ 播放")
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.video_path = None
        self.video_fps = 30.0
        self.total_frames = 0
        self.video_frames.clear()
        self.current_frame_index = 0
        self.current_frame = None
        self.video_label.clear()
        self.frame_label.setText("0 / 0")

    def _seek_and_show_first_frame(self):
        """跳转到第一帧并显示。"""
        if self.video_cap is None:
            return
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.video_cap.read()
        if ret:
            self.display_frame(frame)
            self.current_frame_index = 1
            self.frame_label.setText(f"{self.current_frame_index} / {self.total_frames}")
            # 回到起点，保证播放从第一帧开始
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_index = 0
    
    def resizeEvent(self, event):
        """窗口大小改变时重新缩放"""
        super().resizeEvent(event)
        self._apply_aspect_ratio()
        if self.current_frame is not None:
            self.display_frame(self.current_frame)

    def _apply_aspect_ratio(self):
        """将视频显示区域固定为4:3比例。"""
        controls_h = self.controls_widget.sizeHint().height() if hasattr(self, "controls_widget") else 0
        available_w = max(1, self.width())
        available_h = max(1, self.height() - controls_h - 8)
        target_w = min(available_w, int(available_h * 4 / 3))
        target_h = int(target_w * 3 / 4)
        if target_h > available_h:
            target_h = available_h
            target_w = int(target_h * 4 / 3)
        self.video_label.setFixedSize(max(1, target_w), max(1, target_h))


class TestPage(QWidget):
    """测试页面"""
    
    def __init__(self):
        super().__init__()
        self.current_project_id = None
        self.current_project = None
        self.inference_thread = None
        self.model_path = None
        self.data_paths = []  # 待推理的数据路径列表
        self.current_data_index = 0
        self.inference_results = []  # 推理结果列表
        self.video_frames = []  # 视频帧缓存
        self.class_mapping = {}  # 类别映射
        self.model_classes = []  # 模型类别列表
        self.output_root = Path(__file__).parent.parent.parent / "outputs"
        self._clear_test_outputs_on_startup()
        
        self.init_ui()

    def _clear_test_outputs_on_startup(self):
        """页面启动时清空测试输出文件。"""
        for folder in ("test_images", "test_videos"):
            path = self.output_root / folder
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
            path.mkdir(parents=True, exist_ok=True)
    
    def set_project(self, project_id: int):
        """设置当前项目"""
        self.current_project_id = project_id
        if project_id:
            self.current_project = db.get_project(project_id)
            if self.current_project:
                project_name = self.current_project.get('name', 'Unknown') if isinstance(self.current_project, dict) else self.current_project.name
                print(f"[TestPage] 已切换到项目: {project_name} (ID: {project_id})")
                # 自动设置默认模型路径
                self.set_default_model_path()
        else:
            self.current_project = None
            print("[TestPage] 项目已取消选择")
    
    def set_default_model_path(self):
        """设置默认模型路径（项目训练文件夹中的best.pt）"""
        if not self.current_project_id:
            return
        
        # 构建默认模型路径
        app_root = Path(__file__).parent.parent.parent
        default_path = app_root / "runs" / "train" / f"exp_{self.current_project_id}" / "weights" / "best.pt"
        
        if default_path.exists():
            self.model_path = str(default_path)
            self.model_path_label.setText(f"模型: {self.model_path}")
            self.log_message(f"自动加载模型: {self.model_path}")
        else:
            self.model_path = None
            self.model_path_label.setText("模型: 未选择 (将使用默认模型)")
    
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
        
        # 右侧：可视化面板
        right_panel = self.create_visualization_panel()
        splitter.addWidget(right_panel)
        
        # 设置分割比例
        splitter.setSizes([400, 800])
        
        main_layout.addWidget(splitter)
    
    def create_config_panel(self) -> QWidget:
        """创建配置面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        
        # 标题
        title = QLabel("⑤ 测试")
        title.setObjectName("title")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        
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
        
        # 模型路径组
        model_path_group = self.create_model_path_group()
        scroll_layout.addWidget(model_path_group)
        
        # 类别映射组
        class_mapping_group = self.create_class_mapping_group()
        scroll_layout.addWidget(class_mapping_group)
        
        # 推理参数组
        params_group = self.create_params_group()
        scroll_layout.addWidget(params_group)
        
        # 数据加载组
        data_group = self.create_data_group()
        scroll_layout.addWidget(data_group)
        
        # 控制按钮组
        control_group = self.create_control_group()
        scroll_layout.addWidget(control_group)
        
        # 日志组
        log_group = self.create_log_group()
        scroll_layout.addWidget(log_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        return panel
    
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
        self.model_version = QComboBox()
        self.model_version.addItems(sorted(ULTRALYTICS_MODELS.keys()))
        self.model_version.currentTextChanged.connect(self.on_version_changed)
        layout.addRow("版本:", self.model_version)
        
        # 模型型号选择
        self.model_size = QComboBox()
        layout.addRow("型号:", self.model_size)
        
        # 任务类型
        self.task_type = QComboBox()
        layout.addRow("任务:", self.task_type)
        
        # 立即初始化型号和任务列表
        self._init_model_lists()
        
        return group
    
    def _init_model_lists(self):
        """初始化型号和任务列表"""
        version = self.model_version.currentText()
        
        if not version:
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
    
    def create_model_path_group(self) -> QGroupBox:
        """创建模型路径组"""
        group = QGroupBox("模型路径")
        group.setStyleSheet(self.get_group_style())
        
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        # 模型路径标签
        self.model_path_label = QLabel("模型: 未选择")
        self.model_path_label.setWordWrap(True)
        self.model_path_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        layout.addWidget(self.model_path_label)
        
        # 按钮布局
        btn_layout = QHBoxLayout()
        
        # 选择模型按钮
        self.btn_select_model = QPushButton("📁 选择模型")
        self.btn_select_model.clicked.connect(self.select_model)
        btn_layout.addWidget(self.btn_select_model)
        
        # 使用默认模型按钮
        self.btn_default_model = QPushButton("🔄 使用默认")
        self.btn_default_model.clicked.connect(self.set_default_model_path)
        btn_layout.addWidget(self.btn_default_model)
        
        layout.addLayout(btn_layout)
        
        return group
    
    def create_class_mapping_group(self) -> QGroupBox:
        """创建类别映射组"""
        group = QGroupBox("类别映射")
        group.setStyleSheet(self.get_group_style())
        
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        # 启用映射选项
        enable_mapping_layout = QHBoxLayout()
        self.chk_enable_mapping = QCheckBox("启用类别映射")
        self.chk_enable_mapping.setChecked(False)
        self.chk_enable_mapping.stateChanged.connect(self.on_enable_mapping_changed)
        enable_mapping_layout.addWidget(self.chk_enable_mapping)
        enable_mapping_layout.addStretch()
        layout.addLayout(enable_mapping_layout)
        
        # 模型类别文件加载
        model_class_file_layout = QHBoxLayout()
        self.btn_load_classes = QPushButton("📄 加载模型classes.txt")
        self.btn_load_classes.clicked.connect(self.load_model_classes)
        self.btn_load_classes.setEnabled(False)
        model_class_file_layout.addWidget(self.btn_load_classes)
        
        # 显示已加载的类别数
        self.model_classes_label = QLabel("未加载")
        self.model_classes_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        model_class_file_layout.addWidget(self.model_classes_label)
        model_class_file_layout.addStretch()
        layout.addLayout(model_class_file_layout)
        
        # 类别映射显示
        self.class_mapping_list = QListWidget()
        self.class_mapping_list.setMaximumHeight(100)
        self.class_mapping_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {COLORS['sidebar']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
        """)
        layout.addWidget(self.class_mapping_list)
        
        # 编辑映射按钮
        self.btn_edit_mapping = QPushButton("✏️ 编辑类别映射")
        self.btn_edit_mapping.clicked.connect(self.edit_class_mapping)
        self.btn_edit_mapping.setEnabled(False)
        layout.addWidget(self.btn_edit_mapping)
        
        return group
    
    def on_enable_mapping_changed(self, state):
        """启用映射选项改变"""
        enabled = state == Qt.CheckState.Checked.value
        self.btn_load_classes.setEnabled(enabled)
        self.btn_edit_mapping.setEnabled(enabled and len(self.model_classes) > 0)
    
    def load_model_classes(self):
        """加载模型classes.txt文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型classes.txt文件", 
            "", "Text files (*.txt)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    classes = [line.strip() for line in f if line.strip()]
                if classes:
                    self.model_classes = classes
                    self.model_classes_label.setText(f"已加载 {len(classes)} 个类别")
                    
                    # 自动创建默认映射（一对一）
                    self.class_mapping = {i: i for i in range(len(classes))}
                    self.update_class_mapping_list()
                    
                    self.btn_edit_mapping.setEnabled(True)
                    self.log_message(f"✓ 成功加载类别文件: {file_path}")
                    self.log_message(f"  类别: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}")
                else:
                    QMessageBox.warning(self, "警告", "classes.txt文件为空")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载classes.txt文件失败: {str(e)}")
    
    def update_class_mapping_list(self):
        """更新类别映射列表显示"""
        self.class_mapping_list.clear()
        
        if not self.model_classes:
            return
        
        # 获取项目类别
        project_classes = []
        if self.current_project:
            classes_json = self.current_project.get('classes', '[]') if isinstance(self.current_project, dict) else '[]'
            project_classes = json.loads(classes_json) if classes_json else []
        
        project_class_names = {cls['id']: cls['name'] for cls in project_classes}
        
        for model_id, project_id in self.class_mapping.items():
            if model_id < len(self.model_classes):
                model_name = self.model_classes[model_id]
                project_name = project_class_names.get(project_id, f"ID:{project_id}")
                item_text = f"{model_id}: {model_name} → {project_id}: {project_name}"
                self.class_mapping_list.addItem(item_text)
    
    def edit_class_mapping(self):
        """编辑类别映射"""
        if not self.model_classes:
            QMessageBox.warning(self, "警告", "请先加载模型classes.txt文件")
            return
        
        # 获取项目类别
        project_classes = []
        if self.current_project:
            classes_json = self.current_project.get('classes', '[]') if isinstance(self.current_project, dict) else '[]'
            project_classes = json.loads(classes_json) if classes_json else []
        
        if not project_classes:
            QMessageBox.warning(self, "警告", "当前项目没有类别，请先设置项目类别")
            return
        
        # 创建对话框
        from PyQt6.QtWidgets import QDialog, QComboBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("编辑类别映射")
        dialog.setMinimumSize(400, 300)
        
        layout = QVBoxLayout(dialog)
        
        # 说明标签
        info_label = QLabel("将模型类别映射到项目类别:")
        layout.addWidget(info_label)
        
        # 创建映射选择
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        mapping_widgets = []
        
        for i, model_class in enumerate(self.model_classes):
            row_layout = QHBoxLayout()
            
            # 模型类别标签
            model_label = QLabel(f"{i}: {model_class}")
            model_label.setMinimumWidth(150)
            row_layout.addWidget(model_label)
            
            row_layout.addWidget(QLabel("→"))
            
            # 项目类别选择
            project_combo = QComboBox()
            for cls in project_classes:
                project_combo.addItem(f"{cls['id']}: {cls['name']}", cls['id'])
            
            # 设置当前映射
            current_mapping = self.class_mapping.get(i, i)
            index = project_combo.findData(current_mapping)
            if index >= 0:
                project_combo.setCurrentIndex(index)
            
            row_layout.addWidget(project_combo)
            mapping_widgets.append((i, project_combo))
            
            scroll_layout.addLayout(row_layout)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        # 按钮
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("确定")
        btn_ok.clicked.connect(dialog.accept)
        btn_layout.addWidget(btn_ok)
        
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(dialog.reject)
        btn_layout.addWidget(btn_cancel)
        
        layout.addLayout(btn_layout)
        
        # 显示对话框
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # 更新映射
            for model_id, combo in mapping_widgets:
                project_id = combo.currentData()
                self.class_mapping[model_id] = project_id
            
            self.update_class_mapping_list()
            self.log_message(f"✓ 类别映射已更新: {self.class_mapping}")
    
    def create_params_group(self) -> QGroupBox:
        """创建推理参数组"""
        group = QGroupBox("推理参数")
        group.setStyleSheet(self.get_group_style())
        
        layout = QFormLayout(group)
        layout.setSpacing(10)
        
        # 置信度阈值
        self.conf_threshold = QDoubleSpinBox()
        self.conf_threshold.setRange(0.01, 1.0)
        self.conf_threshold.setValue(0.25)
        self.conf_threshold.setDecimals(2)
        self.conf_threshold.setSingleStep(0.05)
        layout.addRow("置信度阈值:", self.conf_threshold)
        
        # NMS IoU阈值
        self.iou_threshold = QDoubleSpinBox()
        self.iou_threshold.setRange(0.1, 1.0)
        self.iou_threshold.setValue(0.45)
        self.iou_threshold.setDecimals(2)
        self.iou_threshold.setSingleStep(0.05)
        layout.addRow("NMS IoU阈值:", self.iou_threshold)
        
        # 图像尺寸
        self.inference_size = QSpinBox()
        self.inference_size.setRange(320, 1280)
        self.inference_size.setValue(640)
        self.inference_size.setSingleStep(32)
        layout.addRow("推理尺寸:", self.inference_size)
        
        # 设备选择
        self.inference_device = QComboBox()
        self.inference_device.addItems(["自动选择", "CPU", "CUDA:0", "CUDA:1", "CUDA:2", "CUDA:3"])
        layout.addRow("设备:", self.inference_device)
        
        # 保存结果选项
        self.save_results = QCheckBox("保存推理结果")
        self.save_results.setChecked(True)
        layout.addRow(self.save_results)
        
        # 显示标签选项
        self.show_labels = QCheckBox("显示标签")
        self.show_labels.setChecked(True)
        layout.addRow(self.show_labels)
        
        return group
    
    def create_data_group(self) -> QGroupBox:
        """创建数据加载组"""
        group = QGroupBox("数据加载")
        group.setStyleSheet(self.get_group_style())
        
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        # 数据列表
        self.data_list = QListWidget()
        self.data_list.setMaximumHeight(150)
        self.data_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {COLORS['sidebar']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
            QListWidget::item {{
                padding: 4px 8px;
                border-bottom: 1px solid {COLORS['border']};
            }}
        """)
        layout.addWidget(self.data_list)
        
        # 数据数量标签
        self.data_count_label = QLabel("已加载: 0 个文件")
        self.data_count_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        layout.addWidget(self.data_count_label)
        
        # 按钮布局
        btn_layout = QHBoxLayout()
        
        # 加载图片按钮
        self.btn_load_image = QPushButton("🖼️ 图片")
        self.btn_load_image.clicked.connect(lambda: self.load_data("image"))
        btn_layout.addWidget(self.btn_load_image)
        
        # 加载文件夹按钮
        self.btn_load_folder = QPushButton("📁 文件夹")
        self.btn_load_folder.clicked.connect(lambda: self.load_data("folder"))
        btn_layout.addWidget(self.btn_load_folder)
        
        # 加载视频按钮
        self.btn_load_video = QPushButton("🎬 视频")
        self.btn_load_video.clicked.connect(lambda: self.load_data("video"))
        btn_layout.addWidget(self.btn_load_video)
        
        # 清空按钮
        self.btn_clear_data = QPushButton("🗑️ 清空")
        self.btn_clear_data.clicked.connect(self.clear_data)
        btn_layout.addWidget(self.btn_clear_data)
        
        layout.addLayout(btn_layout)
        
        return group
    
    def create_control_group(self) -> QGroupBox:
        """创建控制按钮组"""
        group = QGroupBox("推理控制")
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
        
        # 开始推理按钮
        self.btn_start_inference = QPushButton("▶ 开始推理")
        self.btn_start_inference.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                font-weight: bold;
                padding: 12px 30px;
                font-size: 14px;
            }}
        """)
        self.btn_start_inference.clicked.connect(self.start_inference)
        btn_layout.addWidget(self.btn_start_inference)
        
        # 停止按钮
        self.btn_stop_inference = QPushButton("⏹ 停止")
        self.btn_stop_inference.setEnabled(False)
        self.btn_stop_inference.clicked.connect(self.stop_inference)
        btn_layout.addWidget(self.btn_stop_inference)
        
        layout.addLayout(btn_layout)
        
        return group
    
    def create_log_group(self) -> QGroupBox:
        """创建日志组"""
        group = QGroupBox("日志")
        group.setStyleSheet(self.get_group_style())
        
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        # 日志文本框
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
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
        
        return group
    
    def create_visualization_panel(self) -> QWidget:
        """创建可视化面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        
        # 标题
        title = QLabel("推理结果")
        title.setObjectName("title")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        
        # 创建标签页
        self.result_tabs = QTabWidget()
        self.result_tabs.setStyleSheet(f"""
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
        
        # 图像结果标签页
        self.image_tab = QWidget()
        image_layout = QVBoxLayout(self.image_tab)
        self.image_viewer = ImageViewer()
        image_layout.addWidget(self.image_viewer)
        self.result_tabs.addTab(self.image_tab, "🖼️ 图像")
        
        # 视频结果标签页
        self.video_tab = QWidget()
        video_layout = QVBoxLayout(self.video_tab)
        self.video_player = VideoPlayer()
        video_layout.addWidget(self.video_player)
        self.result_tabs.addTab(self.video_tab, "🎬 视频")
        
        layout.addWidget(self.result_tabs)
        
        # 结果信息面板
        self.result_info = QLabel("等待推理...")
        self.result_info.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['panel']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 10px;
                font-size: 12px;
            }}
        """)
        self.result_info.setWordWrap(True)
        self.result_info.setMinimumHeight(100)
        layout.addWidget(self.result_info)
        
        # 导航和导出按钮
        nav_layout = QHBoxLayout()
        
        # 上一个按钮
        self.btn_prev = QPushButton("◀ 上一个")
        self.btn_prev.clicked.connect(self.show_previous_result)
        self.btn_prev.setEnabled(False)
        nav_layout.addWidget(self.btn_prev)
        
        # 结果计数
        self.result_count_label = QLabel("0 / 0")
        self.result_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self.result_count_label)
        
        # 下一个按钮
        self.btn_next = QPushButton("下一个 ▶")
        self.btn_next.clicked.connect(self.show_next_result)
        self.btn_next.setEnabled(False)
        nav_layout.addWidget(self.btn_next)
        
        nav_layout.addStretch()
        
        # 导出按钮
        self.btn_export = QPushButton("💾 导出结果")
        self.btn_export.clicked.connect(self.export_results)
        self.btn_export.setEnabled(False)
        nav_layout.addWidget(self.btn_export)
        
        layout.addLayout(nav_layout)
        
        return panel
    
    def select_model(self):
        """选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "",
            "所有模型文件 (*.pt *.pth *.onnx *.engine *.trt);;"
            "PyTorch模型 (*.pt *.pth);;"
            "ONNX模型 (*.onnx);;"
            "TensorRT模型 (*.engine *.trt);;"
            "所有文件 (*.*)"
        )
        
        if file_path:
            self.model_path = file_path
            self.model_path_label.setText(f"模型: {file_path}")
            self.log_message(f"已选择模型: {file_path}")
            
            # 根据模型类型给出提示
            if file_path.endswith('.onnx'):
                self.log_message("ℹ 已选择ONNX模型，确保已安装onnxruntime或onnxruntime-gpu")
            elif file_path.endswith('.engine') or file_path.endswith('.trt'):
                self.log_message("ℹ 已选择TensorRT模型，确保已安装tensorrt并配置正确")
    
    def load_data(self, data_type: str):
        """加载数据"""
        if data_type == "image":
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "选择图片", "",
                "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;所有文件 (*.*)"
            )
            if file_paths:
                self.data_paths.extend(file_paths)
                self.log_message(f"已加载 {len(file_paths)} 张图片")
        
        elif data_type == "folder":
            folder_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
            if folder_path:
                # 获取文件夹中的所有图片
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(Path(folder_path).glob(f"*{ext}"))
                    image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
                
                file_paths = [str(f) for f in image_files]
                self.data_paths.extend(file_paths)
                self.log_message(f"已从文件夹加载 {len(file_paths)} 张图片")
        
        elif data_type == "video":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择视频", "",
                "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*.*)"
            )
            if file_path:
                self.data_paths.append(file_path)
                self.log_message(f"已加载视频: {file_path}")
        
        # 更新数据列表显示
        self.update_data_list()
    
    def update_data_list(self):
        """更新数据列表显示"""
        self.data_list.clear()
        for path in self.data_paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setToolTip(path)
            self.data_list.addItem(item)
        
        self.data_count_label.setText(f"已加载: {len(self.data_paths)} 个文件")
    
    def clear_data(self):
        """清空数据"""
        self.data_paths.clear()
        self.data_list.clear()
        self.data_count_label.setText("已加载: 0 个文件")
        self.log_message("已清空数据列表")
    
    def start_inference(self):
        """开始推理"""
        # 检查是否有数据
        if not self.data_paths:
            QMessageBox.warning(self, "错误", "请先加载数据")
            return
        
        # 清空之前的结果
        self.inference_results.clear()
        self.current_data_index = 0
        self.video_player.clear()
        
        # 获取项目类别信息
        project_classes = []
        if self.current_project:
            classes_json = self.current_project.get('classes', '[]') if isinstance(self.current_project, dict) else '[]'
            project_classes = json.loads(classes_json) if classes_json else []
        
        # 获取任务类型
        task = self.task_type.currentData()
        
        # 收集配置
        config = {
            'conf': self.conf_threshold.value(),
            'iou': self.iou_threshold.value(),
            'imgsz': self.inference_size.value(),
            'device': self.inference_device.currentText(),
            'task': task,  # 添加任务类型到配置
        }
        
        # 确定模型路径
        model_path = self.model_path
        if not model_path:
            # 使用预训练模型
            version = self.model_version.currentText()
            model_size = self.model_size.currentData()
            
            if version and model_size:
                model_prefix = ULTRALYTICS_MODELS[version]['prefix']
                if task == 'detect':
                    model_name = f"{model_prefix}{model_size}.pt"
                else:
                    task_suffix_map = {"segment": "seg", "classify": "cls", "pose": "pose", "world": "world"}
                    suffix = task_suffix_map.get(task, task)
                    model_name = f"{model_prefix}{model_size}-{suffix}.pt"
                model_path = model_name
                self.log_message(f"使用预训练模型: {model_path}")
        
        # 获取类别映射
        class_mapping = {}
        if self.chk_enable_mapping.isChecked():
            class_mapping = self.class_mapping
        
        # 创建推理线程
        self.inference_thread = InferenceThread(
            model_path=model_path,
            config=config,
            data_paths=self.data_paths.copy(),
            project_classes=project_classes,
            class_mapping=class_mapping
        )
        self.inference_thread.progress_updated.connect(self.on_progress_updated)
        self.inference_thread.inference_finished.connect(self.on_inference_finished)
        self.inference_thread.log_message.connect(self.on_log_message)
        self.inference_thread.result_ready.connect(self.on_result_ready)
        self.inference_thread.frame_ready.connect(self.on_frame_ready)
        
        # 更新UI状态
        self.btn_start_inference.setEnabled(False)
        self.btn_stop_inference.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.data_paths))
        self.progress_bar.setValue(0)
        self.status_label.setText("推理中...")
        self.status_label.setStyleSheet(f"color: {COLORS['primary']}; font-size: 12px;")
        
        # 启动推理
        self.inference_thread.start()
        
        self.log_message("=" * 50)
        self.log_message("开始推理！")
        self.log_message(f"数据数量: {len(self.data_paths)}")
        if class_mapping:
            self.log_message(f"类别映射: 已启用")
        self.log_message("=" * 50)
    
    def stop_inference(self):
        """停止推理"""
        if self.inference_thread:
            self.inference_thread.stop()
            self.inference_thread.wait()
        
        self.reset_ui_state()
        self.log_message("推理已停止")
    
    def reset_ui_state(self):
        """重置UI状态"""
        self.btn_start_inference.setEnabled(True)
        self.btn_stop_inference.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("就绪")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
    
    def on_progress_updated(self, current: int, total: int):
        """进度更新"""
        self.progress_bar.setValue(current)
        self.status_label.setText(f"推理中... {current}/{total}")
    
    def on_inference_finished(self, success: bool, message: str):
        """推理完成"""
        self.reset_ui_state()
        
        if success:
            QMessageBox.information(self, "推理完成", message)
            # 显示第一个结果
            if self.inference_results:
                self.show_result(0)
        else:
            QMessageBox.warning(self, "推理结束", message)
        
        # 更新导航按钮状态
        self.update_nav_buttons()
    
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
    
    def on_result_ready(self, result: dict):
        """单张推理结果就绪"""
        self.inference_results.append(result)
        
        # 显示最新结果
        self.show_result(len(self.inference_results) - 1)
        
        # 更新导出按钮
        self.btn_export.setEnabled(True)
    
    def on_frame_ready(self, frame: np.ndarray, detection_info: dict):
        """视频帧就绪"""
        # 保留接口兼容；当前视频流程改为写盘后播放，不再使用内存帧缓存
        if detection_info.get('frame_number') == 0:
            self.result_tabs.setCurrentIndex(1)  # 切换到视频标签页
    
    def show_result(self, index: int):
        """显示指定索引的结果"""
        if not self.inference_results or index < 0 or index >= len(self.inference_results):
            return
        
        self.current_data_index = index
        result = self.inference_results[index]
        
        # 检查是否为视频
        if result.get('is_video'):
            # 切换到视频标签页
            self.result_tabs.setCurrentIndex(1)
            video_to_play = result.get('output_video') or result.get('source_path')
            if video_to_play and os.path.exists(video_to_play):
                self.video_player.load_video(video_to_play)
        else:
            # 切换到图像标签页
            self.result_tabs.setCurrentIndex(0)
            
            # 显示写盘后的标注图像
            image_to_show = result.get('annotated_image_path') or result.get('source_path')
            self.image_viewer.show_image(image_to_show)
        
        # 更新结果信息
        info_text = f"""
<b>文件:</b> {result['filename']}<br>
        """
        
        if result.get('is_video'):
            info_text += f"<b>类型:</b> 视频<br>"
            info_text += f"<b>总帧数:</b> {result.get('total_frames', 0)}<br>"
            info_text += f"<b>已处理:</b> {result.get('processed_frames', 0)} 帧<br>"
        else:
            # 显示检测框信息
            info_text += f"<b>检测到:</b> {len(result['detections'])} 个目标<br>"
            
            if result['detections']:
                info_text += "<b>检测详情:</b><br>"
                for i, det in enumerate(result['detections'][:10]):  # 最多显示10个
                    info_text += f"  {i+1}. {det['class_name']} ({det['confidence']:.2f})<br>"
                if len(result['detections']) > 10:
                    info_text += f"  ... 还有 {len(result['detections']) - 10} 个目标<br>"
            
            # 显示分割mask信息（如果有）
            if result.get('masks') and len(result['masks']) > 0:
                info_text += f"<br><b>分割区域:</b> {len(result['masks'])} 个<br>"
                info_text += "<b>分割详情:</b><br>"
                for i, mask in enumerate(result['masks'][:10]):  # 最多显示10个
                    point_count = len(mask.get('points', []))
                    info_text += f"  {i+1}. {mask['class_name']} ({mask['confidence']:.2f}) - {point_count} 个点<br>"
                if len(result['masks']) > 10:
                    info_text += f"  ... 还有 {len(result['masks']) - 10} 个分割区域<br>"
        
        if result.get('speed'):
            speed = result['speed']
            info_text += f"<br><b>推理速度:</b><br>"
            if 'preprocess' in speed:
                info_text += f"  预处理: {speed['preprocess']:.1f}ms<br>"
            if 'inference' in speed:
                info_text += f"  推理: {speed['inference']:.1f}ms<br>"
            if 'postprocess' in speed:
                info_text += f"  后处理: {speed['postprocess']:.1f}ms<br>"
        
        self.result_info.setText(info_text)
        
        # 更新计数
        self.result_count_label.setText(f"{index + 1} / {len(self.inference_results)}")
        
        # 更新导航按钮
        self.update_nav_buttons()
    
    def update_nav_buttons(self):
        """更新导航按钮状态"""
        total = len(self.inference_results)
        current = self.current_data_index
        
        self.btn_prev.setEnabled(current > 0)
        self.btn_next.setEnabled(current < total - 1)
    
    def show_previous_result(self):
        """显示上一个结果"""
        if self.current_data_index > 0:
            self.show_result(self.current_data_index - 1)
    
    def show_next_result(self):
        """显示下一个结果"""
        if self.current_data_index < len(self.inference_results) - 1:
            self.show_result(self.current_data_index + 1)
    
    def export_results(self):
        """导出推理结果"""
        if not self.inference_results:
            QMessageBox.warning(self, "错误", "没有可导出的结果")
            return
        
        # 选择导出目录
        export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not export_dir:
            return
        
        try:
            import shutil
            
            # 创建导出子目录
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_subdir = os.path.join(export_dir, f"inference_results_{timestamp}")
            os.makedirs(export_subdir, exist_ok=True)
            
            # 创建images和labels目录
            images_dir = os.path.join(export_subdir, "images")
            labels_dir = os.path.join(export_subdir, "labels")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            
            exported_count = 0
            
            for result in self.inference_results:
                filename = result['filename']
                name_without_ext = os.path.splitext(filename)[0]
                
                if result.get('is_video'):
                    # 导出视频
                    if result.get('annotated_image') is not None:
                        output_path = os.path.join(images_dir, filename)
                        # 这里应该保存视频文件，简化处理
                        self.log_message(f"视频导出暂不支持: {filename}")
                else:
                    # 保存标注后的图像
                    if result.get('annotated_image') is not None:
                        output_path = os.path.join(images_dir, filename)
                        cv2.imwrite(output_path, result['annotated_image'])
                    else:
                        # 复制原图
                        shutil.copy2(result['source_path'], os.path.join(images_dir, filename))
                    
                    # 保存标注文件（YOLO格式）
                    label_file = os.path.join(labels_dir, f"{name_without_ext}.txt")
                    with open(label_file, 'w') as f:
                        for det in result['detections']:
                            bbox = det['bbox']
                            # 转换为YOLO格式 (x_center, y_center, width, height)
                            x1, y1, x2, y2 = bbox
                            x_center = (x1 + x2) / 2
                            y_center = (y1 + y2) / 2
                            width = x2 - x1
                            height = y2 - y1
                            
                            # 这里假设图像是640x640，实际应该获取真实尺寸
                            # 简化处理，使用归一化坐标
                            f.write(f"{det['class_id']} {x_center} {y_center} {width} {height}\n")
                
                exported_count += 1
            
            # 保存推理日志
            log_file = os.path.join(export_subdir, "inference_log.txt")
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(self.log_text.toPlainText())
            
            QMessageBox.information(
                self, 
                "导出成功", 
                f"已导出 {exported_count} 个结果到:\n{export_subdir}"
            )
            self.log_message(f"✓ 导出完成: {export_subdir}")
            
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出时出错:\n{str(e)}")
            self.log_message(f"✗ 导出失败: {e}")
    
    def clear_log(self):
        """清空日志"""
        self.log_text.clear()
    
    def save_log(self):
        """保存日志"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存日志", "inference_log.txt",
            "文本文件 (*.txt);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                QMessageBox.information(self, "保存成功", f"日志已保存到:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存日志时出错:\n{str(e)}")
