# -*- coding: utf-8 -*-
"""
标注页面
支持矩形框、多边形标注，快捷键操作，自动保存
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QGridLayout, QFrame, QFileDialog, QProgressBar,
    QMenu, QMessageBox, QComboBox, QLineEdit, QSplitter,
    QListWidget, QListWidgetItem, QToolBar, QButtonGroup,
    QRadioButton, QSpinBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QCheckBox, QSlider, QTextEdit, QInputDialog,
    QColorDialog, QDialog
)

# 导入自动标注相关模块
from gui.pages.auto_label_dialog import AutoLabelDialog
from gui.pages.batch_process_dialog import BatchProcessDialog
from core.auto_labeler import BatchLabelingManager
from core.model_manager import ModelManager
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QSize, QPoint, QRect
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QKeyEvent, QMouseEvent, QWheelEvent, QAction, QIcon, QPen, QBrush
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os

from gui.styles import COLORS
from models.database import db
from gui.widgets.loading_dialog import LoadingOverlay


class AnnotateImageLoadWorker(QThread):
    """标注页面图片加载工作线程"""
    
    image_loaded = pyqtSignal(int, object)  # 索引, 缩略图
    finished_loading = pyqtSignal()
    
    def __init__(self, images: List[Dict]):
        super().__init__()
        self.images = images
        self._is_running = True
    
    def run(self):
        """在后台线程中加载图片"""
        for i, image_data in enumerate(self.images):
            if not self._is_running:
                break
            
            storage_path = image_data.get('storage_path', '')
            pixmap = None
            
            if storage_path and os.path.exists(storage_path):
                try:
                    img = cv2.imread(storage_path)
                    if img is not None:
                        img = cv2.resize(img, (80, 80))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        h, w, ch = img.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(qt_image)
                except Exception:
                    pass
            
            if pixmap is None or pixmap.isNull():
                pixmap = QPixmap(80, 80)
                pixmap.fill(QColor(COLORS['sidebar']))
            
            self.image_loaded.emit(i, pixmap)
            
            if i % 10 == 0:
                self.msleep(1)
        
        self.finished_loading.emit()
    
    def stop(self):
        self._is_running = False


class AnnotationCanvas(QFrame):
    """标注画布组件"""
    
    annotation_created = pyqtSignal(dict)  # 标注创建信号
    annotation_selected = pyqtSignal(int)  # 标注选中信号
    annotation_modified = pyqtSignal(int, dict)  # 标注修改信号
    annotation_deleted = pyqtSignal(int)  # 标注删除信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        
        # 图像数据
        self.current_image = None
        self.current_image_path = None
        self.image_scale = 1.0
        self.image_offset = QPoint(0, 0)
        
        # 标注数据
        self.annotations = []  # 当前图像的所有标注
        self.selected_annotation_id = None
        self.current_tool = 'rectangle'  # rectangle, polygon, move
        self.drawing = False
        self.start_point = None
        self.current_point = None
        
        # 多边形绘制
        self.polygon_points = []
        
        # 编辑状态
        self.editing = False
        self.dragging = False
        self.resizing = False
        self.drag_start = None
        self.drag_start_annotation = None
        self.resize_handle = None
        self.resize_start_rect = None
        
        # 图片平移
        self.panning = False
        self.pan_start = None
        self.pan_start_offset = None
        
        # 手柄大小
        self.handle_size = 8
        
        # 类别颜色（动态获取）
        self.class_colors = {}
        
        # 当前选中的类别ID
        self.current_class_id = 0
        
        # 批量处理模式
        self.batch_process_mode = False
        self.batch_process_points = []
        self.batch_process_dialog = None
        
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        # 初始样式会在主题变化时被更新
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['sidebar']};
                border: 2px solid {COLORS['border']};
            }}
        """)
        
        # 设置鼠标追踪
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def load_image(self, image_path: str):
        """加载图像"""
        if not image_path or not os.path.exists(image_path):
            self.current_image = None
            self.current_image_path = None
            self.update()
            return
        
        # 使用OpenCV加载图像
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.current_image = QPixmap.fromImage(qt_image)
            self.current_image_path = image_path
            
            # 重置视图
            self.reset_view()
            self.update()
    
    def reset_view(self):
        """重置视图"""
        if self.current_image is None:
            return
        
        # 计算适应窗口的缩放比例
        widget_rect = self.rect()
        img_width = self.current_image.width()
        img_height = self.current_image.height()
        
        scale_x = (widget_rect.width() - 40) / img_width
        scale_y = (widget_rect.height() - 40) / img_height
        self.image_scale = min(scale_x, scale_y, 1.0)
        
        # 居中显示
        scaled_width = img_width * self.image_scale
        scaled_height = img_height * self.image_scale
        self.image_offset = QPoint(
            int((widget_rect.width() - scaled_width) / 2),
            int((widget_rect.height() - scaled_height) / 2)
        )
    
    def set_annotations(self, annotations: List[Dict]):
        """设置标注数据"""
        self.annotations = annotations
        self.selected_annotation_id = None
        self.update()
    
    def set_tool(self, tool: str):
        """设置当前工具"""
        self.current_tool = tool
        self.drawing = False
        self.polygon_points = []
        self.update()
    
    def image_to_widget(self, x: float, y: float) -> QPoint:
        """图像坐标转换为控件坐标"""
        widget_x = int(x * self.image_scale + self.image_offset.x())
        widget_y = int(y * self.image_scale + self.image_offset.y())
        return QPoint(widget_x, widget_y)
    
    def widget_to_image(self, x: int, y: int) -> Tuple[float, float]:
        """控件坐标转换为图像坐标"""
        img_x = (x - self.image_offset.x()) / self.image_scale
        img_y = (y - self.image_offset.y()) / self.image_scale
        return (img_x, img_y)
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制背景
        painter.fillRect(self.rect(), QColor(COLORS['sidebar']))
        
        if self.current_image is None:
            # 显示提示文字
            painter.setPen(QColor(COLORS['text_secondary']))
            painter.setFont(QFont("Microsoft YaHei", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "请选择一张图片开始标注")
            return
        
        # 绘制图像
        scaled_pixmap = self.current_image.scaled(
            int(self.current_image.width() * self.image_scale),
            int(self.current_image.height() * self.image_scale),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        painter.drawPixmap(self.image_offset, scaled_pixmap)
        
        # 绘制标注
        self.draw_annotations(painter)
        
        # 绘制正在绘制的矩形
        if self.drawing and self.current_tool == 'rectangle' and self.start_point and self.current_point:
            self.draw_drawing_rectangle(painter)
        
        # 绘制正在绘制的多边形
        if self.current_tool == 'polygon' and len(self.polygon_points) > 0:
            self.draw_drawing_polygon(painter)
        
        # 绘制鼠标辅助线
        if self.current_image and self.current_point:
            self.draw_guide_lines(painter)
        
        # 批量处理模式：绘制已选择的像素点
        if self.batch_process_mode and self.batch_process_points:
            self.draw_batch_process_points(painter)
    
    def draw_batch_process_points(self, painter: QPainter):
        """绘制批量处理模式下选择的像素点"""
        # 设置绘制样式
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        
        # 绘制每个像素点
        for i, (img_x, img_y) in enumerate(self.batch_process_points):
            # 将图像坐标转换为控件坐标
            widget_pos = self.image_to_widget(img_x, img_y)
            
            # 绘制圆点
            radius = 6
            painter.drawEllipse(widget_pos, radius, radius)
            
            # 绘制点编号
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Microsoft YaHei", 10, QFont.Weight.Bold))
            painter.drawText(widget_pos.x() + radius + 2, widget_pos.y() - radius, str(i + 1))
            
            # 恢复绘制样式
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.setBrush(QBrush(QColor(255, 0, 0)))
    
    def draw_annotations(self, painter: QPainter):
        """绘制所有标注"""
        for annotation in self.annotations:
            ann_id = annotation['id']
            ann_type = annotation.get('type', 'bbox')
            data = annotation.get('data', {})
            class_id = annotation.get('class_id', 0)
            
            # 获取颜色（优先从class_colors字典，否则使用默认灰色）
            color = self.class_colors.get(class_id, QColor(128, 128, 128))
            if isinstance(color, str):
                color = QColor(color)
            
            # 如果是选中的标注，使用高亮颜色
            is_selected = (ann_id == self.selected_annotation_id)
            pen_width = 3 if is_selected else 2
            
            pen = QPen(color)
            pen.setWidth(pen_width)
            painter.setPen(pen)
            
            brush = QBrush(color)
            brush.setStyle(Qt.BrushStyle.NoBrush)
            painter.setBrush(brush)
            
            if ann_type == 'bbox':
                self.draw_bbox(painter, data, is_selected)
            elif ann_type == 'polygon':
                self.draw_polygon(painter, data, is_selected)
    
    def draw_bbox(self, painter: QPainter, data: Dict, is_selected: bool):
        """绘制矩形框"""
        x = data.get('x', 0)
        y = data.get('y', 0)
        width = data.get('width', 0)
        height = data.get('height', 0)
        
        top_left = self.image_to_widget(x, y)
        bottom_right = self.image_to_widget(x + width, y + height)
        
        rect = QRect(top_left, bottom_right)
        painter.drawRect(rect)
        
        # 如果是选中状态，绘制调整手柄
        if is_selected:
            self.draw_resize_handles(painter, rect)
    
    def draw_polygon(self, painter: QPainter, data: Dict, is_selected: bool):
        """绘制多边形"""
        points = data.get('points', [])
        if len(points) < 3:
            return
        
        widget_points = []
        for point in points:
            widget_point = self.image_to_widget(point['x'], point['y'])
            widget_points.append(widget_point)
        
        # 绘制多边形
        for i in range(len(widget_points)):
            p1 = widget_points[i]
            p2 = widget_points[(i + 1) % len(widget_points)]
            painter.drawLine(p1, p2)
        
        # 绘制顶点
        for point in widget_points:
            painter.drawEllipse(point, 4, 4)
    
    def draw_resize_handles(self, painter: QPainter, rect: QRect):
        """绘制调整大小的手柄"""
        handle_size = 8
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        
        # 四个角
        corners = [
            rect.topLeft(),
            rect.topRight(),
            rect.bottomLeft(),
            rect.bottomRight()
        ]
        
        for corner in corners:
            handle_rect = QRect(
                corner.x() - handle_size // 2,
                corner.y() - handle_size // 2,
                handle_size,
                handle_size
            )
            painter.drawRect(handle_rect)
    
    def draw_drawing_rectangle(self, painter: QPainter):
        """绘制正在绘制的矩形"""
        pen = QPen(QColor(COLORS['primary']))
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        
        # 使用主色调的半透明版本
        primary_color = QColor(COLORS['primary'])
        primary_color.setAlpha(50)
        painter.setBrush(QBrush(primary_color))
        
        rect = QRect(self.start_point, self.current_point)
        painter.drawRect(rect)
    
    def draw_drawing_polygon(self, painter: QPainter):
        """绘制正在绘制的多边形"""
        pen = QPen(QColor(COLORS['primary']))
        pen.setWidth(2)
        painter.setPen(pen)
        
        # 绘制已有点
        for point in self.polygon_points:
            painter.drawEllipse(point, 4, 4)
        
        # 绘制连线
        if len(self.polygon_points) > 1:
            for i in range(len(self.polygon_points) - 1):
                painter.drawLine(self.polygon_points[i], self.polygon_points[i + 1])
        
        # 绘制从最后一点到当前鼠标的线
        if len(self.polygon_points) > 0 and self.current_point:
            painter.drawLine(self.polygon_points[-1], self.current_point)
    
    def draw_guide_lines(self, painter: QPainter):
        """绘制鼠标辅助线"""
        if not self.current_image or not self.current_point:
            return
        
        # 获取图像区域
        img_rect = self.rect()
        scaled_width = int(self.current_image.width() * self.image_scale)
        scaled_height = int(self.current_image.height() * self.image_scale)
        
        # 计算图像显示区域的边界
        img_left = self.image_offset.x()
        img_top = self.image_offset.y()
        img_right = img_left + scaled_width
        img_bottom = img_top + scaled_height
        
        # 获取鼠标位置
        mouse_x = self.current_point.x()
        mouse_y = self.current_point.y()
        
        # 检查鼠标是否在图像区域内
        if not (img_left <= mouse_x <= img_right and img_top <= mouse_y <= img_bottom):
            return
        
        # 设置辅助线样式
        pen = QPen(QColor(255, 255, 255, 150))  # 半透明白色
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidth(2)  # 加粗
        painter.setPen(pen)
        
        # 绘制水平线（穿过鼠标）
        painter.drawLine(img_left, mouse_y, img_right, mouse_y)
        
        # 绘制垂直线（穿过鼠标）
        painter.drawLine(mouse_x, img_top, mouse_x, img_bottom)
    
    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件"""
        if self.current_image is None:
            return
        
        # 批量处理模式：点击选择像素点
        if self.batch_process_mode and event.button() == Qt.MouseButton.LeftButton:
            # 将鼠标位置转换为图像坐标
            img_x, img_y = self.widget_to_image(event.pos().x(), event.pos().y())
            
            # 检查是否在图像范围内
            img_width = self.current_image.width()
            img_height = self.current_image.height()
            
            if 0 <= img_x <= img_width and 0 <= img_y <= img_height:
                # 添加像素点
                self.batch_process_points.append((int(img_x), int(img_y)))
                
                # 更新对话框中的显示
                if self.batch_process_dialog:
                    self.batch_process_dialog.add_point(int(img_x), int(img_y))
                
                self.update()
            return
        
        if event.button() == Qt.MouseButton.LeftButton:
            if self.current_tool == 'rectangle':
                self.drawing = True
                self.start_point = event.pos()
                self.current_point = event.pos()
            elif self.current_tool == 'polygon':
                self.polygon_points.append(event.pos())
                self.update()
            elif self.current_tool == 'move':
                # 检查是否点击了调整手柄
                handle_info = self.get_resize_handle_at(event.pos())
                if handle_info:
                    self.resizing = True
                    self.resize_handle = handle_info['handle']
                    self.selected_annotation_id = handle_info['annotation_id']
                    # 获取选中的标注数据
                    annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
                    if annotation:
                        self.drag_start = event.pos()
                        self.resize_start_rect = annotation['data'].copy()
                        self.annotation_selected.emit(self.selected_annotation_id)
                else:
                    # 检查是否点击了某个标注
                    clicked_annotation = self.get_annotation_at(event.pos())
                    if clicked_annotation:
                        self.selected_annotation_id = clicked_annotation['id']
                        self.dragging = True
                        self.drag_start = event.pos()
                        self.drag_start_annotation = clicked_annotation['data'].copy()
                        self.annotation_selected.emit(self.selected_annotation_id)
                    else:
                        # 没有点击标注，开始平移图片
                        self.panning = True
                        self.pan_start = event.pos()
                        self.pan_start_offset = QPoint(self.image_offset)
                        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        
        self.update()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件"""
        self.current_point = event.pos()
        
        if self.drawing and self.current_tool == 'rectangle':
            self.update()
        elif self.current_tool == 'polygon':
            self.update()
        elif self.current_tool == 'move':
            if self.resizing and self.resize_handle and self.resize_start_rect:
                # 调整大小
                self.resize_annotation(event.pos())
                self.update()
            elif self.dragging and self.drag_start and self.drag_start_annotation:
                # 拖动标注
                self.drag_annotation(event.pos())
                self.update()
            elif self.panning and self.pan_start and self.pan_start_offset:
                # 平移图片
                delta = event.pos() - self.pan_start
                self.image_offset = QPoint(
                    self.pan_start_offset.x() + delta.x(),
                    self.pan_start_offset.y() + delta.y()
                )
                self.update()
            else:
                # 检查鼠标是否在手柄上，改变光标
                handle_info = self.get_resize_handle_at(event.pos())
                if handle_info:
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                else:
                    annotation = self.get_annotation_at(event.pos())
                    if annotation:
                        self.setCursor(Qt.CursorShape.OpenHandCursor)
                    else:
                        # 图片放大时可以平移
                        if self.image_scale > 1.0:
                            self.setCursor(Qt.CursorShape.OpenHandCursor)
                        else:
                            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            # 非移动工具时，设置为箭头光标
            self.setCursor(Qt.CursorShape.ArrowCursor)
        
        # 更新鼠标位置信息
        if self.current_image:
            img_x, img_y = self.widget_to_image(event.pos().x(), event.pos().y())
        
        # 触发重绘以显示辅助线
        self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.drawing and self.current_tool == 'rectangle':
                self.drawing = False
                self.create_rectangle_annotation()
            elif self.resizing:
                self.resizing = False
                self.resize_handle = None
                self.resize_start_rect = None
                # 发送修改信号
                if self.selected_annotation_id is not None:
                    annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
                    if annotation:
                        self.annotation_modified.emit(self.selected_annotation_id, annotation['data'])
            elif self.dragging:
                self.dragging = False
                self.drag_start = None
                self.drag_start_annotation = None
                # 发送修改信号
                if self.selected_annotation_id is not None:
                    annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
                    if annotation:
                        self.annotation_modified.emit(self.selected_annotation_id, annotation['data'])
            elif self.panning:
                self.panning = False
                self.pan_start = None
                self.pan_start_offset = None
        
        # 根据当前状态设置光标
        if self.current_tool == 'move':
            if self.image_scale > 1.0:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            # 非移动工具时，始终设置为箭头光标
            self.setCursor(Qt.CursorShape.ArrowCursor)
        
        self.update()
    
    def get_annotation_at(self, pos: QPoint) -> Optional[Dict]:
        """获取指定位置的标注"""
        for annotation in reversed(self.annotations):
            if self.is_point_in_annotation(pos, annotation):
                return annotation
        return None
    
    def get_resize_handle_at(self, pos: QPoint) -> Optional[Dict]:
        """获取指定位置的调整手柄信息"""
        if self.selected_annotation_id is None:
            return None
        
        annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
        if not annotation or annotation.get('type') != 'bbox':
            return None
        
        data = annotation['data']
        x, y, width, height = data['x'], data['y'], data['width'], data['height']
        
        # 转换为控件坐标
        top_left = self.image_to_widget(x, y)
        bottom_right = self.image_to_widget(x + width, y + height)
        
        # 四个角的手柄
        handles = {
            'top_left': QRect(top_left.x() - self.handle_size, top_left.y() - self.handle_size, 
                             self.handle_size * 2, self.handle_size * 2),
            'top_right': QRect(bottom_right.x() - self.handle_size, top_left.y() - self.handle_size,
                              self.handle_size * 2, self.handle_size * 2),
            'bottom_left': QRect(top_left.x() - self.handle_size, bottom_right.y() - self.handle_size,
                                self.handle_size * 2, self.handle_size * 2),
            'bottom_right': QRect(bottom_right.x() - self.handle_size, bottom_right.y() - self.handle_size,
                                 self.handle_size * 2, self.handle_size * 2),
        }
        
        for handle_name, handle_rect in handles.items():
            if handle_rect.contains(pos):
                return {'handle': handle_name, 'annotation_id': annotation['id']}
        
        return None
    
    def drag_annotation(self, pos: QPoint):
        """拖动标注"""
        if self.drag_start is None or self.drag_start_annotation is None:
            return
        
        annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
        if not annotation:
            return
        
        # 计算拖动偏移（控件坐标）
        delta_x = pos.x() - self.drag_start.x()
        delta_y = pos.y() - self.drag_start.y()
        
        # 转换为图像坐标偏移
        img_delta_x = delta_x / self.image_scale
        img_delta_y = delta_y / self.image_scale
        
        ann_type = annotation.get('type', 'bbox')
        data = annotation['data']
        
        # 获取图像尺寸
        img_width = self.current_image.width() if self.current_image else 0
        img_height = self.current_image.height() if self.current_image else 0
        
        if ann_type == 'bbox':
            new_x = self.drag_start_annotation['x'] + img_delta_x
            new_y = self.drag_start_annotation['y'] + img_delta_y
            width = data.get('width', 0)
            height = data.get('height', 0)
            
            # 限制在图像范围内
            data['x'] = max(0, min(new_x, img_width - width))
            data['y'] = max(0, min(new_y, img_height - height))
        elif ann_type == 'polygon':
            start_points = self.drag_start_annotation['points']
            for i, point in enumerate(data['points']):
                new_x = start_points[i]['x'] + img_delta_x
                new_y = start_points[i]['y'] + img_delta_y
                # 限制在图像范围内
                point['x'] = max(0, min(new_x, img_width))
                point['y'] = max(0, min(new_y, img_height))
    
    def resize_annotation(self, pos: QPoint):
        """调整标注大小"""
        if self.resize_handle is None or self.resize_start_rect is None:
            return
        
        annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
        if not annotation:
            return
        
        data = annotation['data']
        start = self.resize_start_rect
        
        # 将鼠标位置转换为图像坐标
        img_x, img_y = self.widget_to_image(pos.x(), pos.y())
        
        # 限制鼠标位置在图像范围内
        if self.current_image:
            img_width = self.current_image.width()
            img_height = self.current_image.height()
            img_x = max(0, min(img_x, img_width))
            img_y = max(0, min(img_y, img_height))
        
        if self.resize_handle == 'top_left':
            new_x = min(img_x, start['x'] + start['width'])
            new_y = min(img_y, start['y'] + start['height'])
            data['x'] = max(0, new_x)
            data['y'] = max(0, new_y)
            data['width'] = start['x'] + start['width'] - data['x']
            data['height'] = start['y'] + start['height'] - data['y']
        elif self.resize_handle == 'top_right':
            new_y = min(img_y, start['y'] + start['height'])
            data['x'] = start['x']
            data['y'] = max(0, new_y)
            data['width'] = min(img_x, img_width) - start['x'] if self.current_image else img_x - start['x']
            data['height'] = start['y'] + start['height'] - data['y']
        elif self.resize_handle == 'bottom_left':
            new_x = min(img_x, start['x'] + start['width'])
            data['x'] = max(0, new_x)
            data['y'] = start['y']
            data['width'] = start['x'] + start['width'] - data['x']
            data['height'] = min(img_y, img_height) - start['y'] if self.current_image else img_y - start['y']
        elif self.resize_handle == 'bottom_right':
            data['x'] = start['x']
            data['y'] = start['y']
            data['width'] = min(img_x, img_width) - start['x'] if self.current_image else img_x - start['x']
            data['height'] = min(img_y, img_height) - start['y'] if self.current_image else img_y - start['y']
        
        # 确保宽度和高度为正且不超过图像范围
        if data['width'] < 0:
            data['x'] += data['width']
            data['width'] = abs(data['width'])
        if data['height'] < 0:
            data['y'] += data['height']
            data['height'] = abs(data['height'])
        
        # 最终限制在图像范围内
        if self.current_image:
            data['x'] = max(0, min(data['x'], img_width))
            data['y'] = max(0, min(data['y'], img_height))
            data['width'] = min(data['width'], img_width - data['x'])
            data['height'] = min(data['height'], img_height - data['y'])
    
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """鼠标双击事件 - 完成多边形绘制"""
        if self.current_tool == 'polygon' and len(self.polygon_points) >= 3:
            self.create_polygon_annotation()
    
    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮事件 - 缩放"""
        if self.current_image is None:
            return
        
        # 获取鼠标位置
        mouse_pos = event.position().toPoint()
        
        # 计算缩放前鼠标对应的图像坐标
        img_x_before = (mouse_pos.x() - self.image_offset.x()) / self.image_scale
        img_y_before = (mouse_pos.y() - self.image_offset.y()) / self.image_scale
        
        # 计算缩放因子
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        
        # 应用缩放
        new_scale = self.image_scale * zoom_factor
        new_scale = max(0.1, min(5.0, new_scale))  # 限制缩放范围
        
        # 调整偏移量，使鼠标位置对应的图像点保持不变
        self.image_offset = QPoint(
            int(mouse_pos.x() - img_x_before * new_scale),
            int(mouse_pos.y() - img_y_before * new_scale)
        )
        self.image_scale = new_scale
        
        self.update()
    
    def keyPressEvent(self, event: QKeyEvent):
        """键盘事件"""
        from PyQt6.QtCore import QSettings
        
        # 获取快捷键设置
        settings = QSettings("EzYOLO", "Settings")
        reset_view_key = settings.value("reset_view_shortcut", "R").upper()
        
        # 重置视图快捷键
        if event.text().upper() == reset_view_key:
            self.reset_view()
            self.update()
            return
        
        if event.key() == Qt.Key.Key_Escape:
            # 取消当前操作
            if self.current_tool == 'polygon' and len(self.polygon_points) > 0:
                self.polygon_points = []
                self.update()
            elif self.drawing:
                self.drawing = False
                self.update()
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            # 完成多边形绘制
            if self.current_tool == 'polygon' and len(self.polygon_points) >= 3:
                self.create_polygon_annotation()
        elif event.key() == Qt.Key.Key_Delete:
            # 删除选中的标注
            if self.selected_annotation_id is not None:
                self.annotation_deleted.emit(self.selected_annotation_id)
        else:
            # 将未处理的事件传递给父组件
            self.parent().keyPressEvent(event)
    
    def check_annotation_selection(self, pos: QPoint):
        """检查是否选中了某个标注"""
        for annotation in reversed(self.annotations):  # 从后往前检查，先检查上面的
            if self.is_point_in_annotation(pos, annotation):
                self.selected_annotation_id = annotation['id']
                self.annotation_selected.emit(annotation['id'])
                self.update()
                return
        
        # 没有选中任何标注
        self.selected_annotation_id = None
        self.update()
    
    def is_point_in_annotation(self, pos: QPoint, annotation: Dict) -> bool:
        """检查点是否在标注内"""
        ann_type = annotation.get('type', 'bbox')
        data = annotation.get('data', {})
        
        if ann_type == 'bbox':
            x = data.get('x', 0)
            y = data.get('y', 0)
            width = data.get('width', 0)
            height = data.get('height', 0)
            
            top_left = self.image_to_widget(x, y)
            bottom_right = self.image_to_widget(x + width, y + height)
            
            return (top_left.x() <= pos.x() <= bottom_right.x() and
                    top_left.y() <= pos.y() <= bottom_right.y())
        
        elif ann_type == 'polygon':
            # 简化的多边形检测
            points = data.get('points', [])
            if len(points) < 3:
                return False
            
            # 转换为控件坐标
            widget_points = []
            for point in points:
                widget_point = self.image_to_widget(point['x'], point['y'])
                widget_points.append((widget_point.x(), widget_point.y()))
            
            # 使用射线法检测点是否在多边形内
            return self.point_in_polygon(pos.x(), pos.y(), widget_points)
        
        return False
    
    def point_in_polygon(self, x: int, y: int, polygon: List[Tuple[int, int]]) -> bool:
        """射线法判断点是否在多边形内"""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def create_rectangle_annotation(self):
        """创建矩形标注"""
        if self.start_point is None or self.current_point is None:
            return
        
        # 转换为图像坐标
        x1, y1 = self.widget_to_image(self.start_point.x(), self.start_point.y())
        x2, y2 = self.widget_to_image(self.current_point.x(), self.current_point.y())
        
        # 确保 x1 < x2, y1 < y2
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # 限制坐标在图像范围内
        if self.current_image:
            img_width = self.current_image.width()
            img_height = self.current_image.height()
            
            # 限制x和y在图像范围内
            x = max(0, min(x, img_width))
            y = max(0, min(y, img_height))
            
            # 限制width和height不超出图像范围
            width = min(width, img_width - x)
            height = min(height, img_height - y)
        
        # 过滤太小的标注
        if width < 5 or height < 5:
            return
        
        annotation = {
            'type': 'bbox',
            'class_id': self.current_class_id,
            'data': {
                'x': x,
                'y': y,
                'width': width,
                'height': height
            }
        }
        
        self.annotation_created.emit(annotation)
        self.start_point = None
        self.current_point = None
    
    def create_polygon_annotation(self):
        """创建多边形标注"""
        if len(self.polygon_points) < 3:
            return
        
        # 转换为图像坐标
        points = []
        for point in self.polygon_points:
            x, y = self.widget_to_image(point.x(), point.y())
            
            # 限制坐标在图像范围内
            if self.current_image:
                img_width = self.current_image.width()
                img_height = self.current_image.height()
                x = max(0, min(x, img_width))
                y = max(0, min(y, img_height))
            
            points.append({'x': x, 'y': y})
        
        annotation = {
            'type': 'polygon',
            'class_id': self.current_class_id,
            'data': {
                'points': points
            }
        }
        
        self.annotation_created.emit(annotation)
        self.polygon_points = []
    
    def resizeEvent(self, event):
        """窗口大小改变"""
        super().resizeEvent(event)
        if self.current_image:
            self.reset_view()


class AnnotatePage(QWidget):
    """标注页面"""
    
    def __init__(self):
        super().__init__()
        self.current_project_id = None
        self.current_image_id = None
        self.current_image_data = None
        self.images = []
        self.annotations = []
        self.classes = []
        self.current_class_id = 0
        self.history = []  # 撤销历史
        self.history_index = -1
        self.load_worker = None  # 加载线程
        
        # 自动标注相关属性
        self.auto_label_dialog = None
        self.model_manager = None
        self.batch_labeling_manager = None
        
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：图片列表
        self.left_panel = self.create_left_panel()
        splitter.addWidget(self.left_panel)
        
        # 中间：标注画布
        self.center_panel = self.create_center_panel()
        splitter.addWidget(self.center_panel)
        
        # 右侧：属性面板
        self.right_panel = self.create_right_panel()
        splitter.addWidget(self.right_panel)
        
        # 设置分割器比例
        splitter.setSizes([250, 700, 250])
        
        self.main_layout.addWidget(splitter)
        
        # 底部状态栏
        self.status_bar = self.create_status_bar()
        self.main_layout.addWidget(self.status_bar)
    
    def create_left_panel(self) -> QWidget:
        """创建左侧面板 - 图片列表"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setMaximumWidth(300)
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['panel']};
                border-right: 1px solid {COLORS['border']};
            }}
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # 标题
        title = QLabel("图片列表")
        title.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        # 任务类型选择器
        task_layout = QHBoxLayout()
        task_label = QLabel("任务类型:")
        task_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        task_layout.addWidget(task_label)
        
        self.task_combo = QComboBox()
        self.task_combo.addItems(["detect", "segment", "pose", "classify", "obb"])
        self.task_combo.currentTextChanged.connect(self.on_task_changed)
        task_layout.addWidget(self.task_combo)
        layout.addLayout(task_layout)
        
        # 筛选
        self.image_filter = QComboBox()
        self.image_filter.addItems(["全部", "未标注", "已标注"])
        self.image_filter.currentTextChanged.connect(self.filter_images)
        layout.addWidget(self.image_filter)
        
        # 图片列表
        self.image_list = QListWidget()
        self.image_list.setIconSize(QSize(80, 80))
        self.image_list.setSpacing(4)
        self.image_list.itemClicked.connect(self.on_image_selected)
        self.image_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {COLORS['sidebar']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
            QListWidget::item {{
                background-color: {COLORS['panel']};
                border-radius: 4px;
                padding: 4px;
            }}
            QListWidget::item:selected {{
                background-color: {COLORS['primary']};
            }}
        """)
        layout.addWidget(self.image_list)
        
        # 导航按钮
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀ 上一张")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next = QPushButton("下一张 ▶")
        self.btn_next.clicked.connect(self.next_image)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        layout.addLayout(nav_layout)
        
        return panel
    
    def create_center_panel(self) -> QWidget:
        """创建中间面板 - 标注画布"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # 工具栏
        toolbar = self.create_toolbar()
        layout.addWidget(toolbar)
        
        # 标注画布
        self.canvas = AnnotationCanvas()
        self.canvas.annotation_created.connect(self.on_annotation_created)
        self.canvas.annotation_selected.connect(self.on_annotation_selected)
        self.canvas.annotation_modified.connect(self.on_annotation_modified)
        self.canvas.annotation_deleted.connect(self.on_annotation_deleted)
        layout.addWidget(self.canvas, stretch=1)
        
        return panel
    
    def create_toolbar(self) -> QToolBar:
        """创建工具栏"""
        toolbar = QToolBar()
        toolbar.setStyleSheet(f"""
            QToolBar {{
                background-color: {COLORS['panel']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 4px;
            }}
        """)
        
        # 工具按钮组
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        
        # 矩形工具
        self.btn_rectangle = QPushButton("🟦 矩形 (W)")
        self.btn_rectangle.setCheckable(True)
        self.btn_rectangle.setChecked(True)
        self.btn_rectangle.clicked.connect(lambda: self.set_tool('rectangle'))
        toolbar.addWidget(self.btn_rectangle)
        self.tool_group.addButton(self.btn_rectangle)
        
        # 多边形工具
        self.btn_polygon = QPushButton("🛑 多边形 (P)")
        self.btn_polygon.setCheckable(True)
        self.btn_polygon.clicked.connect(lambda: self.set_tool('polygon'))
        toolbar.addWidget(self.btn_polygon)
        self.tool_group.addButton(self.btn_polygon)
        
        # 移动工具
        self.btn_move = QPushButton("✋ 移动 (V)")
        self.btn_move.setCheckable(True)
        self.btn_move.clicked.connect(lambda: self.set_tool('move'))
        toolbar.addWidget(self.btn_move)
        self.tool_group.addButton(self.btn_move)
        
        toolbar.addSeparator()
        
        # 删除按钮
        self.btn_delete = QPushButton("🗑️ 删除 (D)")
        self.btn_delete.clicked.connect(self.delete_selected_annotation)
        toolbar.addWidget(self.btn_delete)
        
        # 撤销按钮
        self.btn_undo = QPushButton("↶ 撤销 (Ctrl+Z)")
        self.btn_undo.clicked.connect(self.undo)
        toolbar.addWidget(self.btn_undo)
        
        # 自动标注按钮
        toolbar.addSeparator()
        self.btn_auto_label = QPushButton("🤖 自动标注")
        self.btn_auto_label.setMenu(self.create_auto_label_menu())
        # 美化按钮样式
        primary_color = COLORS['primary']
        self.btn_auto_label.setStyleSheet(
            f"QPushButton {{"  
            f"    background-color: {primary_color};" 
            f"    color: white;" 
            f"    border: none;" 
            f"    border-radius: 4px;" 
            f"    padding: 6px 12px;" 
            f"    font-weight: bold;" 
            f"}}" 
            f"QPushButton:hover {{" 
            f"    background-color: {primary_color};" 
            f"}}" 
            f"QPushButton::menu-indicator {{" 
            f"    image: none;" 
            f"    subcontrol-position: right center;" 
            f"    subcontrol-origin: padding;" 
            f"    width: 16px;" 
            f"    height: 16px;" 
            f"}}" 
            f"QPushButton::menu-indicator::hover {{" 
            f"    image: none;" 
            f"}}"
        )
        toolbar.addWidget(self.btn_auto_label)
        
        # 批量处理按钮
        toolbar.addSeparator()
        self.btn_batch_process = QPushButton("📋 批量处理")
        self.btn_batch_process.clicked.connect(self.show_batch_process_dialog)
        self.btn_batch_process.setStyleSheet(
            f"QPushButton {{"  
            f"    background-color: {COLORS['secondary']};" 
            f"    color: white;" 
            f"    border: none;" 
            f"    border-radius: 4px;" 
            f"    padding: 6px 12px;" 
            f"    font-weight: bold;" 
            f"}}" 
            f"QPushButton:hover {{" 
            f"    background-color: {COLORS['secondary']};" 
            f"}}"
        )
        toolbar.addWidget(self.btn_batch_process)
        
        return toolbar
    
    def create_right_panel(self) -> QWidget:
        """创建右侧面板 - 属性面板"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setMaximumWidth(300)
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['panel']};
                border-left: 1px solid {COLORS['border']};
            }}
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # 类别列表
        class_group = QGroupBox("类别列表")
        class_group.setStyleSheet(f"""
            QGroupBox {{
                color: {COLORS['text_primary']};
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }}
        """)
        class_layout = QVBoxLayout(class_group)
        
        self.class_list = QListWidget()
        self.class_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {COLORS['sidebar']};
                border: 1px solid {COLORS['border']};
            }}
            QListWidget::item {{
                padding: 4px;
            }}
            QListWidget::item:selected {{
                background-color: {COLORS['primary']};
            }}
        """)
        self.class_list.itemClicked.connect(self.on_class_selected)
        self.class_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.class_list.customContextMenuRequested.connect(self.show_class_context_menu)
        class_layout.addWidget(self.class_list)
        
        # 添加类别按钮
        self.btn_add_class = QPushButton("+ 添加类别")
        self.btn_add_class.clicked.connect(self.add_class)
        class_layout.addWidget(self.btn_add_class)
        
        layout.addWidget(class_group)
        
        # 标注属性
        attr_group = QGroupBox("标注属性")
        attr_group.setStyleSheet(class_group.styleSheet())
        attr_layout = QFormLayout(attr_group)
        
        # 位置信息
        self.attr_x = QSpinBox()
        self.attr_x.setRange(0, 10000)
        self.attr_x.valueChanged.connect(self.on_attr_value_changed)
        attr_layout.addRow("X:", self.attr_x)
        
        self.attr_y = QSpinBox()
        self.attr_y.setRange(0, 10000)
        self.attr_y.valueChanged.connect(self.on_attr_value_changed)
        attr_layout.addRow("Y:", self.attr_y)
        
        self.attr_width = QSpinBox()
        self.attr_width.setRange(0, 10000)
        self.attr_width.valueChanged.connect(self.on_attr_value_changed)
        attr_layout.addRow("宽度:", self.attr_width)
        
        self.attr_height = QSpinBox()
        self.attr_height.setRange(0, 10000)
        self.attr_height.valueChanged.connect(self.on_attr_value_changed)
        attr_layout.addRow("高度:", self.attr_height)
        
        # 类别选择
        self.attr_class = QComboBox()
        self.attr_class.currentIndexChanged.connect(self.on_attr_class_changed)
        attr_layout.addRow("类别:", self.attr_class)
        
        # 应用按钮
        self.btn_apply_attr = QPushButton("应用修改")
        self.btn_apply_attr.clicked.connect(self.apply_annotation_changes)
        attr_layout.addRow(self.btn_apply_attr)
        
        layout.addWidget(attr_group)
        
        # 快捷键说明
        shortcut_group = QGroupBox("快捷键")
        shortcut_group.setStyleSheet(class_group.styleSheet())
        shortcut_layout = QVBoxLayout(shortcut_group)
        
        shortcuts_text = """
W - 矩形工具
P - 多边形工具
V - 移动工具
D - 删除选中
Ctrl+Z - 撤销
Ctrl+Y - 重做
方向键 - 切换图片
Delete - 删除
Esc - 取消操作
        """
        shortcuts_label = QLabel(shortcuts_text)
        shortcuts_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        shortcut_layout.addWidget(shortcuts_label)
        
        layout.addWidget(shortcut_group)
        
        # 导出功能
        export_group = QGroupBox("数据导出")
        export_group.setStyleSheet(class_group.styleSheet())
        export_layout = QVBoxLayout(export_group)
        
        # 导出格式选择
        format_layout = QFormLayout()
        self.export_format = QComboBox()
        self.export_format.addItems(["YOLO格式", "COCO格式"])
        format_layout.addRow("导出格式:", self.export_format)
        export_layout.addLayout(format_layout)
        
        # 导出按钮
        self.btn_export_annotations = QPushButton("📤 导出标注文件")
        self.btn_export_annotations.clicked.connect(self.export_annotations)
        export_layout.addWidget(self.btn_export_annotations)
        
        self.btn_export_dataset = QPushButton("📦 导出完整数据集")
        self.btn_export_dataset.clicked.connect(self.export_dataset)
        export_layout.addWidget(self.btn_export_dataset)
        
        layout.addWidget(export_group)
        
        layout.addStretch()
        
        return panel
    
    def create_auto_label_menu(self) -> QMenu:
        """创建自动标注下拉菜单"""
        menu = QMenu()
        
        # 设置选项
        action_settings = menu.addAction("⚙️ 设置")
        action_settings.triggered.connect(self.show_auto_label_settings)
        
        # 单张推理选项
        action_single = menu.addAction("🔍 单张推理")
        action_single.triggered.connect(self.run_single_inference)
        
        # 批量推理选项
        action_batch = menu.addAction("📋 批量推理")
        action_batch.triggered.connect(self.run_batch_inference)
        
        return menu
    
    def create_status_bar(self) -> QFrame:
        """创建状态栏"""
        status_bar = QFrame()
        status_bar.setFrameStyle(QFrame.Shape.StyledPanel)
        status_bar.setMaximumHeight(40)
        status_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['panel']};
                border-top: 1px solid {COLORS['border']};
            }}
            QLabel {{
                color: {COLORS['text_secondary']};
                font-size: 12px;
                padding: 4px 12px;
            }}
        """)
        
        layout = QHBoxLayout(status_bar)
        layout.setContentsMargins(8, 4, 8, 4)
        
        self.status_image = QLabel("当前: 0/0")
        layout.addWidget(self.status_image)
        
        layout.addWidget(QLabel("|"))
        
        self.status_annotation = QLabel("标注: 0")
        layout.addWidget(self.status_annotation)
        
        layout.addWidget(QLabel("|"))
        
        self.status_position = QLabel("位置: --")
        layout.addWidget(self.status_position)
        
        layout.addStretch()
        
        self.status_tool = QLabel("工具: 矩形")
        layout.addWidget(self.status_tool)
        
        return status_bar
    
    def set_project(self, project_id: int):
        """设置当前项目"""
        # 即使项目ID相同，也重新加载数据（确保图片列表更新）
        self.current_project_id = project_id
        
        # 显示加载动画
        self.loading_overlay = LoadingOverlay(self, "正在加载项目数据...")
        self.loading_overlay.show_loading()
        
        # 创建后台线程来加载项目数据
        from PyQt6.QtCore import QThread, pyqtSignal
        
        class ProjectLoadThread(QThread):
            """项目数据加载线程"""
            
            data_loaded = pyqtSignal(dict)
            finished = pyqtSignal()
            
            def __init__(self, project_id):
                super().__init__()
                self.project_id = project_id
            
            def run(self):
                """运行线程"""
                try:
                    # 加载项目信息
                    project = db.get_project(self.project_id)
                    classes = []
                    
                    if project:
                        # 加载类别
                        import json
                        try:
                            classes = json.loads(project.get('classes', '[]'))
                        except:
                            classes = []
                        
                        if not classes:
                            # 添加默认类别
                            classes = [
                                {'id': 0, 'name': 'person', 'color': '#FF0000'},
                                {'id': 1, 'name': 'car', 'color': '#00FF00'}
                            ]
                    
                    # 加载图片列表数据
                    images = db.get_project_images(self.project_id)
                    
                    # 发送加载完成信号
                    self.data_loaded.emit({'classes': classes, 'images': images})
                finally:
                    self.finished.emit()
        
        # 创建并启动线程
        self.load_thread = ProjectLoadThread(project_id)
        self.load_thread.data_loaded.connect(self.on_project_data_loaded)
        self.load_thread.finished.connect(self.on_project_load_finished)
        self.load_thread.start()
    
    def on_project_data_loaded(self, data):
        """项目数据加载完成回调"""
        # 更新类别
        self.classes = data.get('classes', [])
        self.update_class_list()
        
        # 保存图片数据
        self.images = data.get('images', [])
        
        # 设置任务类型选择器
        if self.current_project_id:
            project = db.get_project(self.current_project_id)
            if project and project.get('type'):
                task_type = project['type']
                if task_type in ['detect', 'segment', 'pose', 'classify', 'obb']:
                    index = self.task_combo.findText(task_type)
                    if index >= 0:
                        self.task_combo.setCurrentIndex(index)
        
        # 开始加载图片列表（使用多线程加载缩略图）
        self.load_image_list()
    
    def on_project_load_finished(self):
        """项目加载完成回调"""
        # 隐藏加载动画
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.hide_loading()
            self.loading_overlay.deleteLater()
            delattr(self, 'loading_overlay')
        
        # 清理线程
        if hasattr(self, 'load_thread'):
            self.load_thread.wait()
            delattr(self, 'load_thread')
    
    def load_image_list(self):
        """加载图片列表 - 使用多线程"""
        # 停止之前的加载
        if self.load_worker and self.load_worker.isRunning():
            self.load_worker.stop()
            self.load_worker.wait()
        
        self.image_list.clear()
        self.images = []
        
        if not self.current_project_id:
            return
        
        # 从数据库获取图片列表（很快）
        self.images = db.get_project_images(self.current_project_id)
        
        # 先创建所有列表项（显示占位符）
        for image in self.images:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, image['id'])
            
            # 设置显示文本
            status_text = "✓" if image.get('status') == 'annotated' else "○"
            item.setText(f"{status_text} {image['filename']}")
            
            self.image_list.addItem(item)
        
        self.update_status_bar()
        
        # 启动后台加载线程加载缩略图
        if self.images:
            self.load_worker = AnnotateImageLoadWorker(self.images)
            self.load_worker.image_loaded.connect(self.on_image_loaded)
            self.load_worker.finished_loading.connect(self.on_load_finished)
            self.load_worker.start()
    
    def on_image_loaded(self, index: int, pixmap: QPixmap):
        """单个图片加载完成回调"""
        if index < self.image_list.count():
            item = self.image_list.item(index)
            if item:
                item.setIcon(QIcon(pixmap))
    
    def on_load_finished(self):
        """加载完成回调"""
        pass
    
    def update_image_list_display(self):
        """更新图片列表显示"""
        # 重新加载图片数据
        if self.current_project_id:
            self.images = db.get_project_images(self.current_project_id)
            
            # 更新图片列表项
            for i in range(self.image_list.count()):
                item = self.image_list.item(i)
                image_id = item.data(Qt.ItemDataRole.UserRole)
                
                # 找到对应的图片数据
                image_data = next((img for img in self.images if img['id'] == image_id), None)
                if image_data:
                    # 更新显示文本
                    status_text = "✓" if image_data.get('status') == 'annotated' else "○"
                    item.setText(f"{status_text} {image_data['filename']}")
    
    def update_class_list(self):
        """更新类别列表"""
        self.class_list.clear()
        self.attr_class.clear()
        
        # 更新canvas的类别颜色
        class_colors = {}
        for cls in self.classes:
            class_colors[cls['id']] = cls.get('color', '#808080')
        self.canvas.class_colors = class_colors
        
        for cls in self.classes:
            # 创建带颜色的列表项
            item = QListWidgetItem(f"■ {cls['name']}")
            item.setData(Qt.ItemDataRole.UserRole, cls['id'])
            
            # 设置颜色
            color = QColor(cls.get('color', '#808080'))
            item.setForeground(color)
            
            self.class_list.addItem(item)
            
            # 添加到属性面板的下拉框
            self.attr_class.addItem(cls['name'], cls['id'])
        
        # 默认选中第一个类别
        if self.class_list.count() > 0:
            self.class_list.setCurrentRow(0)
            self.on_class_selected()
    
    def init_auto_label_components(self):
        """初始化自动标注组件"""
        if not self.model_manager:
            self.model_manager = ModelManager()
        
        if not self.auto_label_dialog:
            self.auto_label_dialog = AutoLabelDialog(self)
            self.auto_label_dialog.single_inference_requested.connect(self.on_single_inference_requested)
            self.auto_label_dialog.batch_inference_requested.connect(self.on_batch_inference_requested)
        
        if not self.batch_labeling_manager:
            self.batch_labeling_manager = BatchLabelingManager()
            self.batch_labeling_manager.progress_updated.connect(self.on_batch_inference_progress)
            self.batch_labeling_manager.batch_completed.connect(self.on_batch_inference_completed)
        
        # 初始化加载动画
        if not hasattr(self, 'loading_label'):
            self.loading_label = QLabel("加载中...")
            self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.loading_label.setStyleSheet("""
                QLabel {
                    background-color: rgba(0, 0, 0, 0.7);
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                    padding: 20px;
                    border-radius: 8px;
                }
            """)
            self.loading_label.hide()
            # 将加载动画添加到主布局
            self.main_layout.addWidget(self.loading_label)
            self.loading_label.setGeometry(
                self.width() // 2 - 100,
                self.height() // 2 - 50,
                200,
                100
            )
            self.loading_label.raise_()
    
    def show_loading_animation(self, message):
        """显示加载动画"""
        if not hasattr(self, 'loading_label'):
            self.init_auto_label_components()
        
        self.loading_label.setText(message)
        self.loading_label.setGeometry(
            self.width() // 2 - 150,
            self.height() // 2 - 50,
            300,
            100
        )
        self.loading_label.show()
        self.loading_label.raise_()
        # 强制刷新界面
        self.repaint()
    
    def hide_loading_animation(self):
        """隐藏加载动画"""
        if hasattr(self, 'loading_label'):
            self.loading_label.hide()
    
    def show_auto_label_settings(self):
        """显示自动标注设置对话框"""
        self.init_auto_label_components()
        self.auto_label_dialog.set_classes(self.classes)
        
        # 显示对话框
        if self.auto_label_dialog.exec() == QDialog.DialogCode.Accepted:
            # 保存设置
            self.auto_label_settings = {
                'model_path': self.auto_label_dialog.get_model_path(),
                'conf_threshold': self.auto_label_dialog.sb_conf_threshold.value(),
                'iou_threshold': self.auto_label_dialog.sb_iou_threshold.value(),
                'class_mapping': self.auto_label_dialog.get_class_mappings(),
                'only_unlabeled': self.auto_label_dialog.chk_only_unlabeled.isChecked(),
                'overwrite_labels': self.auto_label_dialog.chk_overwrite.isChecked()
            }
            
            # 更新标注页面的类别列表
            if hasattr(self.auto_label_dialog, 'project_classes'):
                new_classes = self.auto_label_dialog.project_classes
                if new_classes != self.classes:
                    self.classes = new_classes
                    # 更新数据库
                    db.update_project(self.current_project_id, classes=self.classes)
                    # 更新界面
                    self.update_class_list()
                    QMessageBox.information(self, "成功", "类别列表已更新")
    
    def show_batch_process_dialog(self):
        """显示批量处理标注对话框"""
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return
        
        if not self.images:
            QMessageBox.warning(self, "提示", "项目中没有图片")
            return
        
        # 创建对话框
        dialog = BatchProcessDialog(self, self.classes, len(self.images))
        dialog.process_requested.connect(self.on_batch_process_requested)
        
        # 进入像素点选择模式
        self.batch_process_dialog = dialog
        self.canvas.batch_process_mode = True
        self.canvas.batch_process_points = []
        self.canvas.batch_process_dialog = dialog
        
        # 显示对话框（非模态，允许在图片上点击）
        dialog.show()
    
    def on_batch_process_requested(self, config):
        """处理批量处理请求"""
        # 退出像素点选择模式
        self.canvas.batch_process_mode = False
        self.canvas.batch_process_points = []
        self.canvas.batch_process_dialog = None
        
        # 执行批量处理
        self.execute_batch_process(config)
    
    def execute_batch_process(self, config):
        """执行批量处理"""
        points = config['points']
        start_idx = config['start_idx']
        end_idx = config['end_idx']
        operation = config['operation']
        
        # 获取处理范围内的图片
        images_to_process = self.images[start_idx:end_idx+1]
        
        if not images_to_process:
            QMessageBox.warning(self, "提示", "没有需要处理的图片")
            return
        
        # 显示进度对话框
        from PyQt6.QtWidgets import QProgressDialog
        progress = QProgressDialog("正在批量处理标注...", "取消", 0, len(images_to_process), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        processed_count = 0
        modified_count = 0
        
        try:
            for i, image_data in enumerate(images_to_process):
                if progress.wasCanceled():
                    break
                
                progress.setValue(i)
                progress.setLabelText(f"正在处理第 {i+1}/{len(images_to_process)} 张图片...")
                
                image_id = image_data['id']
                annotations = db.get_image_annotations(image_id)
                
                if not annotations:
                    continue
                
                # 检查每个标注是否覆盖选择的像素点
                for annotation in annotations:
                    annotation_data = annotation.get('data', {})
                    annotation_type = annotation.get('type', 'bbox')
                    
                    # 检查标注是否覆盖任何选择的像素点
                    covers_point = False
                    for point in points:
                        px, py = point
                        if self.is_point_in_annotation_data(px, py, annotation_data, annotation_type):
                            covers_point = True
                            break
                    
                    if covers_point:
                        if operation == 'delete':
                            # 批量删除：检查类别是否在目标类别列表中
                            target_classes = config.get('target_classes', [])
                            if annotation.get('class_id') in target_classes:
                                db.delete_annotation(annotation['id'])
                                modified_count += 1
                        else:
                            # 批量修改：检查类别是否在源类别列表中
                            source_classes = config.get('source_classes', [])
                            target_class = config.get('target_class')
                            if annotation.get('class_id') in source_classes:
                                db.update_annotation(annotation['id'], class_id=target_class)
                                modified_count += 1
                
                processed_count += 1
            
            progress.setValue(len(images_to_process))
            
            # 显示结果
            QMessageBox.information(
                self, 
                "批量处理完成", 
                f"处理完成！\n处理了 {processed_count} 张图片\n修改了 {modified_count} 个标注"
            )
            
            # 刷新当前图片的标注显示
            if self.current_image_id:
                self.load_annotations()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"批量处理出错: {str(e)}")
    
    def is_point_in_annotation_data(self, px: int, py: int, data: dict, ann_type: str) -> bool:
        """检查点是否在标注数据内"""
        if ann_type == 'bbox':
            x = data.get('x', 0)
            y = data.get('y', 0)
            width = data.get('width', 0)
            height = data.get('height', 0)
            return x <= px <= x + width and y <= py <= y + height
        elif ann_type == 'polygon':
            points = data.get('points', [])
            if len(points) < 3:
                return False
            # 使用射线法判断点是否在多边形内
            return self.point_in_polygon(px, py, points)
        return False
    
    def point_in_polygon(self, x: int, y: int, polygon: list) -> bool:
        """射线法判断点是否在多边形内"""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]['x'], polygon[0]['y']
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]['x'], polygon[i % n]['y']
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def run_single_inference(self):
        """运行单张图像推理"""
        if not self.current_image_data:
            QMessageBox.warning(self, "提示", "请先选择一张图片")
            return
        
        # 显示加载动画
        self.show_loading_animation("正在进行单张推理...")
        
        # 使用保存的参数或默认参数运行推理
        try:
            from core.auto_labeler import AutoLabeler
            
            # 获取模型参数（优先使用保存的参数）
            if hasattr(self, 'auto_label_settings'):
                settings = self.auto_label_settings
                model_path = settings.get('model_path', "yolov8n")
                conf_threshold = settings.get('conf_threshold', 0.5)
                iou_threshold = settings.get('iou_threshold', 0.45)
                class_mapping = settings.get('class_mapping', {})
                overwrite_labels = settings.get('overwrite_labels', False)
            else:
                # 默认模型参数
                model_path = "yolov8n"
                conf_threshold = 0.5
                iou_threshold = 0.45
                class_mapping = {}
                overwrite_labels = False
            
            labeler = AutoLabeler(model_path, self.model_manager)
            annotations = labeler.process_image(
                self.current_image_data['storage_path'], 
                conf_threshold, 
                iou_threshold, 
                class_mapping
            )
            
            # 更新当前图像的标注
            if annotations and self.current_image_id:
                # 保存标注到数据库
                # 如果需要覆盖原标签，先删除所有原标注
                if overwrite_labels:
                    db.delete_image_annotations(self.current_image_id)
                
                for annotation in annotations:
                    # 获取类别名称
                    class_id = annotation['class_id']
                    
                    # 检查类别ID是否在项目类别范围内
                    existing_class = next((cls for cls in self.classes if cls['id'] == class_id), None)
                    if existing_class:
                        class_name = existing_class['name']
                    else:
                        # 创建新类别
                        class_name = f"class_{class_id}"
                        # 生成随机颜色
                        import random
                        color = f"#{random.randint(0, 0xFFFFFF):06x}"
                        new_class = {
                            'id': class_id,
                            'name': class_name,
                            'color': color
                        }
                        self.classes.append(new_class)
                        # 更新项目类别
                        db.update_project(self.current_project_id, classes=self.classes)
                    
                    # 保存标注
                    db.add_annotation(
                        self.current_image_id,
                        self.current_project_id,
                        class_id,
                        class_name,
                        annotation['type'],
                        annotation['data']
                    )
                
                # 重新加载当前图像的标注
                self.load_current_image_annotations()
                
                QMessageBox.information(self, "成功", "自动标注完成！")
            else:
                # 如果没有检测到目标，但需要覆盖原标签，也删除原标注
                if overwrite_labels and self.current_image_id:
                    db.delete_image_annotations(self.current_image_id)
                    self.load_current_image_annotations()
                QMessageBox.information(self, "提示", "未检测到目标")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"自动标注失败: {str(e)}")
        finally:
            # 隐藏加载动画
            self.hide_loading_animation()
    
    def run_batch_inference(self):
        """运行批量推理"""
        if not self.current_project_id or len(self.images) == 0:
            QMessageBox.warning(self, "提示", "项目中没有图片")
            return
        
        # 显示加载动画
        self.show_loading_animation("正在进行批量推理...")
        
        # 使用保存的参数或默认参数运行批量推理
        try:
            # 获取模型参数（优先使用保存的参数）
            if hasattr(self, 'auto_label_settings'):
                settings = self.auto_label_settings
                model_path = settings.get('model_path', "yolov8n")
                conf_threshold = settings.get('conf_threshold', 0.5)
                iou_threshold = settings.get('iou_threshold', 0.45)
                class_mapping = settings.get('class_mapping', {})
                only_unlabeled = settings.get('only_unlabeled', True)
            else:
                # 默认模型参数
                model_path = "yolov8n"
                conf_threshold = 0.5
                iou_threshold = 0.45
                class_mapping = {}
                only_unlabeled = True
            
            # 过滤图像（如果只处理未标注的）
            if only_unlabeled:
                # 更宽松的过滤逻辑，包括status为None或空的情况
                filtered_images = [img for img in self.images if img.get('status') not in ['annotated', 'completed']]
            else:
                filtered_images = self.images
            
            if not filtered_images:
                QMessageBox.warning(self, "提示", f"没有符合条件的图片\n总图片数: {len(self.images)}\n未标注图片数: {len([img for img in self.images if img.get('status') not in ['annotated', 'completed']])}")
                return
            
            # 开始批量处理
            self.batch_labeling_manager.start_batch_processing(
                model_path, filtered_images, conf_threshold, iou_threshold, class_mapping, self.model_manager
            )
            
            # 批量处理是异步的，通过信号处理完成事件
        except Exception as e:
            QMessageBox.critical(self, "错误", f"批量推理失败: {str(e)}")
        finally:
            # 隐藏加载动画
            self.hide_loading_animation()
    
    def on_single_inference_requested(self, model_path, conf_threshold, iou_threshold, class_mapping, image_path):
        """单张推理请求回调"""
        from core.auto_labeler import AutoLabeler
        
        try:
            labeler = AutoLabeler(model_path, self.model_manager)
            annotations = labeler.process_image(image_path, conf_threshold, iou_threshold, class_mapping)
            
            # 获取覆盖标签设置
            overwrite_labels = False
            if hasattr(self, 'auto_label_settings'):
                overwrite_labels = self.auto_label_settings.get('overwrite_labels', False)
            
            # 更新当前图像的标注
            if annotations and self.current_image_id:
                # 保存标注到数据库
                # 如果需要覆盖原标签，先删除所有原标注
                if overwrite_labels:
                    db.delete_image_annotations(self.current_image_id)
                
                for annotation in annotations:
                    # 获取类别名称
                    class_id = annotation['class_id']
                    
                    # 检查类别ID是否在项目类别范围内
                    existing_class = next((cls for cls in self.classes if cls['id'] == class_id), None)
                    if existing_class:
                        class_name = existing_class['name']
                    else:
                        # 创建新类别
                        class_name = f"class_{class_id}"
                        # 生成随机颜色
                        import random
                        color = f"#{random.randint(0, 0xFFFFFF):06x}"
                        new_class = {
                            'id': class_id,
                            'name': class_name,
                            'color': color
                        }
                        self.classes.append(new_class)
                        # 更新项目类别
                        db.update_project(self.current_project_id, classes=self.classes)
                    
                    # 保存标注
                    db.add_annotation(
                        self.current_image_id,
                        self.current_project_id,
                        class_id,
                        class_name,
                        annotation['type'],
                        annotation['data']
                    )
                
                # 重新加载当前图像的标注
                self.load_current_image_annotations()
                
                QMessageBox.information(self, "成功", "自动标注完成！")
            else:
                # 如果没有检测到目标，但需要覆盖原标签，也删除原标注
                if overwrite_labels and self.current_image_id:
                    db.delete_image_annotations(self.current_image_id)
                    self.load_current_image_annotations()
                QMessageBox.information(self, "提示", "未检测到目标")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"自动标注失败: {str(e)}")
    
    def on_batch_inference_requested(self, model_path, conf_threshold, iou_threshold, class_mapping, images, only_unlabeled):
        """批量推理请求回调"""
        # 过滤图像（如果只处理未标注的）
        if only_unlabeled:
            filtered_images = [img for img in images if img.get('status') != 'annotated']
        else:
            filtered_images = images
        
        if not filtered_images:
            QMessageBox.warning(self, "提示", "没有符合条件的图片")
            return
        
        # 开始批量处理
        self.batch_labeling_manager.start_batch_processing(
            model_path, filtered_images, conf_threshold, iou_threshold, class_mapping, self.model_manager
        )
    
    def on_batch_inference_progress(self, progress, current, total, image_name):
        """批量推理进度回调"""
        # 更新状态栏
        self.status_annotation.setText(f"自动标注: {current}/{total}")
        self.status_position.setText(f"当前: {image_name}")
        self.repaint()
    
    def on_batch_inference_completed(self, success, message, processed_count):
        """批量推理完成回调"""
        if success:
            QMessageBox.information(self, "成功", f"批量自动标注完成！\n处理了 {processed_count} 张图片")
        else:
            QMessageBox.critical(self, "错误", f"批量自动标注失败: {message}")
        
        # 重新加载图片列表以更新状态
        self.load_image_list()
        
        # 重置状态栏
        self.update_status_bar()
    
    def load_current_image_annotations(self):
        """加载当前图像的标注"""
        if self.current_image_id:
            self.annotations = db.get_image_annotations(self.current_image_id)
            self.canvas.set_annotations(self.annotations)
            self.update_status_bar()
    
    def update_status_bar(self):
        """更新状态栏"""
        if self.images and self.current_image_id:
            # 找到当前图像的索引
            current_index = next((i for i, img in enumerate(self.images) if img['id'] == self.current_image_id), -1)
            if current_index >= 0:
                self.status_image.setText(f"当前: {current_index + 1}/{len(self.images)}")
        else:
            self.status_image.setText("当前: 0/0")
        
        # 更新标注数量
        self.status_annotation.setText(f"标注: {len(self.annotations)}")
        
        # 更新工具状态
        tool_names = {'rectangle': '矩形', 'polygon': '多边形', 'move': '移动'}
        tool_name = tool_names.get(self.canvas.current_tool, '矩形')
        self.status_tool.setText(f"工具: {tool_name}")
    
    def on_class_selected(self):
        """类别选中事件"""
        current_item = self.class_list.currentItem()
        if current_item:
            self.current_class_id = current_item.data(Qt.ItemDataRole.UserRole)
            # 更新画布当前类别
            self.canvas.current_class_id = self.current_class_id
    
    def filter_images(self, filter_text: str):
        """筛选图片"""
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            image_id = item.data(Qt.ItemDataRole.UserRole)
            
            # 找到对应的图片数据
            image_data = next((img for img in self.images if img['id'] == image_id), None)
            if not image_data:
                continue
            
            status = image_data.get('status', 'pending')
            
            if filter_text == "全部":
                item.setHidden(False)
            elif filter_text == "未标注":
                item.setHidden(status != 'pending')
            elif filter_text == "已标注":
                item.setHidden(status == 'pending')
    
    def on_image_selected(self, item: QListWidgetItem):
        """图片选中事件"""
        image_id = item.data(Qt.ItemDataRole.UserRole)
        self.load_image(image_id)
    
    def on_task_changed(self, task_type):
        """任务类型切换事件"""
        if self.current_project_id:
            # 更新项目的任务类型
            db.update_project(self.current_project_id, type=task_type)
            # 重新加载当前图片的标注
            if self.current_image_id:
                self.load_annotations()
    
    def load_image(self, image_id: int):
        """加载图片"""
        self.current_image_id = image_id
        
        # 找到图片数据
        image_data = next((img for img in self.images if img['id'] == image_id), None)
        if not image_data:
            return
        
        self.current_image_data = image_data
        
        # 加载到画布
        if image_data.get('storage_path'):
            self.canvas.load_image(image_data['storage_path'])
        
        # 加载标注
        self.load_annotations()
        
        # 更新状态栏
        self.update_status_bar()
        
        # 高亮当前项并滚动到该项
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == image_id:
                item.setSelected(True)
                # 滚动到当前项
                self.image_list.scrollToItem(item)
            else:
                item.setSelected(False)
    
    def load_annotations(self):
        """加载标注"""
        if not self.current_image_id:
            return
        
        self.annotations = db.get_image_annotations(self.current_image_id)
        self.canvas.set_annotations(self.annotations)
        self.update_status_bar()
    
    def set_tool(self, tool: str):
        """设置工具"""
        self.canvas.set_tool(tool)
        
        tool_names = {
            'rectangle': '矩形',
            'polygon': '多边形',
            'move': '移动'
        }
        self.status_tool.setText(f"工具: {tool_names.get(tool, tool)}")
    
    def on_annotation_created(self, annotation: dict):
        """标注创建事件"""
        if not self.current_image_id or not self.current_project_id:
            return
        
        # 从标注数据中获取类别ID（由画布传递）
        class_id = annotation.get('class_id', self.current_class_id)
        class_name = self.classes[class_id]['name'] if class_id < len(self.classes) else 'unknown'
        
        # 保存到数据库
        ann_id = db.add_annotation(
            image_id=self.current_image_id,
            project_id=self.current_project_id,
            class_id=class_id,
            class_name=class_name,
            annotation_type=annotation['type'],
            data=annotation['data']
        )
        
        # 添加到历史记录
        self.add_history('create', {'annotation_id': ann_id})
        
        # 更新图片状态为已标注
        db.update_image_status(self.current_image_id, 'annotated')
        
        # 重新加载标注
        self.load_annotations()
        
        # 更新图片列表显示
        self.update_image_list_display()
    
    def on_annotation_selected(self, annotation_id: int):
        """标注选中事件"""
        # 更新属性面板
        annotation = next((ann for ann in self.annotations if ann['id'] == annotation_id), None)
        if annotation:
            self.update_attribute_panel(annotation)
    
    def on_annotation_modified(self, annotation_id: int, data: dict):
        """标注修改事件（拖动或调整大小后）"""
        annotation = next((ann for ann in self.annotations if ann['id'] == annotation_id), None)
        if annotation:
            # 更新数据库中的标注
            db.update_annotation(annotation_id, data=data)
            
            # 更新属性面板
            self.update_attribute_panel(annotation)
            
            # 添加到历史记录
            self.add_history('modify', {'annotation_id': annotation_id, 'old_data': data.copy()})
    
    def on_annotation_deleted(self, annotation_id: int):
        """标注删除事件"""
        # 保存到历史记录
        annotation = next((ann for ann in self.annotations if ann['id'] == annotation_id), None)
        if annotation:
            self.add_history('delete', annotation)
        
        # 从数据库删除
        db.delete_annotation(annotation_id)
        
        # 重新加载
        self.load_annotations()
        self.canvas.selected_annotation_id = None
        self.clear_attribute_panel()
        
        # 检查图片是否还有标注
        remaining_annotations = db.get_image_annotations(self.current_image_id)
        if not remaining_annotations:
            # 如果没有标注了，更新状态为未标注
            db.update_image_status(self.current_image_id, 'pending')
            # 更新图片列表显示
            self.update_image_list_display()
    
    def delete_selected_annotation(self):
        """删除选中的标注"""
        if self.canvas.selected_annotation_id is not None:
            self.on_annotation_deleted(self.canvas.selected_annotation_id)
    
    def update_attribute_panel(self, annotation: dict):
        """更新属性面板"""
        data = annotation.get('data', {})
        ann_type = annotation.get('type', 'bbox')
        
        # 临时断开信号，避免循环触发
        self.attr_x.blockSignals(True)
        self.attr_y.blockSignals(True)
        self.attr_width.blockSignals(True)
        self.attr_height.blockSignals(True)
        self.attr_class.blockSignals(True)
        
        if ann_type == 'bbox':
            self.attr_x.setValue(int(data.get('x', 0)))
            self.attr_y.setValue(int(data.get('y', 0)))
            self.attr_width.setValue(int(data.get('width', 0)))
            self.attr_height.setValue(int(data.get('height', 0)))
        
        # 设置类别
        class_id = annotation.get('class_id', 0)
        index = self.attr_class.findData(class_id)
        if index >= 0:
            self.attr_class.setCurrentIndex(index)
        
        # 恢复信号
        self.attr_x.blockSignals(False)
        self.attr_y.blockSignals(False)
        self.attr_width.blockSignals(False)
        self.attr_height.blockSignals(False)
        self.attr_class.blockSignals(False)
    
    def clear_attribute_panel(self):
        """清空属性面板"""
        self.attr_x.blockSignals(True)
        self.attr_y.blockSignals(True)
        self.attr_width.blockSignals(True)
        self.attr_height.blockSignals(True)
        self.attr_class.blockSignals(True)
        
        self.attr_x.setValue(0)
        self.attr_y.setValue(0)
        self.attr_width.setValue(0)
        self.attr_height.setValue(0)
        
        self.attr_x.blockSignals(False)
        self.attr_y.blockSignals(False)
        self.attr_width.blockSignals(False)
        self.attr_height.blockSignals(False)
        self.attr_class.blockSignals(False)
    
    def on_attr_value_changed(self):
        """属性值改变事件 - 实时更新画布"""
        if self.canvas.selected_annotation_id is None:
            return
        
        annotation = next((ann for ann in self.annotations if ann['id'] == self.canvas.selected_annotation_id), None)
        if not annotation:
            return
        
        # 更新标注数据
        data = annotation['data']
        data['x'] = self.attr_x.value()
        data['y'] = self.attr_y.value()
        data['width'] = self.attr_width.value()
        data['height'] = self.attr_height.value()
        
        # 刷新画布
        self.canvas.update()
    
    def on_attr_class_changed(self, index):
        """属性类别改变事件"""
        if self.canvas.selected_annotation_id is None:
            return
        
        annotation = next((ann for ann in self.annotations if ann['id'] == self.canvas.selected_annotation_id), None)
        if not annotation:
            return
        
        class_id = self.attr_class.currentData()
        if class_id is not None:
            annotation['class_id'] = class_id
            self.canvas.update()
    
    def apply_annotation_changes(self):
        """应用标注修改到数据库"""
        if self.canvas.selected_annotation_id is None:
            return
        
        annotation = next((ann for ann in self.annotations if ann['id'] == self.canvas.selected_annotation_id), None)
        if not annotation:
            return
        
        # 获取新的类别信息
        class_id = self.attr_class.currentData()
        class_name = self.classes[class_id]['name'] if class_id < len(self.classes) else 'unknown'
        
        # 更新标注数据
        annotation['class_id'] = class_id
        annotation['class_name'] = class_name
        
        # 更新到数据库
        # 先删除旧标注，再添加新标注（简化处理）
        db.delete_annotation(annotation['id'])
        new_id = db.add_annotation(
            image_id=self.current_image_id,
            project_id=self.current_project_id,
            class_id=class_id,
            class_name=class_name,
            annotation_type=annotation['type'],
            data=annotation['data']
        )
        
        # 更新选中状态
        annotation['id'] = new_id
        self.canvas.selected_annotation_id = new_id
        
        # 刷新显示
        self.load_annotations()
        
        QMessageBox.information(self, "成功", "标注修改已保存")
    
    def show_class_context_menu(self, position):
        """显示类别右键菜单"""
        item = self.class_list.itemAt(position)
        if not item:
            return
        
        menu = QMenu(self)
        
        edit_action = menu.addAction("编辑")
        delete_action = menu.addAction("删除")
        
        action = menu.exec(self.class_list.mapToGlobal(position))
        
        if action == edit_action:
            self.edit_class(item)
        elif action == delete_action:
            self.delete_class(item)
    
    def edit_class(self, item: QListWidgetItem):
        """编辑类别"""
        class_id = item.data(Qt.ItemDataRole.UserRole)
        class_info = next((c for c in self.classes if c['id'] == class_id), None)
        if not class_info:
            return
        
        # 编辑名称
        name, ok = QInputDialog.getText(
            self, "编辑类别", 
            "请输入类别名称:",
            text=class_info['name']
        )
        if not ok or not name:
            return
        
        # 编辑颜色
        color = QColorDialog.getColor(
            QColor(class_info.get('color', '#FF0000')), 
            self, "选择类别颜色"
        )
        if not color.isValid():
            color = QColor(class_info.get('color', '#FF0000'))
        
        # 更新类别信息
        class_info['name'] = name
        class_info['color'] = color.name()
        
        # 保存到数据库
        db.update_project(self.current_project_id, classes=self.classes)
        
        # 更新显示
        self.update_class_list()
    
    def delete_class(self, item: QListWidgetItem):
        """删除类别"""
        class_id = item.data(Qt.ItemDataRole.UserRole)
        class_info = next((c for c in self.classes if c['id'] == class_id), None)
        if not class_info:
            return
        
        # 确认删除
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除类别 '{class_info['name']}' 吗？\n该类别下的所有标注将被删除！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # 删除该类别下的所有标注
        for annotation in self.annotations:
            if annotation.get('class_id') == class_id:
                db.delete_annotation(annotation['id'])
        
        # 从列表中删除类别
        self.classes = [c for c in self.classes if c['id'] != class_id]
        
        # 重新编号
        for i, cls in enumerate(self.classes):
            cls['id'] = i
        
        # 保存到数据库
        db.update_project(self.current_project_id, classes=self.classes)
        
        # 更新显示
        self.update_class_list()
        self.load_annotations()
    
    def add_class(self):
        """添加类别"""
        name, ok = QInputDialog.getText(self, "添加类别", "请输入类别名称:")
        if ok and name:
            # 选择颜色
            color = QColorDialog.getColor(QColor(255, 0, 0), self, "选择类别颜色")
            if not color.isValid():
                color = QColor(255, 0, 0)
            
            class_id = len(self.classes)
            self.classes.append({
                'id': class_id,
                'name': name,
                'color': color.name()
            })
            
            # 更新项目类别
            db.update_project(self.current_project_id, classes=self.classes)
            
            self.update_class_list()
            # 选中新添加的类别
            self.class_list.setCurrentRow(self.class_list.count() - 1)
            self.on_class_selected()
    
    def prev_image(self):
        """上一张图片"""
        if not self.images or not self.current_image_id:
            return
        
        current_index = next((i for i, img in enumerate(self.images) if img['id'] == self.current_image_id), 0)
        if current_index > 0:
            new_image_id = self.images[current_index - 1]['id']
            self.load_image(new_image_id)
    
    def next_image(self):
        """下一张图片"""
        if not self.images or not self.current_image_id:
            return
        
        current_index = next((i for i, img in enumerate(self.images) if img['id'] == self.current_image_id), -1)
        if current_index < len(self.images) - 1:
            new_image_id = self.images[current_index + 1]['id']
            self.load_image(new_image_id)
    
    def add_history(self, action: str, data: dict):
        """添加历史记录"""
        # 删除当前位置之后的历史
        self.history = self.history[:self.history_index + 1]
        
        # 添加新记录
        self.history.append({
            'action': action,
            'data': data
        })
        self.history_index = len(self.history) - 1
        
        # 限制历史记录数量
        if len(self.history) > 50:
            self.history.pop(0)
            self.history_index -= 1
    
    def undo(self):
        """撤销"""
        if self.history_index < 0:
            return
        
        record = self.history[self.history_index]
        action = record['action']
        data = record['data']
        
        if action == 'create':
            # 撤销创建 = 删除
            ann_id = data.get('annotation_id')
            if ann_id:
                db.delete_annotation(ann_id)
        elif action == 'delete':
            # 撤销删除 = 创建
            db.add_annotation(
                image_id=data['image_id'],
                project_id=data['project_id'],
                class_id=data['class_id'],
                class_name=data['class_name'],
                annotation_type=data['type'],
                data=data['data']
            )
        
        self.history_index -= 1
        self.load_annotations()
    
    def update_status_bar(self):
        """更新状态栏"""
        total = len(self.images)
        current = 0
        if self.current_image_id:
            current = next((i for i, img in enumerate(self.images) if img['id'] == self.current_image_id), 0) + 1
        
        self.status_image.setText(f"当前: {current}/{total}")
        self.status_annotation.setText(f"标注: {len(self.annotations)}")
    
    def keyPressEvent(self, event: QKeyEvent):
        """键盘事件"""
        from PyQt6.QtCore import QSettings
        
        # 获取快捷键设置
        settings = QSettings("EzYOLO", "Settings")
        rect_tool_key = settings.value("rect_tool_shortcut", "W").upper()
        poly_tool_key = settings.value("poly_tool_shortcut", "P").upper()
        move_tool_key = settings.value("move_tool_shortcut", "V").upper()
        prev_image_key = settings.value("prev_image_shortcut", "A").upper()
        next_image_key = settings.value("next_image_shortcut", "D").upper()
        delete_key = settings.value("delete_shortcut", "DELETE").upper()
        
        # 处理工具快捷键
        key_text = event.text().upper()
        if key_text == rect_tool_key:
            self.btn_rectangle.setChecked(True)
            self.set_tool('rectangle')
            return
        elif key_text == poly_tool_key:
            self.btn_polygon.setChecked(True)
            self.set_tool('polygon')
            return
        elif key_text == move_tool_key:
            self.btn_move.setChecked(True)
            self.set_tool('move')
            return
        elif key_text == prev_image_key:
            self.prev_image()
            return
        elif key_text == next_image_key:
            self.next_image()
            return
        elif key_text == delete_key:
            self.delete_selected_annotation()
        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Z:
            self.undo()
        else:
            # 处理数字键1-9切换标签
            if key_text in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                class_index = int(key_text) - 1
                if class_index < self.class_list.count():
                    self.class_list.setCurrentRow(class_index)
                    self.on_class_selected()
                    return
            super().keyPressEvent(event)
    
    def export_annotations(self):
        """导出标注文件"""
        import shutil
        from PyQt6.QtWidgets import QMessageBox, QFileDialog
        
        if not self.current_project_id:
            QMessageBox.warning(self, "导出失败", "请先选择一个项目")
            return
        
        # 选择导出目录
        export_dir = QFileDialog.getExistingDirectory(
            self, "选择导出目录", 
            os.path.expanduser("~")
        )
        
        if not export_dir:
            return
        
        try:
            # 获取导出格式
            export_format = self.export_format.currentText()
            
            # 获取项目信息
            project = db.get_project(self.current_project_id)
            if not project:
                QMessageBox.warning(self, "导出失败", "无法获取项目信息")
                return
            
            project_name = project.get('name', 'untitled')
            
            # 创建导出目录结构
            export_path = os.path.join(export_dir, f"{project_name}_annotations")
            os.makedirs(export_path, exist_ok=True)
            
            # 获取项目图片
            images = db.get_project_images(self.current_project_id)
            if not images:
                QMessageBox.warning(self, "导出失败", "项目中没有图片")
                return
            
            # 获取类别映射
            class_mapping = {cls['id']: cls['name'] for cls in self.classes}
            
            if export_format == "YOLO格式":
                # 创建labels目录
                labels_dir = os.path.join(export_path, 'labels')
                os.makedirs(labels_dir, exist_ok=True)
                
                # 导出每个图片的标注
                exported_count = 0
                for image in images:
                    image_id = image['id']
                    annotations = db.get_image_annotations(image_id)
                    
                    if annotations:
                        # 创建标注文件
                        filename = os.path.splitext(image['filename'])[0] + '.txt'
                        label_file = os.path.join(labels_dir, filename)
                        
                        with open(label_file, 'w', encoding='utf-8') as f:
                            for ann in annotations:
                                class_id = ann.get('class_id', 0)
                                ann_type = ann.get('type', 'bbox')
                                data = ann.get('data', {})
                                
                                if ann_type == 'bbox':
                                    # YOLO格式：class_id x_center y_center width height
                                    x = data.get('x', 0)
                                    y = data.get('y', 0)
                                    width = data.get('width', 0)
                                    height = data.get('height', 0)
                                    
                                    # 计算中心点和归一化
                                    img_width = image.get('width', 1920)  # 默认宽度
                                    img_height = image.get('height', 1080)  # 默认高度
                                    
                                    x_center = (x + width/2) / img_width
                                    y_center = (y + height/2) / img_height
                                    norm_width = width / img_width
                                    norm_height = height / img_height
                                    
                                    # 写入文件
                                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                        
                        exported_count += 1
                
                # 创建classes.txt文件
                classes_file = os.path.join(export_path, 'classes.txt')
                with open(classes_file, 'w', encoding='utf-8') as f:
                    for cls in sorted(self.classes, key=lambda x: x['id']):
                        f.write(f"{cls['name']}\n")
                
                QMessageBox.information(self, "导出成功", f"已导出 {exported_count} 个标注文件到\n{export_path}")
                
            elif export_format == "COCO格式":
                # 导出COCO格式
                import json
                
                # 创建COCO格式的标注数据
                coco_data = {
                    "info": {
                        "description": f"Annotations for {project_name}",
                        "version": "1.0",
                        "year": 2024
                    },
                    "licenses": [],
                    "images": [],
                    "annotations": [],
                    "categories": []
                }
                
                # 添加类别
                for cls in self.classes:
                    coco_data["categories"].append({
                        "id": cls['id'],
                        "name": cls['name'],
                        "supercategory": "object"
                    })
                
                # 添加图片和标注
                annotation_id = 1
                for image in images:
                    # 添加图片信息
                    image_info = {
                        "id": image['id'],
                        "file_name": image['filename'],
                        "width": image.get('width', 1920),
                        "height": image.get('height', 1080),
                        "date_captured": "",
                        "license": 0,
                        "coco_url": "",
                        "flickr_url": ""
                    }
                    coco_data["images"].append(image_info)
                    
                    # 添加标注
                    annotations = db.get_image_annotations(image['id'])
                    for ann in annotations:
                        ann_type = ann.get('type', 'bbox')
                        data = ann.get('data', {})
                        
                        if ann_type == 'bbox':
                            # COCO格式：x y width height
                            x = int(data.get('x', 0))
                            y = int(data.get('y', 0))
                            width = int(data.get('width', 0))
                            height = int(data.get('height', 0))
                            
                            coco_annotation = {
                                "id": annotation_id,
                                "image_id": image['id'],
                                "category_id": ann.get('class_id', 0),
                                "segmentation": [],
                                "area": width * height,
                                "bbox": [x, y, width, height],
                                "iscrowd": 0
                            }
                            coco_data["annotations"].append(coco_annotation)
                            annotation_id += 1
                
                # 保存COCO格式文件
                coco_file = os.path.join(export_path, 'annotations.json')
                with open(coco_file, 'w', encoding='utf-8') as f:
                    json.dump(coco_data, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "导出成功", f"已导出 COCO 格式标注到\n{coco_file}")
                
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出过程中出错:\n{str(e)}")
    
    def export_dataset(self):
        """导出完整数据集"""
        import shutil
        from PyQt6.QtWidgets import QMessageBox, QFileDialog
        
        if not self.current_project_id:
            QMessageBox.warning(self, "导出失败", "请先选择一个项目")
            return
        
        # 选择导出目录
        export_dir = QFileDialog.getExistingDirectory(
            self, "选择导出目录", 
            os.path.expanduser("~")
        )
        
        if not export_dir:
            return
        
        try:
            # 获取导出格式
            export_format = self.export_format.currentText()
            
            # 获取项目信息
            project = db.get_project(self.current_project_id)
            if not project:
                QMessageBox.warning(self, "导出失败", "无法获取项目信息")
                return
            
            project_name = project.get('name', 'untitled')
            
            # 创建导出目录结构
            dataset_dir = os.path.join(export_dir, project_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # 创建images和labels目录
            images_dir = os.path.join(dataset_dir, 'images')
            labels_dir = os.path.join(dataset_dir, 'labels')
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            
            # 获取项目图片
            images = db.get_project_images(self.current_project_id)
            if not images:
                QMessageBox.warning(self, "导出失败", "项目中没有图片")
                return
            
            # 获取类别映射
            class_mapping = {cls['id']: cls['name'] for cls in self.classes}
            
            # 复制图片并导出标注
            copied_count = 0
            exported_count = 0
            
            for image in images:
                # 复制图片
                src_image = image.get('storage_path', '')
                if src_image and os.path.exists(src_image):
                    dst_image = os.path.join(images_dir, image['filename'])
                    shutil.copy2(src_image, dst_image)
                    copied_count += 1
                
                # 导出标注
                image_id = image['id']
                annotations = db.get_image_annotations(image_id)
                
                if annotations:
                    # 创建标注文件
                    filename = os.path.splitext(image['filename'])[0] + '.txt'
                    label_file = os.path.join(labels_dir, filename)
                    
                    with open(label_file, 'w', encoding='utf-8') as f:
                        for ann in annotations:
                            class_id = ann.get('class_id', 0)
                            ann_type = ann.get('type', 'bbox')
                            data = ann.get('data', {})
                            
                            if ann_type == 'bbox':
                                # YOLO格式：class_id x_center y_center width height
                                x = data.get('x', 0)
                                y = data.get('y', 0)
                                width = data.get('width', 0)
                                height = data.get('height', 0)
                                
                                # 计算中心点和归一化
                                img_width = image.get('width', 1920)  # 默认宽度
                                img_height = image.get('height', 1080)  # 默认高度
                                
                                x_center = (x + width/2) / img_width
                                y_center = (y + height/2) / img_height
                                norm_width = width / img_width
                                norm_height = height / img_height
                                
                                # 写入文件
                                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                    
                    exported_count += 1
            
            # 创建classes.txt文件
            classes_file = os.path.join(dataset_dir, 'classes.txt')
            with open(classes_file, 'w', encoding='utf-8') as f:
                for cls in sorted(self.classes, key=lambda x: x['id']):
                    f.write(f"{cls['name']}\n")
            
            # 创建data.yaml文件（YOLO格式）
            yaml_file = os.path.join(dataset_dir, 'data.yaml')
            yaml_content = f"""
train: images
test: images
val: images

nc: {len(self.classes)}
names: {[cls['name'] for cls in sorted(self.classes, key=lambda x: x['id'])]}
"""
            
            with open(yaml_file, 'w', encoding='utf-8') as f:
                f.write(yaml_content)
            
            QMessageBox.information(self, "导出成功", f"已导出完整数据集到\n{dataset_dir}\n\n" 
                                   f"- 复制图片: {copied_count} 张\n" 
                                   f"- 导出标注: {exported_count} 个")
            
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出过程中出错:\n{str(e)}")
