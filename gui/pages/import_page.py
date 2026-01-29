# -*- coding: utf-8 -*-
"""
导入页面
支持图像文件夹批量导入、视频抽帧、已有标注导入
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QGridLayout, QFrame, QFileDialog, QProgressBar,
    QMenu, QMessageBox, QComboBox, QLineEdit, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QIcon
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import os

from gui.styles import COLORS
from models.database import db
from core.import_manager import ImportManager
from core.annotation_importer import AnnotationImporter
from gui.widgets.loading_dialog import LoadingOverlay


# 视频导入线程
class VideoImportThread(QThread):
    """视频导入后台线程"""
    progress_updated = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, int, int)
    
    def __init__(self, video_path, project_id, frame_interval):
        super().__init__()
        self.video_path = video_path
        self.project_id = project_id
        self.frame_interval = frame_interval
    
    def run(self):
        """运行视频导入"""
        try:
            from core.import_manager import ImportManager
            import_manager = ImportManager(self.project_id)
            
            def progress_callback(progress, message):
                self.progress_updated.emit(progress, message)
            
            imported, skipped = import_manager.import_video(
                self.video_path,
                frame_interval=self.frame_interval,
                progress_callback=progress_callback
            )
            
            self.finished.emit(True, "视频导入完成", imported, skipped)
        except Exception as e:
            self.finished.emit(False, f"导入失败: {str(e)}", 0, 0)


class ImageLoadWorker(QThread):
    """图片加载工作线程"""
    
    # 信号：进度更新、单个图片加载完成、全部完成
    progress = pyqtSignal(int, int)  # 当前进度, 总数
    image_loaded = pyqtSignal(int, object, str)  # 索引, 缩略图, 存储路径
    finished_loading = pyqtSignal()
    
    def __init__(self, images: List[Dict]):
        super().__init__()
        self.images = images
        self._is_running = True
    
    def run(self):
        """在后台线程中加载图片"""
        total = len(self.images)
        
        for i, image_data in enumerate(self.images):
            if not self._is_running:
                break
            
            storage_path = image_data.get('storage_path', '')
            pixmap = None
            
            if storage_path and os.path.exists(storage_path):
                try:
                    # 使用OpenCV加载，比QPixmap更快
                    img = cv2.imread(storage_path)
                    if img is not None:
                        # 直接缩小到缩略图尺寸，减少内存占用
                        img = cv2.resize(img, (160, 160))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        h, w, ch = img.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(qt_image)
                except Exception:
                    pass
            
            # 如果加载失败，创建空白图
            if pixmap is None or pixmap.isNull():
                pixmap = QPixmap(160, 160)
                pixmap.fill(QColor(COLORS['sidebar']))
            
            # 发送信号到主线程更新UI
            self.image_loaded.emit(i, pixmap, storage_path)
            self.progress.emit(i + 1, total)
            
            # 每加载10张图片休眠一下，让UI有机会更新
            if i % 10 == 0:
                self.msleep(1)
        
        self.finished_loading.emit()
    
    def stop(self):
        """停止加载"""
        self._is_running = False


class ImportPage(QWidget):
    """导入页面"""
    
    def __init__(self):
        super().__init__()
        self.current_project_id = None
        self.images = []
        self.thumbnail_cache = {}
        self.load_worker = None
        self.thumbnail_widgets = []  # 存储缩略图控件引用
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        
        # 标题区域
        title_layout = QHBoxLayout()
        
        title = QLabel("数据导入")
        title.setObjectName("title")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        title_layout.addWidget(title)
        
        title_layout.addStretch()
        
        # 项目选择
        project_label = QLabel("当前项目:")
        project_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        title_layout.addWidget(project_label)
        
        self.project_combo = QComboBox()
        self.project_combo.setFixedWidth(200)
        self.project_combo.addItem("请选择项目...")
        self.project_combo.currentIndexChanged.connect(self.on_project_changed)
        title_layout.addWidget(self.project_combo)
        
        # 新建项目按钮
        new_project_btn = QPushButton("+ 新建项目")
        new_project_btn.setObjectName("secondary")
        new_project_btn.clicked.connect(self.create_new_project)
        title_layout.addWidget(new_project_btn)
        
        main_layout.addLayout(title_layout)
        
        # 说明文字
        desc = QLabel("请导入需要标注和训练的图像数据，支持批量导入文件夹、单张图片、视频抽帧以及已有标注导入")
        desc.setObjectName("subtitle")
        desc.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        desc.setWordWrap(True)
        main_layout.addWidget(desc)
        
        # 工具栏
        toolbar = self.create_toolbar()
        main_layout.addWidget(toolbar)
        
        # 进度条（默认隐藏）
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
        main_layout.addWidget(self.progress_bar)
        
        # 图像列表区域 - 使用QListWidget代替自定义网格，性能更好
        self.image_list = QListWidget()
        self.image_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.image_list.setIconSize(QSize(160, 160))
        self.image_list.setSpacing(16)
        self.image_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.image_list.setMovement(QListWidget.Movement.Static)
        self.image_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.image_list.setUniformItemSizes(True)  # 统一项目大小，优化布局
        self.image_list.setGridSize(QSize(180, 200))  # 设置固定网格大小
        self.image_list.itemClicked.connect(self.on_image_clicked)
        self.image_list.setStyleSheet('''
            QListWidget {
                background-color: #1E1E1E;
                border: none;
            }
            QListWidget::item {
                background-color: #2D2D30;
                border: 2px solid #3e3e42;
                border-radius: 8px;
                padding: 8px;
            }
            QListWidget::item:selected {
                border: 2px solid #007ACC;
            }
        ''')
        main_layout.addWidget(self.image_list)
        
        # 状态栏
        self.status_bar = self.create_status_bar()
        main_layout.addWidget(self.status_bar)
        
        # 加载项目列表
        self.load_projects()
    
    def create_toolbar(self) -> QFrame:
        """创建工具栏"""
        toolbar = QFrame()
        toolbar.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['panel']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
            }}
        """)
        
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # 导入文件夹按钮
        self.btn_import_folder = QPushButton("📁 导入文件夹")
        self.btn_import_folder.setToolTip("批量导入整个文件夹中的图像")
        self.btn_import_folder.clicked.connect(self.import_folder)
        layout.addWidget(self.btn_import_folder)
        
        # 导入图片按钮
        self.btn_import_images = QPushButton("🖼 导入图片")
        self.btn_import_images.setToolTip("选择单张或多张图片导入")
        self.btn_import_images.clicked.connect(self.import_images)
        layout.addWidget(self.btn_import_images)
        
        # 导入视频按钮
        self.btn_import_video = QPushButton("🎬 导入视频")
        self.btn_import_video.setToolTip("从视频中抽取帧导入")
        self.btn_import_video.clicked.connect(self.import_video)
        layout.addWidget(self.btn_import_video)
        
        # 导入标注按钮
        self.btn_import_annotations = QPushButton("📋 导入标注")
        self.btn_import_annotations.setToolTip("导入已有的标注文件（YOLO/COCO/VOC格式）")
        self.btn_import_annotations.clicked.connect(self.import_annotations)
        layout.addWidget(self.btn_import_annotations)
        
        layout.addStretch()
        
        # 视图切换按钮
        self.view_combo = QComboBox()
        self.view_combo.addItems(["全部", "未标注", "已标注"])
        self.view_combo.setFixedWidth(100)
        self.view_combo.currentTextChanged.connect(self.filter_images)
        layout.addWidget(QLabel("筛选:"))
        layout.addWidget(self.view_combo)
        
        # 删除选中按钮
        self.btn_delete_selected = QPushButton("🗑 删除选中")
        self.btn_delete_selected.setObjectName("secondary")
        self.btn_delete_selected.clicked.connect(self.delete_selected_images)
        layout.addWidget(self.btn_delete_selected)
        
        # 删除全部按钮
        self.btn_clear = QPushButton("🗑 清空")
        self.btn_clear.setObjectName("secondary")
        self.btn_clear.clicked.connect(self.clear_all_images)
        layout.addWidget(self.btn_clear)
        
        return toolbar
    
    def create_status_bar(self) -> QFrame:
        """创建状态栏"""
        status_bar = QFrame()
        status_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['panel']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
            QLabel {{
                color: {COLORS['text_secondary']};
                font-size: 12px;
                padding: 4px 12px;
            }}
        """)
        
        layout = QHBoxLayout(status_bar)
        layout.setContentsMargins(12, 8, 12, 8)
        
        self.status_total = QLabel("共 0 张图片")
        layout.addWidget(self.status_total)
        
        layout.addWidget(QLabel("|"))
        
        self.status_annotated = QLabel("已标注: 0")
        self.status_annotated.setStyleSheet(f"color: {COLORS['success']};")
        layout.addWidget(self.status_annotated)
        
        layout.addWidget(QLabel("|"))
        
        self.status_pending = QLabel("未标注: 0")
        layout.addWidget(self.status_pending)
        
        layout.addStretch()
        
        return status_bar
    
    def load_projects(self):
        """加载项目列表"""
        self.project_combo.clear()
        self.project_combo.addItem("请选择项目...", None)
        
        projects = db.get_all_projects()
        for project in projects:
            self.project_combo.addItem(project['name'], project['id'])
    
    def on_project_changed(self, index):
        """项目选择改变"""
        project_id = self.project_combo.currentData()
        if project_id:
            self.current_project_id = project_id
            self.load_project_images()
        else:
            self.current_project_id = None
            self.image_list.clear()
            self.images = []
            self.update_status_bar()
    
    def create_new_project(self):
        """创建新项目"""
        from PyQt6.QtWidgets import QInputDialog
        
        name, ok = QInputDialog.getText(self, "新建项目", "请输入项目名称:")
        if ok and name:
            project_id = db.create_project(
                name=name,
                description="",
                project_type="detection",
                classes=[]
            )
            self.load_projects()
            index = self.project_combo.findData(project_id)
            if index >= 0:
                self.project_combo.setCurrentIndex(index)
    
    def load_project_images(self):
        """加载项目图像 - 使用多线程"""
        if not self.current_project_id:
            return
        
        # 停止之前的加载
        if self.load_worker and self.load_worker.isRunning():
            self.load_worker.stop()
            self.load_worker.wait()
        
        # 清空列表
        self.image_list.clear()
        self.thumbnail_widgets.clear()
        self.thumbnail_cache.clear()
        
        # 从数据库获取图片列表（很快）
        self.images = db.get_project_images(self.current_project_id)
        self.update_status_bar()
        
        if not self.images:
            return
        
        # 先创建所有列表项（显示占位符）
        for image_data in self.images:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, image_data['id'])
            item.setText(image_data['filename'])
            item.setToolTip(f"{image_data['filename']}\n{image_data.get('width', 0)}x{image_data.get('height', 0)}")
            
            # 设置状态标记
            status = image_data.get('status', 'pending')
            if status == 'annotated':
                item.setBackground(QColor(COLORS['success']))
            
            # 设置项目大小提示，确保即使没有图标也有足够高度
            item.setSizeHint(QSize(180, 200))
            
            self.image_list.addItem(item)
        
        # 显示加载遮罩
        self.loading_overlay = LoadingOverlay(self, f"正在加载 {len(self.images)} 张图片...")
        self.loading_overlay.show_loading()
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.images))
        self.progress_bar.setValue(0)
        
        # 启动后台加载线程
        self.load_worker = ImageLoadWorker(self.images)
        self.load_worker.image_loaded.connect(self.on_image_loaded)
        self.load_worker.progress.connect(self.on_load_progress)
        self.load_worker.finished_loading.connect(self.on_load_finished)
        self.load_worker.start()
    
    def on_image_loaded(self, index: int, pixmap: QPixmap, storage_path: str):
        """单个图片加载完成回调（在主线程执行）"""
        if index < self.image_list.count():
            item = self.image_list.item(index)
            if item:
                # 设置图标
                icon = QIcon(pixmap)
                item.setIcon(icon)
                # 缓存
                self.thumbnail_cache[storage_path] = pixmap
    
    def on_load_progress(self, current: int, total: int):
        """加载进度回调"""
        self.progress_bar.setValue(current)
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.label.setText(f"正在加载... {current}/{total}")
    
    def on_load_finished(self):
        """加载完成回调"""
        self.progress_bar.setVisible(False)
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.hide_loading()
            self.loading_overlay.deleteLater()
            delattr(self, 'loading_overlay')
    
    def filter_images(self, filter_text: str):
        """筛选图像"""
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            image_id = item.data(Qt.ItemDataRole.UserRole)
            
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
    
    def on_image_clicked(self, item: QListWidgetItem):
        """图像点击事件"""
        image_id = item.data(Qt.ItemDataRole.UserRole)
        # TODO: 实现图像预览或编辑
        pass
    
    def update_status_bar(self):
        """更新状态栏"""
        total = len(self.images)
        annotated = sum(1 for img in self.images if img.get('status') == 'annotated')
        pending = total - annotated
        
        self.status_total.setText(f"共 {total} 张图片")
        self.status_annotated.setText(f"已标注: {annotated}")
        self.status_pending.setText(f"未标注: {pending}")
    
    def import_folder(self):
        """导入文件夹"""
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择或创建一个项目")
            return
        
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择图像文件夹", "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder_path:
            self.process_folder_import(folder_path)
    
    def import_images(self):
        """导入单张或多张图片"""
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择或创建一个项目")
            return
        
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;所有文件 (*.*)"
        )
        
        if file_paths:
            self.process_image_import(file_paths)
    
    def import_video(self):
        """导入视频"""
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择或创建一个项目")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )
        
        if file_path:
            self.process_video_import(file_path)
    
    def process_video_import(self, file_path: str):
        """处理视频导入"""
        if not self.current_project_id:
            return
        
        from PyQt6.QtWidgets import QInputDialog
        interval, ok = QInputDialog.getInt(
            self, "抽帧设置",
            "请输入抽帧间隔（每隔多少帧抽取一帧）:",
            value=30, min=1, max=1000
        )
        
        if not ok:
            return
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 创建并启动视频导入线程
        self.video_import_thread = VideoImportThread(
            file_path, 
            self.current_project_id, 
            interval
        )
        
        # 连接信号
        self.video_import_thread.progress_updated.connect(self.update_import_progress)
        self.video_import_thread.finished.connect(self.on_video_import_finished)
        
        # 启动线程
        self.video_import_thread.start()
    
    def import_annotations(self):
        """导入已有标注"""
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择或创建一个项目")
            return
        
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QRadioButton, QPushButton, QLabel
        
        dialog = QDialog(self)
        dialog.setWindowTitle("选择标注格式")
        dialog.setFixedSize(300, 200)
        dialog.setStyleSheet("""
            QDialog {
                background-color: """ + COLORS['background'] + """;
            }
            QLabel {
                color: """ + COLORS['text_primary'] + """;
                font-size: 14px;
            }
            QRadioButton {
                color: """ + COLORS['text_primary'] + """;
                font-size: 14px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid """ + COLORS['border'] + """;
                border-radius: 8px;
                background-color: """ + COLORS['sidebar'] + """;
            }
            QRadioButton::indicator:checked {
                border: 2px solid """ + COLORS['primary'] + """;
                background-color: white;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        
        label = QLabel("请选择要导入的标注格式:")
        layout.addWidget(label)
        
        yolo_radio = QRadioButton("YOLO格式 (txt)")
        yolo_radio.setChecked(True)
        layout.addWidget(yolo_radio)
        
        coco_radio = QRadioButton("COCO格式 (json)")
        layout.addWidget(coco_radio)
        
        voc_radio = QRadioButton("Pascal VOC格式 (xml)")
        layout.addWidget(voc_radio)
        
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.setObjectName("secondary")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        if yolo_radio.isChecked():
            self.import_yolo_annotations()
        elif coco_radio.isChecked():
            self.import_coco_annotations()
        elif voc_radio.isChecked():
            self.import_voc_annotations()
    
    def import_yolo_annotations(self):
        """导入YOLO标注"""
        labels_dir = QFileDialog.getExistingDirectory(
            self, "选择YOLO标签文件夹 (labels)", "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not labels_dir:
            return
        
        reply = QMessageBox.question(
            self, "选择图像文件夹",
            "是否需要选择对应的图像文件夹？\n（如果标签文件和图像文件在同一目录，可选择否）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        images_dir = None
        if reply == QMessageBox.StandardButton.Yes:
            images_dir = QFileDialog.getExistingDirectory(
                self, "选择图像文件夹 (images)", "",
                QFileDialog.Option.ShowDirsOnly
            )
        
        try:
            importer = AnnotationImporter(self.current_project_id)
            imported, skipped = importer.import_yolo_annotations(labels_dir, images_dir)
            self.load_project_images()
            QMessageBox.information(
                self, "导入完成",
                f"YOLO标注导入完成！\n成功导入: {imported} 个标注\n跳过: {skipped} 个"
            )
        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"导入过程中发生错误:\n{str(e)}")
    
    def import_coco_annotations(self):
        """导入COCO标注"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择COCO标注文件", "",
            "JSON文件 (*.json);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            importer = AnnotationImporter(self.current_project_id)
            imported, skipped = importer.import_coco_annotations(file_path)
            self.load_project_images()
            QMessageBox.information(
                self, "导入完成",
                f"COCO标注导入完成！\n成功导入: {imported} 个标注\n跳过: {skipped} 个"
            )
        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"导入过程中发生错误:\n{str(e)}")
    
    def import_voc_annotations(self):
        """导入VOC标注"""
        voc_dir = QFileDialog.getExistingDirectory(
            self, "选择VOC标注文件夹 (Annotations)", "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not voc_dir:
            return
        
        try:
            importer = AnnotationImporter(self.current_project_id)
            imported, skipped = importer.import_voc_annotations(voc_dir)
            self.load_project_images()
            QMessageBox.information(
                self, "导入完成",
                f"VOC标注导入完成！\n成功导入: {imported} 个标注\n跳过: {skipped} 个"
            )
        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"导入过程中发生错误:\n{str(e)}")
    
    def process_folder_import(self, folder_path: str):
        """处理文件夹导入"""
        if not self.current_project_id:
            return
        
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            import_manager = ImportManager(self.current_project_id)
            imported, skipped = import_manager.import_folder(
                folder_path,
                progress_callback=self.update_import_progress
            )
            
            self.load_project_images()
            
            QMessageBox.information(
                self, "导入完成",
                f"文件夹导入完成！\n成功导入: {imported} 张\n跳过: {skipped} 张"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"导入过程中发生错误:\n{str(e)}")
        finally:
            self.progress_bar.setVisible(False)
    
    def process_image_import(self, file_paths: List[str]):
        """处理图像导入"""
        if not self.current_project_id:
            return
        
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            import_manager = ImportManager(self.current_project_id)
            imported, skipped = import_manager.import_images(
                file_paths,
                progress_callback=self.update_import_progress
            )
            
            self.load_project_images()
            
            QMessageBox.information(
                self, "导入完成",
                f"图片导入完成！\n成功导入: {imported} 张\n跳过: {skipped} 张"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"导入过程中发生错误:\n{str(e)}")
        finally:
            self.progress_bar.setVisible(False)
    
    def update_import_progress(self, progress: int, message: str):
        """更新导入进度"""
        self.progress_bar.setValue(progress)
    
    def on_video_import_finished(self, success: bool, message: str, imported: int, skipped: int):
        """视频导入完成回调"""
        # 隐藏进度条
        self.progress_bar.setVisible(False)
        
        if success:
            # 重新加载项目图片
            self.load_project_images()
            
            # 显示成功消息
            QMessageBox.information(
                self, "导入完成",
                f"视频导入完成！\n成功导入: {imported} 帧\n跳过: {skipped} 帧"
            )
        else:
            # 显示错误消息
            QMessageBox.critical(self, "导入失败", message)
    
    def clear_all_images(self):
        """清空所有图像"""
        if not self.images:
            return
        
        reply = QMessageBox.question(
            self, "确认清空",
            f"确定要删除当前项目中的所有 {len(self.images)} 张图片吗？\n此操作不可恢复！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            deleted = 0
            failed = 0
            
            for image in self.images:
                if db.delete_image(image['id']):
                    deleted += 1
                else:
                    failed += 1
            
            self.load_project_images()
            
            if failed == 0:
                QMessageBox.information(self, "清空完成", f"已成功删除 {deleted} 张图片")
            else:
                QMessageBox.warning(self, "清空完成", f"成功删除 {deleted} 张，失败 {failed} 张")
    
    def delete_selected_images(self):
        """删除选中的图片"""
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "请先选择要删除的图片")
            return
        
        count = len(selected_items)
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除选中的 {count} 张图片吗？\n此操作不可恢复！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        deleted = 0
        failed = 0
        
        for item in selected_items:
            image_id = item.data(Qt.ItemDataRole.UserRole)
            if db.delete_image(image_id):
                deleted += 1
            else:
                failed += 1
        
        self.load_project_images()
        
        if failed == 0:
            QMessageBox.information(self, "删除完成", f"已成功删除 {deleted} 张图片")
        else:
            QMessageBox.warning(self, "删除完成", f"成功删除 {deleted} 张，失败 {failed} 张")
