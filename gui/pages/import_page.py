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
from typing import List, Dict, Optional, Tuple
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
    
    def __init__(self, image_tasks: List[Tuple[int, Dict]]):
        super().__init__()
        self.image_tasks = image_tasks
        self._is_running = True
    
    def run(self):
        """在后台线程中加载图片"""
        total = len(self.image_tasks)
        
        for task_index, (row_index, image_data) in enumerate(self.image_tasks):
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
            self.image_loaded.emit(row_index, pixmap, storage_path)
            self.progress.emit(task_index + 1, total)
            
            # 每加载10张图片休眠一下，让UI有机会更新
            if task_index % 10 == 0:
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
        self._image_load_generation = 0
        self.thumbnail_widgets = []  # 存储缩略图控件引用
        self.init_ui()

    def _remove_cached_thumbnails(self, storage_paths):
        """按路径移除缩略图缓存。"""
        for path in storage_paths:
            if path in self.thumbnail_cache:
                del self.thumbnail_cache[path]

    def stop_image_loading(self, reset_progress: bool = True):
        """停止当前缩略图加载线程，并可选清理进度状态。"""
        self._image_load_generation += 1
        if self.load_worker and self.load_worker.isRunning():
            self.load_worker.stop()
            self.load_worker.wait()
        self.load_worker = None
        if reset_progress and hasattr(self, 'progress_bar'):
            self.progress_bar.setVisible(False)
            self.progress_bar.setValue(0)
    
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
        
        # 任务类别显示和设置
        self.task_type_label = QLabel("任务类别: 未设置")
        self.task_type_label.setStyleSheet(f"color: {COLORS['text_secondary']}; padding: 0 10px;")
        self.task_type_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.task_type_label.mousePressEvent = self.on_task_type_clicked
        title_layout.addWidget(self.task_type_label)
        
        # 新建项目按钮
        new_project_btn = QPushButton("+ 新建项目")
        new_project_btn.setObjectName("secondary")
        new_project_btn.clicked.connect(self.create_new_project)
        title_layout.addWidget(new_project_btn)
        
        # 删除项目按钮
        delete_project_btn = QPushButton("🗑 删除项目")
        delete_project_btn.setObjectName("secondary")
        delete_project_btn.clicked.connect(self.delete_current_project)
        title_layout.addWidget(delete_project_btn)
        
        title_layout.addStretch()
        
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

        self.btn_refresh_status = QPushButton("刷新")
        self.btn_refresh_status.setObjectName("secondary")
        self.btn_refresh_status.setMinimumWidth(72)
        self.btn_refresh_status.setFixedHeight(34)
        self.btn_refresh_status.setToolTip("重新读取当前项目的图片与标注状态")
        self.btn_refresh_status.clicked.connect(self.refresh_project_images)
        layout.addWidget(self.btn_refresh_status)
        self._update_refresh_button_state()
        
        return status_bar

    def _update_refresh_button_state(self):
        """根据当前项目状态更新刷新按钮可用性。"""
        if hasattr(self, 'btn_refresh_status'):
            self.btn_refresh_status.setEnabled(bool(self.current_project_id))
    
    def load_projects(self):
        """加载项目列表"""
        self.project_combo.clear()
        self.project_combo.addItem("请选择项目...", None)
        
        projects = db.get_all_projects()
        for project in projects:
            self.project_combo.addItem(project['name'], project['id'])
        self._update_refresh_button_state()
    
    def on_project_changed(self, index):
        """项目选择改变"""
        project_id = self.project_combo.currentData()
        if project_id:
            self.current_project_id = project_id
            self._update_refresh_button_state()
            self.load_project_images()
            # 更新任务类别显示
            self.update_task_type_display()
        else:
            self.stop_image_loading()
            self.current_project_id = None
            self._update_refresh_button_state()
            self.image_list.clear()
            self.images = []
            self.thumbnail_widgets.clear()
            self.progress_bar.setVisible(False)
            self.update_status_bar()
            # 重置任务类别显示
            self.task_type_label.setText("任务类别: 未设置")

    def refresh_project_images(self):
        """手动刷新当前项目图片与标注状态。"""
        if not self.current_project_id:
            return
        self.load_project_images()
    
    def update_task_type_display(self):
        """更新任务类别显示"""
        if self.current_project_id:
            project = db.get_project(self.current_project_id)
            if project:
                task_type = project.get('type', '未设置')
                self.task_type_label.setText(f"任务类别: {task_type}")
    
    def on_task_type_clicked(self, event):
        """任务类别点击事件"""
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return
        
        # 显示任务类别选择对话框
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QRadioButton, QPushButton, QLabel
        
        dialog = QDialog(self)
        dialog.setWindowTitle("选择任务类型")
        dialog.setFixedSize(300, 250)
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
        
        label = QLabel("请选择项目的任务类型:")
        layout.addWidget(label)
        
        # 获取当前任务类型
        current_task_type = "detect"
        project = db.get_project(self.current_project_id)
        if project and project.get('type'):
            current_task_type = project['type']
        
        detect_radio = QRadioButton("detect (目标检测)")
        if current_task_type == "detect":
            detect_radio.setChecked(True)
        layout.addWidget(detect_radio)
        
        segment_radio = QRadioButton("segment (实例分割)")
        if current_task_type == "segment":
            segment_radio.setChecked(True)
        layout.addWidget(segment_radio)
        
        pose_radio = QRadioButton("pose (关键点检测)")
        if current_task_type == "pose" or current_task_type == "point":
            pose_radio.setChecked(True)
        layout.addWidget(pose_radio)
        
        classify_radio = QRadioButton("classify (图像分类)")
        if current_task_type == "classify":
            classify_radio.setChecked(True)
        layout.addWidget(classify_radio)
        

        
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
        
        # 确定任务类型
        if detect_radio.isChecked():
            task_type = "detect"
        elif segment_radio.isChecked():
            task_type = "segment"
        elif pose_radio.isChecked():
            task_type = "pose"
        elif classify_radio.isChecked():
            task_type = "classify"

        else:
            task_type = "detect"
        
        # 更新项目的任务类型
        db.update_project(self.current_project_id, type=task_type)
        # 更新显示
        self.update_task_type_display()
    
    def create_new_project(self):
        """创建新项目"""
        from PyQt6.QtWidgets import QInputDialog, QDialog, QVBoxLayout, QHBoxLayout, QRadioButton, QPushButton, QLabel
        
        name, ok = QInputDialog.getText(self, "新建项目", "请输入项目名称:")
        if not ok or not name:
            return
        
        # 创建任务标签选择对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("选择任务类型")
        dialog.setFixedSize(300, 300)
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
        
        label = QLabel("请选择项目的任务类型:")
        layout.addWidget(label)
        
        detect_radio = QRadioButton("detect (目标检测)")
        detect_radio.setChecked(True)
        layout.addWidget(detect_radio)
        
        segment_radio = QRadioButton("segment (实例分割)")
        layout.addWidget(segment_radio)
        
        pose_radio = QRadioButton("pose (关键点检测)")
        layout.addWidget(pose_radio)
        
        classify_radio = QRadioButton("classify (图像分类)")
        layout.addWidget(classify_radio)
        

        
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
        
        # 确定任务类型
        if detect_radio.isChecked():
            task_type = "detect"
        elif segment_radio.isChecked():
            task_type = "segment"
        elif pose_radio.isChecked():
            task_type = "pose"
        elif classify_radio.isChecked():
            task_type = "classify"

        else:
            task_type = "detect"
        
        # 创建项目
        project_id = db.create_project(
            name=name,
            description="",
            project_type=task_type,
            classes=[]
        )
        self.load_projects()
        index = self.project_combo.findData(project_id)
        if index >= 0:
            self.project_combo.setCurrentIndex(index)
    
    def delete_current_project(self):
        """删除当前选中的项目"""
        project_id = self.project_combo.currentData()
        
        if not project_id:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return
        
        # 获取项目名称
        project_name = self.project_combo.currentText()
        
        # 确认删除
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除项目 \"{project_name}\" 吗？\n\n这将删除该项目中的所有图片和标注数据，此操作不可恢复！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.stop_image_loading()
                # 获取项目信息，包括存储路径
                project = db.get_project(project_id)
                project_storage_path = project.get('storage_path', '') if project else ''
                
                # 删除项目（数据库会自动级联删除相关图片和标注）
                db.delete_project(project_id)
                
                # 删除项目文件夹
                if project_storage_path and os.path.exists(project_storage_path):
                    try:
                        import shutil
                        shutil.rmtree(project_storage_path)
                    except Exception as e:
                        # 文件夹删除失败不影响项目删除
                        print(f"删除项目文件夹失败: {e}")
                
                # 清空当前项目ID
                self.current_project_id = None
                removed_storage_paths = [img.get('storage_path', '') for img in self.images]
                
                # 清空图片列表
                self.image_list.clear()
                self.images = []
                self.thumbnail_widgets.clear()
                self._remove_cached_thumbnails(removed_storage_paths)
                
                # 更新状态栏
                self.update_status_bar()
                
                # 重新加载项目列表
                self.load_projects()
                
                QMessageBox.information(self, "成功", f"项目 \"{project_name}\" 已删除")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除项目失败: {str(e)}")
    
    def load_project_images(self):
        """加载项目图像 - 使用多线程"""
        if not self.current_project_id:
            return

        # 停止之前的加载，并创建新的加载世代
        self.stop_image_loading(reset_progress=False)
        current_generation = self._image_load_generation
        
        # 清空列表
        self.image_list.clear()
        self.thumbnail_widgets.clear()
        
        # 从数据库获取图片列表（很快）
        self.images = db.get_project_images(self.current_project_id)
        self.update_status_bar()
        
        if not self.images:
            self.progress_bar.setVisible(False)
            return
        
        # 先创建所有列表项（显示占位符）
        uncached_tasks = []
        for index, image_data in enumerate(self.images):
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

            storage_path = image_data.get('storage_path', '')
            cached_pixmap = self.thumbnail_cache.get(storage_path)
            if cached_pixmap is not None and not cached_pixmap.isNull():
                item.setIcon(QIcon(cached_pixmap))
            else:
                uncached_tasks.append((index, image_data))
            
            self.image_list.addItem(item)

        self.filter_images(self.view_combo.currentText())
        
        if not uncached_tasks:
            self.progress_bar.setVisible(False)
            return

        # 仅对未命中的缩略图显示非阻塞进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(uncached_tasks))
        self.progress_bar.setValue(0)
        
        # 启动后台加载线程
        self.load_worker = ImageLoadWorker(uncached_tasks)
        self.load_worker.image_loaded.connect(
            lambda index, pixmap, storage_path, generation=current_generation: self.on_image_loaded(
                generation, index, pixmap, storage_path
            )
        )
        self.load_worker.progress.connect(
            lambda current, total, generation=current_generation: self.on_load_progress(
                generation, current, total
            )
        )
        self.load_worker.finished_loading.connect(
            lambda generation=current_generation: self.on_load_finished(generation)
        )
        self.load_worker.start()
    
    def on_image_loaded(self, generation: int, index: int, pixmap: QPixmap, storage_path: str):
        """单个图片加载完成回调（在主线程执行）"""
        if generation != self._image_load_generation:
            return
        if index < self.image_list.count():
            item = self.image_list.item(index)
            if item:
                # 设置图标
                icon = QIcon(pixmap)
                item.setIcon(icon)
                # 缓存
                self.thumbnail_cache[storage_path] = pixmap
    
    def on_load_progress(self, generation: int, current: int, total: int):
        """加载进度回调"""
        if generation != self._image_load_generation:
            return
        self.progress_bar.setValue(current)
    
    def on_load_finished(self, generation: int):
        """加载完成回调"""
        if generation != self._image_load_generation:
            return
        self.load_worker = None
        self.progress_bar.setVisible(False)

        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.hide_loading()
            self.loading_overlay.deleteLater()
            delattr(self, 'loading_overlay')

    def _build_image_list_item(self, image_data: Dict) -> QListWidgetItem:
        """创建图片列表项（占位图标）"""
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, image_data['id'])
        item.setText(image_data['filename'])
        item.setToolTip(f"{image_data['filename']}\n{image_data.get('width', 0)}x{image_data.get('height', 0)}")

        status = image_data.get('status', 'pending')
        if status == 'annotated':
            item.setBackground(QColor(COLORS['success']))

        item.setSizeHint(QSize(180, 200))
        return item

    def _load_thumbnail_pixmap(self, storage_path: str) -> QPixmap:
        """同步加载单张缩略图（用于增量追加）"""
        pixmap = None
        if storage_path and os.path.exists(storage_path):
            try:
                img = cv2.imread(storage_path)
                if img is not None:
                    img = cv2.resize(img, (160, 160))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, ch = img.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
            except Exception:
                pass

        if pixmap is None or pixmap.isNull():
            pixmap = QPixmap(160, 160)
            pixmap.fill(QColor(COLORS['sidebar']))
        return pixmap

    def _append_imported_images(self, before_image_ids: set) -> int:
        """导入后仅增量追加新图片，返回追加数量"""
        if not self.current_project_id:
            return 0

        latest_images = db.get_project_images(self.current_project_id)
        new_images = [img for img in latest_images if img.get('id') not in before_image_ids]
        if not new_images:
            return 0

        self.images.extend(new_images)

        for image_data in new_images:
            item = self._build_image_list_item(image_data)
            self.image_list.addItem(item)

            storage_path = image_data.get('storage_path', '')
            pixmap = self.thumbnail_cache.get(storage_path)
            if pixmap is None:
                pixmap = self._load_thumbnail_pixmap(storage_path)
                self.thumbnail_cache[storage_path] = pixmap
            item.setIcon(QIcon(pixmap))

        self.update_status_bar()
        self.filter_images(self.view_combo.currentText())
        return len(new_images)

    
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
        
        self._video_import_before_ids = {img.get('id') for img in self.images}
        
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
        
        # 检查项目是否有任务标签
        project = db.get_project(self.current_project_id)
        if not project:
            QMessageBox.warning(self, "提示", "项目信息获取失败")
            return
        
        task_type = project.get('type')
        if not task_type or task_type not in ['detect', 'segment', 'pose', 'classify']:
            # 提示选择任务类型
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QRadioButton, QPushButton, QLabel
            
            dialog = QDialog(self)
            dialog.setWindowTitle("选择任务类型")
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
            
            label = QLabel("请选择项目的任务类型:")
            layout.addWidget(label)
            
            detect_radio = QRadioButton("detect (目标检测)")
            detect_radio.setChecked(True)
            layout.addWidget(detect_radio)
            
            segment_radio = QRadioButton("segment (实例分割)")
            layout.addWidget(segment_radio)
            
            pose_radio = QRadioButton("pose (关键点检测)")
            layout.addWidget(pose_radio)
            
            cls_radio = QRadioButton("cls (分类)")
            layout.addWidget(cls_radio)
            

            
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
            
            # 确定任务类型
            if detect_radio.isChecked():
                task_type = "detect"
            elif segment_radio.isChecked():
                task_type = "segment"
            elif pose_radio.isChecked():
                task_type = "pose"
            elif cls_radio.isChecked():
                task_type = "classify"

            else:
                task_type = "detect"
            
            # 更新项目的任务类型
            db.update_project(self.current_project_id, type=task_type)
        
        # 选择标注格式
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
        
        # 检查项目是否已经有标注
        project_images = db.get_project_images(self.current_project_id)
        has_annotations = False
        for image in project_images:
            annotations = db.get_image_annotations(image['id'])
            if annotations:
                has_annotations = True
                break
        
        # 如果有标注，提示是否覆盖
        overwrite = False
        if has_annotations:
            reply = QMessageBox.question(
                self, "覆盖标注",
                "项目中已经存在标注，是否覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                overwrite = True
        
        # 显示加载动画
        self.loading_overlay = LoadingOverlay(self, "正在导入YOLO标注...")
        self.loading_overlay.show_loading()
        
        # 创建后台线程来执行导入操作
        from PyQt6.QtCore import QThread, pyqtSignal
        
        class AnnotationImportThread(QThread):
            """标注导入线程"""
            
            finished = pyqtSignal(bool, str, int, int)
            
            def __init__(self, project_id, labels_dir, images_dir, overwrite):
                super().__init__()
                self.project_id = project_id
                self.labels_dir = labels_dir
                self.images_dir = images_dir
                self.overwrite = overwrite
            
            def run(self):
                """运行导入"""
                try:
                    from core.annotation_importer import AnnotationImporter
                    importer = AnnotationImporter(self.project_id)
                    imported, skipped = importer.import_yolo_annotations(
                        self.labels_dir, self.images_dir, self.overwrite
                    )
                    self.finished.emit(True, "导入成功", imported, skipped)
                except Exception as e:
                    self.finished.emit(False, f"导入失败: {e}", 0, 0)
        
        # 创建并启动线程
        self.import_thread = AnnotationImportThread(
            self.current_project_id, labels_dir, images_dir, overwrite
        )
        self.import_thread.finished.connect(self.on_annotation_import_finished)
        self.import_thread.start()
    
    def on_annotation_import_finished(self, success, message, imported, skipped):
        """标注导入完成回调"""
        # 隐藏加载动画
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.hide_loading()
            self.loading_overlay.deleteLater()
            delattr(self, 'loading_overlay')
        
        # 重新加载项目图片
        self.load_project_images()
        
        # 显示结果
        if success:
            QMessageBox.information(
                self, "导入完成",
                f"YOLO标注导入完成！\n成功导入: {imported} 个标注\n跳过: {skipped} 个"
            )
        else:
            QMessageBox.critical(self, "导入失败", message)
    
    def import_coco_annotations(self):
        """导入COCO标注"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择COCO标注文件", "",
            "JSON文件 (*.json);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
        
        # 检查项目是否已经有标注
        project_images = db.get_project_images(self.current_project_id)
        has_annotations = False
        for image in project_images:
            annotations = db.get_image_annotations(image['id'])
            if annotations:
                has_annotations = True
                break
        
        # 如果有标注，提示是否覆盖
        overwrite = False
        if has_annotations:
            reply = QMessageBox.question(
                self, "覆盖标注",
                "项目中已经存在标注，是否覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                overwrite = True
        
        # 显示加载动画
        self.loading_overlay = LoadingOverlay(self, "正在导入COCO标注...")
        self.loading_overlay.show_loading()
        
        # 创建后台线程来执行导入操作
        from PyQt6.QtCore import QThread, pyqtSignal
        
        class AnnotationImportThread(QThread):
            """标注导入线程"""
            
            finished = pyqtSignal(bool, str, int, int)
            
            def __init__(self, project_id, file_path, overwrite):
                super().__init__()
                self.project_id = project_id
                self.file_path = file_path
                self.overwrite = overwrite
            
            def run(self):
                """运行导入"""
                try:
                    from core.annotation_importer import AnnotationImporter
                    importer = AnnotationImporter(self.project_id)
                    imported, skipped = importer.import_coco_annotations(
                        self.file_path, self.overwrite
                    )
                    self.finished.emit(True, "导入成功", imported, skipped)
                except Exception as e:
                    self.finished.emit(False, f"导入失败: {e}", 0, 0)
        
        # 创建并启动线程
        self.import_thread = AnnotationImportThread(
            self.current_project_id, file_path, overwrite
        )
        self.import_thread.finished.connect(self.on_annotation_import_finished)
        self.import_thread.start()
    
    def import_voc_annotations(self):
        """导入VOC标注"""
        voc_dir = QFileDialog.getExistingDirectory(
            self, "选择VOC标注文件夹 (Annotations)", "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not voc_dir:
            return
        
        # 检查项目是否已经有标注
        project_images = db.get_project_images(self.current_project_id)
        has_annotations = False
        for image in project_images:
            annotations = db.get_image_annotations(image['id'])
            if annotations:
                has_annotations = True
                break
        
        # 如果有标注，提示是否覆盖
        overwrite = False
        if has_annotations:
            reply = QMessageBox.question(
                self, "覆盖标注",
                "项目中已经存在标注，是否覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                overwrite = True
        
        # 显示加载动画
        self.loading_overlay = LoadingOverlay(self, "正在导入VOC标注...")
        self.loading_overlay.show_loading()
        
        # 创建后台线程来执行导入操作
        from PyQt6.QtCore import QThread, pyqtSignal
        
        class AnnotationImportThread(QThread):
            """标注导入线程"""
            
            finished = pyqtSignal(bool, str, int, int)
            
            def __init__(self, project_id, voc_dir, overwrite):
                super().__init__()
                self.project_id = project_id
                self.voc_dir = voc_dir
                self.overwrite = overwrite
            
            def run(self):
                """运行导入"""
                try:
                    from core.annotation_importer import AnnotationImporter
                    importer = AnnotationImporter(self.project_id)
                    imported, skipped = importer.import_voc_annotations(
                        self.voc_dir, self.overwrite
                    )
                    self.finished.emit(True, "导入成功", imported, skipped)
                except Exception as e:
                    self.finished.emit(False, f"导入失败: {e}", 0, 0)
        
        # 创建并启动线程
        self.import_thread = AnnotationImportThread(
            self.current_project_id, voc_dir, overwrite
        )
        self.import_thread.finished.connect(self.on_annotation_import_finished)
        self.import_thread.start()
    
    def process_folder_import(self, folder_path: str):
        """处理文件夹导入"""
        if not self.current_project_id:
            return
        
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            before_ids = {img.get('id') for img in self.images}
            
            import_manager = ImportManager(self.current_project_id)
            imported, skipped = import_manager.import_folder(
                folder_path,
                progress_callback=self.update_import_progress
            )
            if imported > 0:
                appended = self._append_imported_images(before_ids)
                if appended == 0:
                    # 兜底：若无法识别新增项，回退全量重载
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
            before_ids = {img.get('id') for img in self.images}
            
            import_manager = ImportManager(self.current_project_id)
            imported, skipped = import_manager.import_images(
                file_paths,
                progress_callback=self.update_import_progress
            )
            if imported > 0:
                appended = self._append_imported_images(before_ids)
                if appended == 0:
                    # 兜底：若无法识别新增项，回退全量重载
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
            before_ids = getattr(self, '_video_import_before_ids', None)
            if imported > 0 and isinstance(before_ids, set):
                appended = self._append_imported_images(before_ids)
                if appended == 0:
                    # 兜底：若无法识别新增项，回退全量重载
                    self.load_project_images()
            elif imported > 0:
                # 未获取到导入前快照时，保持原有行为
                self.load_project_images()
            
            # 显示成功消息
            QMessageBox.information(
                self, "导入完成",
                f"视频导入完成！\n成功导入: {imported} 帧\n跳过: {skipped} 帧"
            )
        else:
            # 显示错误消息
            QMessageBox.critical(self, "导入失败", message)

        if hasattr(self, '_video_import_before_ids'):
            delattr(self, '_video_import_before_ids')
    
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
            self.stop_image_loading()
            deleted = 0
            failed = 0
            
            # 使用副本迭代，避免删除过程中修改原列表导致遍历异常
            images_snapshot = list(self.images)
            for image in images_snapshot:
                if db.delete_image(image['id']):
                    deleted += 1
                else:
                    failed += 1

            # 全部删除成功时，直接本地清空，避免触发整页重载
            if failed == 0:
                removed_storage_paths = [img.get('storage_path', '') for img in self.images]
                self.images.clear()
                self.image_list.clear()
                self.thumbnail_widgets.clear()
                self._remove_cached_thumbnails(removed_storage_paths)
                self.update_status_bar()
            else:
                # 部分失败时回退到全量重载，确保UI与数据库一致
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

        self.stop_image_loading()
        deleted = 0
        failed = 0
        deleted_ids = []
        
        for item in selected_items:
            image_id = item.data(Qt.ItemDataRole.UserRole)
            if db.delete_image(image_id):
                deleted += 1
                deleted_ids.append(image_id)
            else:
                failed += 1

        # 仅移除已成功删除的项，避免每次删除都整页重载
        if deleted_ids:
            deleted_id_set = set(deleted_ids)

            # 先更新内存数据
            removed_storage_paths = {
                img.get('storage_path', '')
                for img in self.images
                if img.get('id') in deleted_id_set
            }
            self.images = [img for img in self.images if img.get('id') not in deleted_id_set]

            # 清理缩略图缓存
            for path in removed_storage_paths:
                if path in self.thumbnail_cache:
                    del self.thumbnail_cache[path]

            # 再移除列表项（倒序删除避免索引变化）
            rows_to_remove = []
            for i in range(self.image_list.count()):
                item = self.image_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) in deleted_id_set:
                    rows_to_remove.append(i)
            for row in reversed(rows_to_remove):
                self.image_list.takeItem(row)

            self.update_status_bar()
        
        if failed == 0:
            QMessageBox.information(self, "删除完成", f"已成功删除 {deleted} 张图片")
        else:
            QMessageBox.warning(self, "删除完成", f"成功删除 {deleted} 张，失败 {failed} 张")
