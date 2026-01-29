# -*- coding: utf-8 -*-
"""
自动打标签弹窗
用于设置自动打标签的参数和选项
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QFormLayout, QRadioButton, QDoubleSpinBox,
    QCheckBox, QListWidget, QListWidgetItem, QSplitter, QMessageBox,
    QFileDialog, QScrollArea, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
import os
from typing import Dict, List, Optional

from gui.styles import COLORS

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


class AutoLabelDialog(QDialog):
    """自动打标签弹窗"""
    
    # 信号定义
    single_inference_requested = pyqtSignal(str, float, float, dict, str)  # 单张推理请求
    batch_inference_requested = pyqtSignal(str, float, float, dict, list, bool)  # 批量推理请求
    
    def __init__(self, parent=None, project_classes=None):
        super().__init__(parent)
        self.setWindowTitle("自动打标签设置")
        self.setMinimumSize(700, 500)
        
        # 当前项目类别
        self.project_classes = project_classes or []
        
        # 模型信息
        self.selected_model_version = "YOLOv8"
        self.selected_model_size = "n"
        self.model_source = "official"  # official or custom
        self.custom_model_path = ""
        
        # 推理参数
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45
        self.infer_only_unlabeled = True
        self.overwrite_labels = False
        
        # 类别映射
        self.class_mappings = {}
        
        # 初始化UI
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        
        # 模型选择组
        model_group = self.create_model_selection_group()
        main_layout.addWidget(model_group)
        
        # 推理参数组
        params_group = self.create_inference_params_group()
        main_layout.addWidget(params_group)
        
        # 类别映射组
        mapping_group = self.create_class_mapping_group()
        main_layout.addWidget(mapping_group)
        
        # 按钮组
        button_layout = QHBoxLayout()
        
        # 保存按钮
        self.btn_save = QPushButton("保存")
        self.btn_save.clicked.connect(self.accept)
        button_layout.addWidget(self.btn_save)
        
        # 取消按钮
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_cancel)
        
        main_layout.addLayout(button_layout)
    
    def create_model_selection_group(self) -> QGroupBox:
        """创建模型选择组"""
        group = QGroupBox("模型选择")
        group.setStyleSheet(self.get_group_style())
        
        layout = QVBoxLayout(group)
        
        # 模型版本和型号
        version_size_layout = QHBoxLayout()
        
        # 模型版本
        version_layout = QFormLayout()
        self.cb_model_version = QComboBox()
        self.cb_model_version.addItems(sorted(ULTRALYTICS_MODELS.keys()))
        self.cb_model_version.currentTextChanged.connect(self.on_model_version_changed)
        version_layout.addRow("模型版本:", self.cb_model_version)
        version_size_layout.addLayout(version_layout)
        
        # 模型型号
        size_layout = QFormLayout()
        self.cb_model_size = QComboBox()
        size_layout.addRow("型号:", self.cb_model_size)
        version_size_layout.addLayout(size_layout)
        
        layout.addLayout(version_size_layout)
        
        # 模型来源
        source_group = QGroupBox("模型来源")
        source_group.setStyleSheet(self.get_inner_group_style())
        source_layout = QVBoxLayout(source_group)
        
        # 官方预训练模型
        self.rbtn_official = QRadioButton("官方预训练模型")
        self.rbtn_official.setChecked(True)
        self.rbtn_official.toggled.connect(self.on_model_source_changed)
        source_layout.addWidget(self.rbtn_official)
        
        # 自定义模型
        custom_layout = QHBoxLayout()
        self.rbtn_custom = QRadioButton("自定义模型")
        self.rbtn_custom.toggled.connect(self.on_model_source_changed)
        custom_layout.addWidget(self.rbtn_custom)
        
        self.btn_browse_model = QPushButton("浏览...")
        self.btn_browse_model.setEnabled(False)
        self.btn_browse_model.clicked.connect(self.browse_custom_model)
        custom_layout.addWidget(self.btn_browse_model)
        
        source_layout.addLayout(custom_layout)
        
        layout.addWidget(source_group)
        
        # 初始化模型型号列表
        self.on_model_version_changed(self.cb_model_version.currentText())
        
        return group
    
    def create_inference_params_group(self) -> QGroupBox:
        """创建推理参数组"""
        group = QGroupBox("推理参数")
        group.setStyleSheet(self.get_group_style())
        
        layout = QVBoxLayout(group)
        
        # 阈值设置
        thresholds_layout = QHBoxLayout()
        
        # 置信度阈值
        conf_layout = QFormLayout()
        self.sb_conf_threshold = QDoubleSpinBox()
        self.sb_conf_threshold.setRange(0.0, 1.0)
        self.sb_conf_threshold.setSingleStep(0.05)
        self.sb_conf_threshold.setValue(0.5)
        conf_layout.addRow("置信度:", self.sb_conf_threshold)
        thresholds_layout.addLayout(conf_layout)
        
        # IOU阈值
        iou_layout = QFormLayout()
        self.sb_iou_threshold = QDoubleSpinBox()
        self.sb_iou_threshold.setRange(0.0, 1.0)
        self.sb_iou_threshold.setSingleStep(0.05)
        self.sb_iou_threshold.setValue(0.45)
        iou_layout.addRow("IOU阈值:", self.sb_iou_threshold)
        thresholds_layout.addLayout(iou_layout)
        
        layout.addLayout(thresholds_layout)
        
        # 推理选项
        options_layout = QVBoxLayout()
        
        self.chk_only_unlabeled = QCheckBox("仅推理无标签数据")
        self.chk_only_unlabeled.setChecked(True)
        options_layout.addWidget(self.chk_only_unlabeled)
        
        self.chk_overwrite = QCheckBox("覆盖原标签")
        self.chk_overwrite.setChecked(False)
        options_layout.addWidget(self.chk_overwrite)
        
        layout.addLayout(options_layout)
        
        return group
    
    def create_class_mapping_group(self) -> QGroupBox:
        """创建类别映射组"""
        group = QGroupBox("类别映射")
        group.setStyleSheet(self.get_group_style())
        
        layout = QVBoxLayout(group)
        
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
        self.btn_load_classes = QPushButton("加载模型classes.txt")
        self.btn_load_classes.clicked.connect(self.load_model_classes)
        self.btn_load_classes.setEnabled(False)
        model_class_file_layout.addWidget(self.btn_load_classes)
        self.model_classes_path = ""
        self.model_classes = []
        layout.addLayout(model_class_file_layout)
        
        # 分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 模型类别列表
        model_class_widget = QWidget()
        model_class_layout = QVBoxLayout(model_class_widget)
        model_class_layout.addWidget(QLabel("模型类别"))
        self.model_class_list = QListWidget()
        model_class_layout.addWidget(self.model_class_list)
        splitter.addWidget(model_class_widget)
        
        # 项目类别列表
        project_class_widget = QWidget()
        project_class_layout = QVBoxLayout(project_class_widget)
        project_class_layout.addWidget(QLabel("项目类别"))
        self.project_class_list = QListWidget()
        project_class_layout.addWidget(self.project_class_list)
        splitter.addWidget(project_class_widget)
        
        layout.addWidget(splitter)
        
        # 映射按钮
        mapping_buttons_layout = QHBoxLayout()
        
        self.btn_edit_mapping = QPushButton("编辑映射")
        self.btn_edit_mapping.clicked.connect(self.edit_mapping)
        self.btn_edit_mapping.setEnabled(False)
        mapping_buttons_layout.addWidget(self.btn_edit_mapping)
        
        self.btn_apply_all = QPushButton("一键应用模型类别")
        self.btn_apply_all.clicked.connect(self.apply_all_model_classes)
        self.btn_apply_all.setEnabled(False)
        mapping_buttons_layout.addWidget(self.btn_apply_all)
        
        layout.addLayout(mapping_buttons_layout)
        
        # 初始化类别列表
        self.update_class_lists()
        
        return group
    
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
    
    def get_inner_group_style(self) -> str:
        """获取内部分组框样式"""
        return f"""
            QGroupBox {{
                font-weight: normal;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                font-size: 12px;
            }}
        """
    
    def on_model_version_changed(self, version: str):
        """模型版本改变时更新型号列表"""
        self.cb_model_size.clear()
        if version in ULTRALYTICS_MODELS:
            sizes = ULTRALYTICS_MODELS[version]['sizes']
            for size in sizes:
                display_name = SIZE_NAMES.get(size, size)
                self.cb_model_size.addItem(display_name, size)
    
    def on_model_source_changed(self):
        """模型来源改变时更新界面"""
        if self.rbtn_custom.isChecked():
            self.btn_browse_model.setEnabled(True)
            self.model_source = "custom"
        else:
            self.btn_browse_model.setEnabled(False)
            self.model_source = "official"
    
    def browse_custom_model(self):
        """浏览自定义模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", 
            "", "PyTorch models (*.pt *.pth)"
        )
        if file_path:
            self.custom_model_path = file_path
    
    def update_class_lists(self):
        """更新类别列表"""
        # 清空列表
        self.model_class_list.clear()
        self.project_class_list.clear()
        
        # 添加模型类别（示例）
        model_classes = ["person", "car", "dog", "cat", "bird"]
        for i, cls in enumerate(model_classes):
            item = QListWidgetItem(f"{i}: {cls}")
            self.model_class_list.addItem(item)
        
        # 添加项目类别
        for cls in self.project_classes:
            item = QListWidgetItem(f"{cls['id']}: {cls['name']}")
            color = QColor(cls.get('color', '#808080'))
            item.setForeground(color)
            self.project_class_list.addItem(item)
    
    def edit_mapping(self):
        """编辑类别映射"""
        # 这里可以实现一个更复杂的映射编辑界面
        QMessageBox.information(self, "编辑映射", "类别映射编辑功能开发中...")
    
    def add_class(self):
        """添加新类别"""
        # 这里可以实现添加新类别的功能
        QMessageBox.information(self, "添加类别", "添加类别功能开发中...")
    
    def on_single_inference(self):
        """单张推理"""
        # 获取模型路径
        model_path = self.get_model_path()
        if not model_path:
            QMessageBox.warning(self, "错误", "请选择有效的模型")
            return
        
        # 发送信号
        self.single_inference_requested.emit(
            model_path,
            self.sb_conf_threshold.value(),
            self.sb_iou_threshold.value(),
            self.class_mappings,
            getattr(self, 'current_image_path', '')
        )
        # 关闭弹窗
        self.accept()
    
    def on_batch_inference(self):
        """一键推理"""
        # 获取模型路径
        model_path = self.get_model_path()
        if not model_path:
            QMessageBox.warning(self, "错误", "请选择有效的模型")
            return
        
        # 发送信号
        self.batch_inference_requested.emit(
            model_path,
            self.sb_conf_threshold.value(),
            self.sb_iou_threshold.value(),
            self.class_mappings,
            getattr(self, 'current_images', []),
            self.chk_only_unlabeled.isChecked()
        )
        # 关闭弹窗
        self.accept()
    
    def run_single_inference(self, image_path: str):
        """运行单张推理"""
        self.current_image_path = image_path
        self.exec()
    
    def run_batch_inference(self, images: list):
        """运行批量推理"""
        self.current_images = images
        self.exec()
    
    def get_model_path(self) -> str:
        """获取模型路径"""
        if self.model_source == "custom":
            return self.custom_model_path
        else:
            # 构建官方模型名称
            version = self.cb_model_version.currentText()
            size = self.cb_model_size.currentData() or self.cb_model_size.currentText()
            if version in ULTRALYTICS_MODELS:
                prefix = ULTRALYTICS_MODELS[version]['prefix']
                return f"{prefix}{size}"
        return ""
    
    def on_enable_mapping_changed(self, state):
        """启用映射选项改变时的处理"""
        enabled = state == Qt.CheckState.Checked.value
        self.btn_load_classes.setEnabled(enabled)
        self.btn_edit_mapping.setEnabled(enabled and len(self.model_classes) > 0)
        self.btn_apply_all.setEnabled(enabled and len(self.model_classes) > 0)
        
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
                    self.model_classes_path = file_path
                    self.model_classes = classes
                    self.update_model_class_list()
                    self.btn_edit_mapping.setEnabled(True)
                    self.btn_apply_all.setEnabled(True)
                    QMessageBox.information(self, "成功", f"成功加载 {len(classes)} 个模型类别")
                else:
                    QMessageBox.warning(self, "警告", "classes.txt文件为空")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载classes.txt文件失败: {str(e)}")
    
    def update_model_class_list(self):
        """更新模型类别列表"""
        self.model_class_list.clear()
        if self.model_classes:
            for i, cls in enumerate(self.model_classes):
                item = QListWidgetItem(f"{i}: {cls}")
                self.model_class_list.addItem(item)
        else:
            # 添加默认模型类别（示例）
            model_classes = ["person", "car", "dog", "cat", "bird"]
            for i, cls in enumerate(model_classes):
                item = QListWidgetItem(f"{i}: {cls}")
                self.model_class_list.addItem(item)
    
    def apply_all_model_classes(self):
        """一键应用模型类别到项目"""
        if not self.model_classes:
            QMessageBox.warning(self, "警告", "请先加载模型classes.txt文件")
            return
        
        # 创建新的项目类别列表
        new_classes = []
        for i, cls_name in enumerate(self.model_classes):
            # 生成随机颜色
            import random
            color = f"#{random.randint(0, 0xFFFFFF):06x}"
            new_classes.append({
                'id': i,
                'name': cls_name,
                'color': color
            })
        
        # 更新项目类别
        self.project_classes = new_classes
        self.update_project_class_list()
        
        # 发送信号通知主窗口更新类别
        # 这里可以添加一个信号来通知主窗口
        QMessageBox.information(self, "成功", f"成功应用 {len(new_classes)} 个模型类别到项目")
    
    def update_project_class_list(self):
        """更新项目类别列表"""
        self.project_class_list.clear()
        for cls in self.project_classes:
            item = QListWidgetItem(f"{cls['id']}: {cls['name']}")
            color = QColor(cls.get('color', '#808080'))
            item.setForeground(color)
            self.project_class_list.addItem(item)
    
    def edit_mapping(self):
        """编辑类别映射"""
        if not self.model_classes:
            QMessageBox.warning(self, "警告", "请先加载模型classes.txt文件")
            return
        
        # 这里可以实现一个更复杂的映射编辑界面
        # 暂时使用简单的消息框
        QMessageBox.information(self, "编辑映射", "类别映射编辑功能开发中...")
    
    def get_class_mappings(self):
        """获取类别映射"""
        if not self.chk_enable_mapping.isChecked():
            return {}
        
        # 这里可以返回更复杂的映射
        # 暂时返回空映射
        return self.class_mappings
    
    def update_class_lists(self):
        """更新类别列表"""
        # 清空列表
        self.model_class_list.clear()
        self.project_class_list.clear()
        
        # 添加模型类别
        if self.model_classes:
            for i, cls in enumerate(self.model_classes):
                item = QListWidgetItem(f"{i}: {cls}")
                self.model_class_list.addItem(item)
        else:
            # 添加默认模型类别（示例）
            model_classes = ["person", "car", "dog", "cat", "bird"]
            for i, cls in enumerate(model_classes):
                item = QListWidgetItem(f"{i}: {cls}")
                self.model_class_list.addItem(item)
        
        # 添加项目类别
        for cls in self.project_classes:
            item = QListWidgetItem(f"{cls['id']}: {cls['name']}")
            color = QColor(cls.get('color', '#808080'))
            item.setForeground(color)
            self.project_class_list.addItem(item)
    
    def set_classes(self, classes: list):
        """设置项目类别"""
        self.project_classes = classes
        self.update_class_lists()
