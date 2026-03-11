# -*- coding: utf-8 -*-
"""
自动打标签弹窗
用于设置自动打标签的参数和选项
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QFormLayout, QRadioButton, QDoubleSpinBox,
    QCheckBox, QListWidget, QListWidgetItem, QSplitter, QMessageBox,
    QFileDialog, QScrollArea, QWidget, QTabWidget, QTextEdit, QLineEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
import os
import json
from typing import Dict, List, Optional

from gui.styles import COLORS

# LLM配置文件路径
LLM_CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'llm_config.json')
SAM_CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'sam_config.json')

# 默认LLM配置
DEFAULT_LLM_CONFIG = {
    'api_key': '',
    'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'model_name': 'qwen-vl-max',
    'system_prompt': '''你是面向计算机视觉数据集的目标检测标注专家，仅输出指定目标的边界框坐标。
核心要求：
1. 目标类别：仅处理用户指定的类别，需找出图片中所有该类别实例；
2. 坐标格式：每个边界框以 [xmin, ymin, xmax, ymax] 格式输出；
3. 输出格式：每行一个目标，格式为 "标签,[xmin,ymin,xmax,ymax]"；
4. 无目标时输出空内容；
5. 仅返回坐标数据，无任何说明文字。''',
    'user_prompt': '''请检测图片中的所有 {target}，按以下格式返回每行一个：
{target},[xmin,ymin,xmax,ymax]
{target},[xmin,ymin,xmax,ymax]
...'''
}

DEFAULT_SAM_CONFIG = {
    "sam_type": "SAM",
    "model_file": "sam_b.pt",
    "device": "cpu",
    "imgsz": 1024,
    "conf": 0.4,
    "iou": 0.9,
    "retina_masks": True,
}

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

# SAM模型配置
SAM_MODELS = {
    "SAM": {
        "models": {
            "SAM base": "sam_b.pt",
            "SAM large": "sam_l.pt"
        }
    },
    "SAM2": {
        "models": {
            "SAM 2 tiny": "sam2_t.pt",
            "SAM 2 small": "sam2_s.pt",
            "SAM 2 base": "sam2_b.pt",
            "SAM 2 large": "sam2_l.pt",
            "SAM 2.1 tiny": "sam2.1_t.pt",
            "SAM 2.1 small": "sam2.1_s.pt",
            "SAM 2.1 base": "sam2.1_b.pt",
            "SAM 2.1 large": "sam2.1_l.pt"
        }
    },
    "MobileSAM": {
        "models": {
            "MobileSAM": "mobile_sam.pt"
        }
    },
    "FastSAM": {
        "models": {
            "FastSAM-s": "FastSAM-s.pt",
            "FastSAM-x": "FastSAM-x.pt"
        }
    }
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
    single_inference_requested = pyqtSignal(str, float, float, dict, str, str)  # 单张推理请求
    batch_inference_requested = pyqtSignal(str, float, float, dict, list, bool, str)  # 批量推理请求
    
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

        # 加载SAM配置
        self.load_sam_config()
        
        # 加载LLM配置
        self.load_llm_config()
        
    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(f"""
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
        
        # YOLO自动标注页面
        yolo_tab = self.create_yolo_tab()
        self.tab_widget.addTab(yolo_tab, "🤖 YOLO自动标注")
        
        # SAM自动标注页面
        sam_tab = self.create_sam_tab()
        self.tab_widget.addTab(sam_tab, "🎯 SAM自动标注")
        
        # LLM自动标注页面
        llm_tab = self.create_llm_tab()
        self.tab_widget.addTab(llm_tab, "🧠 LLM自动标注")
        
        main_layout.addWidget(self.tab_widget)
        
        # 按钮组
        button_layout = QHBoxLayout()
        
        # 保存按钮
        self.btn_save = QPushButton("保存")
        self.btn_save.clicked.connect(self.on_save_clicked)
        button_layout.addWidget(self.btn_save)
        
        # 取消按钮
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_cancel)
        
        main_layout.addLayout(button_layout)
    
    def create_yolo_tab(self) -> QWidget:
        """创建YOLO自动标注页面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # 模型选择组
        model_group = self.create_model_selection_group()
        layout.addWidget(model_group)
        
        # 推理参数组
        params_group = self.create_inference_params_group()
        layout.addWidget(params_group)
        
        # 类别映射组
        mapping_group = self.create_class_mapping_group()
        layout.addWidget(mapping_group)
        
        layout.addStretch()
        return tab
    
    def create_sam_tab(self) -> QWidget:
        """创建SAM自动标注页面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # SAM模型选择组
        sam_model_group = self.create_sam_model_group()
        layout.addWidget(sam_model_group)
        
        # SAM推理参数组
        sam_params_group = self.create_sam_params_group()
        layout.addWidget(sam_params_group)
        
        layout.addStretch()
        return tab
    
    def create_model_selection_group(self) -> QGroupBox:
        """创建模型选择组"""
        group = QGroupBox("模型选择")
        group.setStyleSheet(self.get_group_style())
        
        layout = QVBoxLayout(group)
        
        # 模型版本、型号和任务
        version_size_task_layout = QHBoxLayout()
        
        # 模型版本
        version_layout = QFormLayout()
        self.cb_model_version = QComboBox()
        self.cb_model_version.addItems(sorted(ULTRALYTICS_MODELS.keys()))
        self.cb_model_version.currentTextChanged.connect(self.on_model_version_changed)
        version_layout.addRow("模型版本:", self.cb_model_version)
        version_size_task_layout.addLayout(version_layout)
        
        # 模型型号
        size_layout = QFormLayout()
        self.cb_model_size = QComboBox()
        size_layout.addRow("型号:", self.cb_model_size)
        version_size_task_layout.addLayout(size_layout)
        
        # 任务类型
        task_layout = QFormLayout()
        self.cb_model_task = QComboBox()
        task_layout.addRow("任务类型:", self.cb_model_task)
        version_size_task_layout.addLayout(task_layout)
        
        layout.addLayout(version_size_task_layout)
        
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
        """模型版本改变时更新型号和任务类型列表"""
        self.cb_model_size.clear()
        self.cb_model_task.clear()
        
        if version in ULTRALYTICS_MODELS:
            # 更新型号列表
            sizes = ULTRALYTICS_MODELS[version]['sizes']
            for size in sizes:
                display_name = SIZE_NAMES.get(size, size)
                self.cb_model_size.addItem(display_name, size)
            
            # 更新任务类型列表
            tasks = ULTRALYTICS_MODELS[version]['tasks']
            for task in tasks:
                self.cb_model_task.addItem(task)
            
            # 默认选择第一个任务类型
            if tasks:
                self.cb_model_task.setCurrentIndex(0)
    
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
        
        # 获取任务类型
        model_task = self.cb_model_task.currentText() if hasattr(self, 'cb_model_task') else 'detect'
        
        # 发送信号
        self.single_inference_requested.emit(
            model_path,
            self.sb_conf_threshold.value(),
            self.sb_iou_threshold.value(),
            self.class_mappings,
            getattr(self, 'current_image_path', ''),
            model_task
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
        
        # 获取任务类型
        model_task = self.cb_model_task.currentText() if hasattr(self, 'cb_model_task') else 'detect'
        
        # 发送信号
        self.batch_inference_requested.emit(
            model_path,
            self.sb_conf_threshold.value(),
            self.sb_iou_threshold.value(),
            self.class_mappings,
            getattr(self, 'current_images', []),
            self.chk_only_unlabeled.isChecked(),
            model_task
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
    
    def on_save_clicked(self):
        """保存按钮点击事件，输出调试信息"""
        # 获取模型信息
        model_path = self.get_model_path()
        model_task = self.cb_model_task.currentText() if hasattr(self, 'cb_model_task') else 'detect'
        model_version = self.cb_model_version.currentText() if hasattr(self, 'cb_model_version') else ''
        model_size = self.cb_model_size.currentData() or self.cb_model_size.currentText() if hasattr(self, 'cb_model_size') else ''
        
        # 构建模型名称
        if self.model_source == "custom":
            model_name = os.path.basename(model_path) if model_path else "自定义模型"
        else:
            model_name = f"{model_version}-{model_size}-{model_task}" if model_version else model_path
        
        # 输出调试信息
        print("=" * 50)
        print("自动标注设置保存:")
        print(f"  当前任务类型: {model_task}")
        print(f"  模型名称: {model_name}")
        print(f"  模型路径: {model_path}")
        print(f"  模型版本: {model_version}")
        print(f"  模型大小: {model_size}")
        print(f"  模型来源: {self.model_source}")
        print("=" * 50)
        
        # 检查SAM模型是否存在
        if hasattr(self, 'cb_sam_type') and hasattr(self, 'cb_sam_model'):
            sam_type = self.cb_sam_type.currentText()
            model_file = self.cb_sam_model.currentData()
            
            if model_file:
                # 检查模型文件是否存在
                model_paths = [
                    model_file,
                    os.path.join('models', model_file),
                    os.path.join(os.path.expanduser('~'), '.cache', 'ultralytics', model_file),
                ]
                
                model_exists = False
                for path in model_paths:
                    if os.path.exists(path):
                        model_exists = True
                        break
                
                if not model_exists:
                    # 模型不存在，提示用户下载
                    reply = QMessageBox.question(
                        self,
                        "模型不存在",
                        f"SAM模型文件不存在: {model_file}\n\n"
                        f"模型类型: {sam_type}\n"
                        f"需要下载模型才能使用SAM功能。\n\n"
                        f"是否现在下载？\n"
                        f"（下载可能需要一些时间，取决于网络状况）",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    
                    if reply == QMessageBox.StandardButton.Yes:
                        # 尝试下载模型
                        self.download_sam_model(sam_type, model_file)
                    else:
                        # 用户选择不下载，清空SAM配置
                        print(f"[SAM] 用户选择不下载模型，SAM功能将不可用")
        
        # 保存SAM配置
        self.save_sam_config()

        # 保存LLM配置
        self.save_llm_config()
        
        # 调用accept保存设置
        self.accept()
    
    def download_sam_model(self, sam_type: str, model_file: str):
        """下载SAM模型"""
        try:
            from PyQt6.QtWidgets import QProgressDialog
            from PyQt6.QtCore import Qt, QThread, pyqtSignal
            
            class ModelDownloadWorker(QThread):
                """模型下载工作线程"""
                download_finished = pyqtSignal(bool, str)
                
                def __init__(self, sam_type, model_file):
                    super().__init__()
                    self.sam_type = sam_type
                    self.model_file = model_file
                
                def run(self):
                    try:
                        # 根据类型导入正确的类来触发下载
                        if self.sam_type == "FastSAM":
                            from ultralytics import FastSAM
                            model = FastSAM(self.model_file)
                        else:
                            # SAM, SAM2, MobileSAM 都使用SAM类
                            from ultralytics import SAM
                            model = SAM(self.model_file)
                        
                        self.download_finished.emit(True, f"模型 {self.model_file} 下载成功")
                    except Exception as e:
                        self.download_finished.emit(False, f"下载失败: {str(e)}")
            
            # 显示进度对话框
            progress = QProgressDialog(f"正在下载模型 {model_file}...", "取消", 0, 0, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setCancelButton(None)
            progress.show()
            
            # 创建下载线程
            self.download_worker = ModelDownloadWorker(sam_type, model_file)
            self.download_worker.download_finished.connect(
                lambda success, msg: self.on_model_download_finished(success, msg, progress)
            )
            self.download_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动下载失败: {str(e)}")
    
    def on_model_download_finished(self, success: bool, message: str, progress_dialog):
        """模型下载完成回调"""
        progress_dialog.close()
        
        if success:
            QMessageBox.information(self, "成功", message)
        else:
            QMessageBox.critical(self, "下载失败", message)
    
    def get_model_task(self) -> str:
        """获取当前选择的任务类型"""
        return self.cb_model_task.currentText() if hasattr(self, 'cb_model_task') else 'detect'
    
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
    
    # ==================== SAM相关方法 ====================
    
    def create_sam_model_group(self) -> QGroupBox:
        """创建SAM模型选择组"""
        group = QGroupBox("SAM模型选择")
        group.setStyleSheet(self.get_group_style())
        
        layout = QFormLayout(group)
        
        # SAM类型选择
        self.cb_sam_type = QComboBox()
        self.cb_sam_type.addItems(list(SAM_MODELS.keys()))
        self.cb_sam_type.currentTextChanged.connect(self.on_sam_type_changed)
        layout.addRow("模型类型:", self.cb_sam_type)
        
        # SAM型号选择
        self.cb_sam_model = QComboBox()
        layout.addRow("模型型号:", self.cb_sam_model)
        
        # 初始化型号列表
        self.on_sam_type_changed(self.cb_sam_type.currentText())
        
        # 设备选择
        self.cb_sam_device = QComboBox()
        self.cb_sam_device.addItems(["自动选择", "CPU", "CUDA:0", "CUDA:1", "CUDA:2", "CUDA:3"])
        layout.addRow("设备:", self.cb_sam_device)
        
        return group
    
    def create_sam_params_group(self) -> QGroupBox:
        """创建SAM推理参数组"""
        group = QGroupBox("推理参数")
        group.setStyleSheet(self.get_group_style())
        
        layout = QFormLayout(group)
        
        # 图像尺寸
        self.sb_sam_imgsz = QDoubleSpinBox()
        self.sb_sam_imgsz.setRange(256, 2048)
        self.sb_sam_imgsz.setValue(1024)
        self.sb_sam_imgsz.setSingleStep(64)
        layout.addRow("图像尺寸:", self.sb_sam_imgsz)
        
        # 置信度阈值（FastSAM用）
        self.sb_sam_conf = QDoubleSpinBox()
        self.sb_sam_conf.setRange(0.01, 1.0)
        self.sb_sam_conf.setValue(0.4)
        self.sb_sam_conf.setDecimals(2)
        self.sb_sam_conf.setSingleStep(0.05)
        layout.addRow("置信度阈值:", self.sb_sam_conf)
        
        # IoU阈值（FastSAM用）
        self.sb_sam_iou = QDoubleSpinBox()
        self.sb_sam_iou.setRange(0.1, 1.0)
        self.sb_sam_iou.setValue(0.9)
        self.sb_sam_iou.setDecimals(2)
        self.sb_sam_iou.setSingleStep(0.05)
        layout.addRow("IoU阈值:", self.sb_sam_iou)
        
        # Retina masks选项（FastSAM用）
        self.chk_sam_retina = QCheckBox("使用Retina Masks")
        self.chk_sam_retina.setChecked(True)
        layout.addRow(self.chk_sam_retina)
        
        return group
    
    def on_sam_type_changed(self, sam_type: str):
        """SAM类型改变时更新型号列表"""
        self.cb_sam_model.clear()
        if sam_type in SAM_MODELS:
            models = SAM_MODELS[sam_type]["models"]
            for name, file in models.items():
                self.cb_sam_model.addItem(f"{name} ({file})", file)
    
    def get_sam_config(self) -> dict:
        """获取SAM配置"""
        sam_type = self.cb_sam_type.currentText()
        model_file = self.cb_sam_model.currentData()
        
        # 获取设备
        device = self.cb_sam_device.currentText()
        if device == "自动选择":
            try:
                import torch
                device = '0' if torch.cuda.is_available() else 'cpu'
            except:
                device = 'cpu'
        elif device == "CPU":
            device = 'cpu'
        elif device.startswith("CUDA:"):
            device = device.split(":")[1]
        
        return {
            'sam_type': sam_type,
            'model_file': model_file,
            'device': device,
            'imgsz': int(self.sb_sam_imgsz.value()),
            'conf': self.sb_sam_conf.value(),
            'iou': self.sb_sam_iou.value(),
            'retina_masks': self.chk_sam_retina.isChecked()
        }

    def load_sam_config(self):
        """从配置文件加载SAM配置并应用到UI。"""
        config = DEFAULT_SAM_CONFIG.copy()
        if os.path.exists(SAM_CONFIG_FILE):
            try:
                with open(SAM_CONFIG_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    if isinstance(saved, dict):
                        config.update(saved)
            except Exception:
                pass

        sam_type = config.get("sam_type", "SAM")
        if sam_type in SAM_MODELS:
            self.cb_sam_type.setCurrentText(sam_type)
        else:
            sam_type = self.cb_sam_type.currentText()

        model_file = config.get("model_file", "sam_b.pt")
        model_index = self.cb_sam_model.findData(model_file)
        if model_index >= 0:
            self.cb_sam_model.setCurrentIndex(model_index)

        device = str(config.get("device", "cpu"))
        if device == "cpu":
            display_device = "CPU"
        elif device.isdigit():
            display_device = f"CUDA:{device}"
        else:
            display_device = "自动选择"
        device_index = self.cb_sam_device.findText(display_device)
        if device_index >= 0:
            self.cb_sam_device.setCurrentIndex(device_index)

        self.sb_sam_imgsz.setValue(float(config.get("imgsz", 1024)))
        self.sb_sam_conf.setValue(float(config.get("conf", 0.4)))
        self.sb_sam_iou.setValue(float(config.get("iou", 0.9)))
        self.chk_sam_retina.setChecked(bool(config.get("retina_masks", True)))

    def save_sam_config(self) -> bool:
        """保存SAM配置到配置文件。"""
        config = self.get_sam_config()
        config_dir = os.path.dirname(SAM_CONFIG_FILE)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        try:
            with open(SAM_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    @classmethod
    def get_saved_sam_config(cls) -> dict:
        """读取已保存的SAM配置（不依赖弹窗实例）。"""
        config = DEFAULT_SAM_CONFIG.copy()
        if os.path.exists(SAM_CONFIG_FILE):
            try:
                with open(SAM_CONFIG_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    if isinstance(saved, dict):
                        config.update(saved)
            except Exception:
                pass
        return config
    
    # ==================== LLM相关方法 ====================
    
    def create_llm_tab(self) -> QWidget:
        """创建LLM自动标注页面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # API配置组
        api_group = self.create_llm_api_group()
        layout.addWidget(api_group)
        
        # 提示词配置组
        prompt_group = self.create_llm_prompt_group()
        layout.addWidget(prompt_group)
        
        layout.addStretch()
        return tab
    
    def create_llm_api_group(self) -> QGroupBox:
        """创建LLM API配置组"""
        group = QGroupBox("API配置")
        group.setStyleSheet(self.get_group_style())
        
        layout = QFormLayout(group)
        
        # API Key
        self.le_llm_api_key = QLineEdit()
        self.le_llm_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.le_llm_api_key.setPlaceholderText("请输入API Key")
        layout.addRow("API Key:", self.le_llm_api_key)
        
        # Base URL
        self.le_llm_base_url = QLineEdit()
        self.le_llm_base_url.setPlaceholderText("请输入Base URL")
        layout.addRow("Base URL:", self.le_llm_base_url)
        
        # Model Name
        self.le_llm_model_name = QLineEdit()
        self.le_llm_model_name.setPlaceholderText("请输入模型名称")
        layout.addRow("模型名称:", self.le_llm_model_name)
        
        return group
    
    def create_llm_prompt_group(self) -> QGroupBox:
        """创建LLM提示词配置组"""
        group = QGroupBox("提示词配置")
        group.setStyleSheet(self.get_group_style())
        
        layout = QVBoxLayout(group)
        
        # 系统提示词
        layout.addWidget(QLabel("系统提示词:"))
        self.te_llm_system_prompt = QTextEdit()
        self.te_llm_system_prompt.setMaximumHeight(120)
        layout.addWidget(self.te_llm_system_prompt)
        
        # 用户提示词
        layout.addWidget(QLabel("用户提示词 (使用 {target} 作为类别占位符):"))
        self.te_llm_user_prompt = QTextEdit()
        self.te_llm_user_prompt.setMaximumHeight(120)
        layout.addWidget(self.te_llm_user_prompt)
        
        return group
    
    def load_llm_config(self):
        """加载LLM配置"""
        config = DEFAULT_LLM_CONFIG.copy()
        
        # 从文件加载配置
        if os.path.exists(LLM_CONFIG_FILE):
            try:
                with open(LLM_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                    config.update(saved_config)
            except Exception as e:
                print(f"加载LLM配置失败: {e}")
        
        # 设置到UI
        self.le_llm_api_key.setText(config.get('api_key', ''))
        self.le_llm_base_url.setText(config.get('base_url', ''))
        self.le_llm_model_name.setText(config.get('model_name', ''))
        self.te_llm_system_prompt.setPlainText(config.get('system_prompt', ''))
        self.te_llm_user_prompt.setPlainText(config.get('user_prompt', ''))
    
    def save_llm_config(self):
        """保存LLM配置"""
        config = {
            'api_key': self.le_llm_api_key.text(),
            'base_url': self.le_llm_base_url.text(),
            'model_name': self.le_llm_model_name.text(),
            'system_prompt': self.te_llm_system_prompt.toPlainText(),
            'user_prompt': self.te_llm_user_prompt.toPlainText()
        }
        
        # 确保配置目录存在
        config_dir = os.path.dirname(LLM_CONFIG_FILE)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        
        # 保存到文件
        try:
            with open(LLM_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存LLM配置失败: {e}")
            return False
    
    def get_llm_config(self) -> dict:
        """获取LLM配置"""
        return {
            'api_key': self.le_llm_api_key.text(),
            'base_url': self.le_llm_base_url.text(),
            'model_name': self.le_llm_model_name.text(),
            'system_prompt': self.te_llm_system_prompt.toPlainText(),
            'user_prompt': self.te_llm_user_prompt.toPlainText()
        }
