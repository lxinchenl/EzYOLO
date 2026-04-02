# -*- coding: utf-8 -*-
"""
训练结果页面
显示runs文件夹中的训练结果图
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, 
    QListWidgetItem, QPushButton, QSplitter, QScrollArea,
    QGroupBox, QFormLayout, QGridLayout, QMessageBox, QFileDialog,
    QDialog, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QLineEdit,
    QTabWidget, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import os
import cv2
from typing import List, Dict

from gui.styles import COLORS


class ONNXExportDialog(QDialog):
    """ONNX导出配置对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ONNX导出配置")
        self.setMinimumWidth(400)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 基本参数
        basic_group = QGroupBox("基本参数")
        basic_layout = QFormLayout(basic_group)
        
        # imgsz
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(32, 4096)
        self.spin_imgsz.setValue(640)
        self.spin_imgsz.setSingleStep(32)
        basic_layout.addRow("输入尺寸 (imgsz):", self.spin_imgsz)
        
        # batch
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 64)
        self.spin_batch.setValue(1)
        basic_layout.addRow("批量大小 (batch):", self.spin_batch)
        
        # opset
        self.spin_opset = QSpinBox()
        self.spin_opset.setRange(7, 17)
        self.spin_opset.setValue(12)
        basic_layout.addRow("ONNX Opset:", self.spin_opset)
        
        # device
        self.combo_device = QComboBox()
        self.combo_device.addItems(["cpu", "0", "1", "2", "3"])
        basic_layout.addRow("设备 (device):", self.combo_device)
        
        layout.addWidget(basic_group)
        
        # 选项参数
        options_group = QGroupBox("选项")
        options_layout = QVBoxLayout(options_group)
        
        self.chk_half = QCheckBox("半精度 (half) - FP16")
        self.chk_half.setChecked(False)
        options_layout.addWidget(self.chk_half)
        
        self.chk_dynamic = QCheckBox("动态轴 (dynamic)")
        self.chk_dynamic.setChecked(False)
        options_layout.addWidget(self.chk_dynamic)
        
        self.chk_simplify = QCheckBox("简化模型 (simplify)")
        self.chk_simplify.setChecked(True)
        options_layout.addWidget(self.chk_simplify)
        
        self.chk_nms = QCheckBox("包含NMS (nms)")
        self.chk_nms.setChecked(False)
        options_layout.addWidget(self.chk_nms)
        
        layout.addWidget(options_group)
        
        # NMS参数组（仅在NMS选中时启用）
        self.nms_group = QGroupBox("NMS参数")
        self.nms_group.setEnabled(False)
        nms_layout = QFormLayout(self.nms_group)
        
        # conf
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setValue(0.25)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setSingleStep(0.05)
        nms_layout.addRow("置信度阈值 (conf):", self.spin_conf)
        
        # iou
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.1, 1.0)
        self.spin_iou.setValue(0.45)
        self.spin_iou.setDecimals(2)
        self.spin_iou.setSingleStep(0.05)
        nms_layout.addRow("IoU阈值 (iou):", self.spin_iou)
        
        # agnostic_nms
        self.chk_agnostic_nms = QCheckBox("类别无关NMS (agnostic_nms)")
        self.chk_agnostic_nms.setChecked(False)
        nms_layout.addRow(self.chk_agnostic_nms)
        
        layout.addWidget(self.nms_group)
        
        # NMS选中时启用NMS参数组
        self.chk_nms.stateChanged.connect(
            lambda state: self.nms_group.setEnabled(state == Qt.CheckState.Checked.value)
        )
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("确定")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
    
    def get_config(self) -> dict:
        """获取配置"""
        config = {
            'imgsz': self.spin_imgsz.value(),
            'half': self.chk_half.isChecked(),
            'dynamic': self.chk_dynamic.isChecked(),
            'simplify': self.chk_simplify.isChecked(),
            'opset': self.spin_opset.value(),
            'nms': self.chk_nms.isChecked(),
            'batch': self.spin_batch.value(),
            'device': self.combo_device.currentText()
        }
        
        # 如果启用NMS，添加NMS参数
        if self.chk_nms.isChecked():
            config['conf'] = self.spin_conf.value()
            config['iou'] = self.spin_iou.value()
            config['agnostic_nms'] = self.chk_agnostic_nms.isChecked()
        
        return config


class TensorRTExportDialog(QDialog):
    """TensorRT导出配置对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TensorRT导出配置")
        self.setMinimumWidth(400)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 基本参数
        basic_group = QGroupBox("基本参数")
        basic_layout = QFormLayout(basic_group)
        
        # imgsz
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(32, 4096)
        self.spin_imgsz.setValue(640)
        self.spin_imgsz.setSingleStep(32)
        basic_layout.addRow("输入尺寸 (imgsz):", self.spin_imgsz)
        
        # batch
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 64)
        self.spin_batch.setValue(1)
        basic_layout.addRow("批量大小 (batch):", self.spin_batch)
        
        # workspace
        self.spin_workspace = QSpinBox()
        self.spin_workspace.setRange(1, 16)
        self.spin_workspace.setValue(4)
        basic_layout.addRow("工作空间 (workspace GB):", self.spin_workspace)
        
        # device
        self.combo_device = QComboBox()
        self.combo_device.addItems(["0", "1", "2", "3"])
        basic_layout.addRow("设备 (device):", self.combo_device)
        
        layout.addWidget(basic_group)
        
        # 选项参数
        options_group = QGroupBox("选项")
        options_layout = QVBoxLayout(options_group)
        
        self.chk_half = QCheckBox("半精度 (half) - FP16")
        self.chk_half.setChecked(True)
        options_layout.addWidget(self.chk_half)
        
        self.chk_int8 = QCheckBox("INT8量化 (int8)")
        self.chk_int8.setChecked(False)
        options_layout.addWidget(self.chk_int8)
        
        self.chk_dynamic = QCheckBox("动态轴 (dynamic)")
        self.chk_dynamic.setChecked(False)
        options_layout.addWidget(self.chk_dynamic)
        
        self.chk_simplify = QCheckBox("简化模型 (simplify)")
        self.chk_simplify.setChecked(True)
        options_layout.addWidget(self.chk_simplify)
        
        self.chk_nms = QCheckBox("包含NMS (nms)")
        self.chk_nms.setChecked(False)
        options_layout.addWidget(self.chk_nms)
        
        layout.addWidget(options_group)
        
        # NMS参数组（仅在NMS选中时启用）
        self.nms_group = QGroupBox("NMS参数")
        self.nms_group.setEnabled(False)
        nms_layout = QFormLayout(self.nms_group)
        
        # conf
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setValue(0.25)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setSingleStep(0.05)
        nms_layout.addRow("置信度阈值 (conf):", self.spin_conf)
        
        # iou
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.1, 1.0)
        self.spin_iou.setValue(0.45)
        self.spin_iou.setDecimals(2)
        self.spin_iou.setSingleStep(0.05)
        nms_layout.addRow("IoU阈值 (iou):", self.spin_iou)
        
        # agnostic_nms
        self.chk_agnostic_nms = QCheckBox("类别无关NMS (agnostic_nms)")
        self.chk_agnostic_nms.setChecked(False)
        nms_layout.addRow(self.chk_agnostic_nms)
        
        layout.addWidget(self.nms_group)
        
        # NMS选中时启用NMS参数组
        self.chk_nms.stateChanged.connect(
            lambda state: self.nms_group.setEnabled(state == Qt.CheckState.Checked.value)
        )
        
        # INT8校准参数（仅在INT8选中时启用）
        self.int8_group = QGroupBox("INT8校准参数")
        self.int8_group.setEnabled(False)
        int8_layout = QFormLayout(self.int8_group)
        
        self.edit_data = QLineEdit()
        self.edit_data.setPlaceholderText("校准数据集路径（如：coco128.yaml）")
        int8_layout.addRow("校准数据 (data):", self.edit_data)
        
        self.spin_fraction = QDoubleSpinBox()
        self.spin_fraction.setRange(0.1, 1.0)
        self.spin_fraction.setValue(1.0)
        self.spin_fraction.setSingleStep(0.1)
        int8_layout.addRow("数据比例 (fraction):", self.spin_fraction)
        
        layout.addWidget(self.int8_group)
        
        # INT8选中时启用校准参数
        self.chk_int8.stateChanged.connect(
            lambda state: self.int8_group.setEnabled(state == Qt.CheckState.Checked.value)
        )
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("确定")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
    
    def get_config(self) -> dict:
        """获取配置"""
        config = {
            'imgsz': self.spin_imgsz.value(),
            'half': self.chk_half.isChecked(),
            'dynamic': self.chk_dynamic.isChecked(),
            'simplify': self.chk_simplify.isChecked(),
            'workspace': self.spin_workspace.value(),
            'int8': self.chk_int8.isChecked(),
            'nms': self.chk_nms.isChecked(),
            'batch': self.spin_batch.value(),
            'device': self.combo_device.currentText()
        }
        
        # 如果启用NMS，添加NMS参数
        if self.chk_nms.isChecked():
            config['conf'] = self.spin_conf.value()
            config['iou'] = self.spin_iou.value()
            config['agnostic_nms'] = self.chk_agnostic_nms.isChecked()
        
        # 如果启用INT8，添加校准参数
        if self.chk_int8.isChecked():
            config['data'] = self.edit_data.text() or None
            config['fraction'] = self.spin_fraction.value()
        
        return config


class ResultPage(QWidget):
    """训练结果页面"""
    
    def __init__(self):
        super().__init__()
        self.current_project_id = None
        self.current_images = []
        self.current_image_idx = -1
        
        self.init_ui()
        # 初始时不自动扫描，等待设置项目
    
    def set_project(self, project_id: int):
        """设置当前项目"""
        self.current_project_id = project_id
        if project_id:
            print(f"[ResultPage] 已切换到项目: {project_id}")
            self.scan_runs_directory()
        else:
            print("[ResultPage] 项目已取消选择")
            # 清空显示
            self.image_list.clear()
            self.current_images = []
            self.image_label.setText("请先选择一个项目")
            # 清空指标
            for label in self.metric_labels.values():
                label.setText("--")
    
    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        
        # 标题
        title = QLabel("训练结果")
        title.setObjectName("title")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 8px;")
        main_layout.addWidget(title)
        
        # 主分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：图像列表
        left_panel = self.create_image_list_panel()
        splitter.addWidget(left_panel)
        
        # 右侧：图像显示
        right_panel = self.create_image_display_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([300, 900])
        main_layout.addWidget(splitter)
        
        # 底部：指标和导出
        bottom_panel = self.create_metrics_export_panel()
        main_layout.addWidget(bottom_panel)
    
    def create_image_list_panel(self) -> QWidget:
        """创建图像列表面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 标题
        list_title = QLabel("结果图像")
        list_title.setStyleSheet("font-weight: bold; margin-bottom: 8px;")
        layout.addWidget(list_title)
        
        # 刷新按钮
        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.clicked.connect(self.scan_runs_directory)
        layout.addWidget(refresh_btn)
        
        # 图像列表
        self.image_list = QListWidget()
        self.image_list.setObjectName("image_list")
        
        # 简单样式
        self.image_list.setStyleSheet('''
            QListWidget {
                background-color: #252526;
                border: 1px solid #3e3e42;
                border-radius: 6px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #007ACC;
                color: white;
            }
        ''')
        
        self.image_list.itemClicked.connect(self.on_image_selected)
        layout.addWidget(self.image_list)
        
        return panel
    
    def create_image_display_panel(self) -> QWidget:
        """创建图像显示面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 标题
        display_title = QLabel("图像预览")
        display_title.setStyleSheet("font-weight: bold; margin-bottom: 8px;")
        layout.addWidget(display_title)
        
        # 图像容器
        self.image_container = QWidget()
        self.image_container.setObjectName("image_container")
        
        self.image_container.setStyleSheet('''
            QWidget {
                background-color: #252526;
                border: 1px solid #3e3e42;
                border-radius: 6px;
                min-height: 400px;
            }
        ''')
        
        self.image_layout = QVBoxLayout(self.image_container)
        self.image_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 预览标签
        self.image_label = QLabel("选择左侧图像查看预览")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("color: #858585;")
        self.image_layout.addWidget(self.image_label)
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_container)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        return panel
    
    def create_metrics_export_panel(self) -> QWidget:
        """创建指标和导出面板"""
        panel = QGroupBox("训练指标与导出")
        
        panel.setStyleSheet('''
            QGroupBox {
                border: 1px solid #3e3e42;
                border-radius: 6px;
                padding: 10px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        ''')
        
        layout = QVBoxLayout(panel)
        
        # 指标网格
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(10)
        
        # 指标标签
        metrics = ['mAP50', 'mAP50_95', '精确率', '召回率']
        self.metric_labels = {}
        
        for i, metric in enumerate(metrics):
            label = QLabel(metric + ':')
            value_label = QLabel('--')
            value_label.setObjectName(f"metric_{metric.lower()}")
            value_label.setStyleSheet("font-weight: bold;")
            self.metric_labels[metric] = value_label
            
            metrics_grid.addWidget(label, i, 0)
            metrics_grid.addWidget(value_label, i, 1)
        
        layout.addLayout(metrics_grid)
        
        # 导出按钮
        export_layout = QHBoxLayout()
        
        self.btn_export_onnx = QPushButton("导出为ONNX")
        self.btn_export_onnx.clicked.connect(lambda: self.export_model('onnx'))
        export_layout.addWidget(self.btn_export_onnx)
        
        self.btn_export_tensorrt = QPushButton("导出为TensorRT")
        self.btn_export_tensorrt.clicked.connect(lambda: self.export_model('tensorrt'))
        export_layout.addWidget(self.btn_export_tensorrt)
        
        self.btn_export_torchscript = QPushButton("导出为TorchScript")
        self.btn_export_torchscript.clicked.connect(lambda: self.export_model('torchscript'))
        export_layout.addWidget(self.btn_export_torchscript)
        
        self.btn_export_pt = QPushButton("导出为pt")
        self.btn_export_pt.clicked.connect(self.export_model_pt)
        export_layout.addWidget(self.btn_export_pt)
        
        # 导出结果文件夹按钮（使用不同颜色）
        self.btn_export_folder = QPushButton("导出结果文件夹")
        self.btn_export_folder.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                font-weight: bold;
            }}
        """)
        self.btn_export_folder.clicked.connect(self.export_result_folder)
        export_layout.addWidget(self.btn_export_folder)
        
        export_layout.addStretch()
        layout.addLayout(export_layout)
        
        return panel
    
    def scan_runs_directory(self):
        """扫描runs目录"""
        self.image_list.clear()
        
        # 递归搜索runs目录下的所有训练结果文件夹
        runs_dir = "runs"
        
        if not os.path.exists(runs_dir):
            QMessageBox.warning(self, "提示", "runs目录不存在")
            return
        
        # 递归查找所有包含weights文件夹的目录（这些是训练结果目录）
        projects = []
        for root, dirs, files in os.walk(runs_dir):
            if 'weights' in dirs:
                # 提取目录名（相对于runs目录）
                rel_path = os.path.relpath(root, runs_dir)
                projects.append(rel_path)
        
        # 过滤出与当前项目对应的结果（格式为exp_项目ID）
        project_runs = []
        for project in projects:
            if project.startswith(f"exp_{self.current_project_id}"):
                project_runs.append(project)
        
        # 如果没有找到对应项目的结果，查找包含项目ID的文件夹
        if not project_runs:
            for project in projects:
                if str(self.current_project_id) in project:
                    project_runs.append(project)
        
        # 如果还是没有，显示所有结果但标记为非对应
        if not project_runs:
            project_runs = projects
            QMessageBox.information(self, "提示", f"未找到与项目 {self.current_project_id} 对应的训练结果，显示所有可用结果")
        
        if not project_runs:
            QMessageBox.warning(self, "提示", "没有找到训练项目")
            return
        
        # 选择最新的项目结果
        latest_project = sorted(project_runs)[-1]
        project_dir = os.path.join(runs_dir, latest_project)
        
        # 找到所有图像文件
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_files = []
        
        for root, _, files in os.walk(project_dir):
            for file in files:
                if any(file.endswith(ext) for ext in image_extensions):
                    image_files.append((file, os.path.join(root, file)))
        
        # 清空列表并添加图像
        self.image_list.clear()
        self.current_images = image_files
        
        for filename, filepath in image_files:
            item = QListWidgetItem(filename)
            item.setData(Qt.ItemDataRole.UserRole, filepath)
            self.image_list.addItem(item)
        
        # 更新标题
        self.setWindowTitle(f"训练结果 - 项目 {self.current_project_id} - {latest_project}")
        
        # 读取训练指标
        self.read_training_metrics(project_dir)
    
    def on_image_selected(self, item):
        """选择图像"""
        filepath = item.data(Qt.ItemDataRole.UserRole)
        if filepath and os.path.exists(filepath):
            self.display_image(filepath)
    
    def display_image(self, filepath):
        """显示图像"""
        try:
            # 使用OpenCV读取图像
            img = cv2.imread(filepath)
            if img is None:
                self.image_label.setText("无法加载图像")
                return
            
            # 转换为RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channel = img.shape
            
            # 创建QImage
            qimg = QImage(img.data, width, height, width * channel, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # 调整大小以适应容器
            container_width = self.image_container.width() - 40
            container_height = self.image_container.height() - 40
            
            if pixmap.width() > container_width or pixmap.height() > container_height:
                pixmap = pixmap.scaled(
                    container_width, container_height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            
            # 更新标签
            self.image_label.setPixmap(pixmap)
            self.image_label.setText("")
            
        except Exception as e:
            self.image_label.setText(f"加载失败: {str(e)}")
    
    def export_model(self, format_type):
        """导出模型"""
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return
        
        runs_dir = "runs"
        
        if not os.path.exists(runs_dir):
            QMessageBox.warning(self, "提示", "runs目录不存在")
            return
        
        # 递归查找所有包含weights文件夹的目录
        projects = []
        for root, dirs, files in os.walk(runs_dir):
            if 'weights' in dirs:
                rel_path = os.path.relpath(root, runs_dir)
                projects.append(rel_path)
        
        # 过滤出与当前项目对应的结果
        project_runs = []
        for project in projects:
            if project.startswith(f"exp_{self.current_project_id}") or str(self.current_project_id) in project:
                project_runs.append(project)
        
        if not project_runs:
            project_runs = projects
        
        if not project_runs:
            QMessageBox.warning(self, "提示", "没有找到训练项目")
            return
        
        latest_project = sorted(project_runs)[-1]
        project_dir = os.path.join(runs_dir, latest_project)
        
        # 找到best.pt文件
        best_model = os.path.join(project_dir, "weights", "best.pt")
        if not os.path.exists(best_model):
            QMessageBox.warning(self, "提示", "未找到best.pt模型文件")
            return
        
        # 显示导出配置对话框
        export_config = {}
        if format_type == 'onnx':
            dialog = ONNXExportDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            export_config = dialog.get_config()
        elif format_type == 'tensorrt':
            dialog = TensorRTExportDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            export_config = dialog.get_config()
        
        # 选择导出路径
        file_filter = ""
        if format_type == 'onnx':
            file_filter = "ONNX files (*.onnx)"
        elif format_type == 'tensorrt':
            file_filter = "TensorRT files (*.engine)"
        elif format_type == 'torchscript':
            file_filter = "TorchScript files (*.pt)"
        
        default_name = f"best.{format_type}"
        if format_type == 'tensorrt':
            default_name = "best.engine"
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, f"导出为{format_type.upper()}", 
            os.path.join(project_dir, default_name),
            file_filter
        )
        
        if not save_path:
            return
        
        try:
            from ultralytics import YOLO
            
            # 加载模型
            model = YOLO(best_model)
            
            # 构建导出参数（Ultralytics 的 TensorRT 导出格式名是 engine）
            export_format = 'engine' if format_type == 'tensorrt' else format_type
            export_kwargs = {'format': export_format}
            
            # 添加配置参数（针对ONNX和TensorRT）
            if format_type in ['onnx', 'tensorrt']:
                export_kwargs.update(export_config)
                # 过滤掉None值
                export_kwargs = {k: v for k, v in export_kwargs.items() if v is not None}
            
            print(f"[导出] 参数: {export_kwargs}")
            
            # 导出
            model.export(**export_kwargs)
            
            # 移动文件
            if format_type == 'onnx':
                export_path = os.path.join(project_dir, "weights", "best.onnx")
            elif format_type == 'tensorrt':
                export_path = os.path.join(project_dir, "weights", "best.engine")
            elif format_type == 'torchscript':
                export_path = os.path.join(project_dir, "weights", "best.torchscript.pt")
            
            if os.path.exists(export_path):
                import shutil
                shutil.move(export_path, save_path)
                QMessageBox.information(self, "成功", f"模型已导出为 {save_path}")
            else:
                QMessageBox.warning(self, "失败", "导出失败，请检查日志")
                
        except Exception as e:
            QMessageBox.warning(self, "错误", f"导出出错: {str(e)}")
    
    def export_model_pt(self):
        """导出为pt文件"""
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return
        
        runs_dir = "runs"
        
        if not os.path.exists(runs_dir):
            QMessageBox.warning(self, "提示", "runs目录不存在")
            return
        
        # 递归查找所有包含weights文件夹的目录
        projects = []
        for root, dirs, files in os.walk(runs_dir):
            if 'weights' in dirs:
                rel_path = os.path.relpath(root, runs_dir)
                projects.append(rel_path)
        
        # 过滤出与当前项目对应的结果
        project_runs = []
        for project in projects:
            if project.startswith(f"exp_{self.current_project_id}") or str(self.current_project_id) in project:
                project_runs.append(project)
        
        if not project_runs:
            project_runs = projects
        
        if not project_runs:
            QMessageBox.warning(self, "提示", "没有找到训练项目")
            return
        
        latest_project = sorted(project_runs)[-1]
        project_dir = os.path.join(runs_dir, latest_project)
        
        # 找到best.pt文件
        best_model = os.path.join(project_dir, "weights", "best.pt")
        if not os.path.exists(best_model):
            QMessageBox.warning(self, "提示", "未找到best.pt模型文件")
            return
        
        # 选择导出路径
        save_path, _ = QFileDialog.getSaveFileName(
            self, "导出为pt", 
            os.path.join(project_dir, "best.pt"),
            "PyTorch files (*.pt)"
        )
        
        if not save_path:
            return
        
        try:
            import shutil
            # 复制文件
            shutil.copy2(best_model, save_path)
            QMessageBox.information(self, "成功", f"模型已导出为 {save_path}")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"导出出错: {str(e)}")
    
    def export_result_folder(self):
        """导出结果文件夹"""
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return
        
        runs_dir = "runs"
        
        if not os.path.exists(runs_dir):
            QMessageBox.warning(self, "提示", "runs目录不存在")
            return
        
        # 递归查找所有包含weights文件夹的目录
        projects = []
        for root, dirs, files in os.walk(runs_dir):
            if 'weights' in dirs:
                rel_path = os.path.relpath(root, runs_dir)
                projects.append(rel_path)
        
        # 过滤出与当前项目对应的结果
        project_runs = []
        for project in projects:
            if project.startswith(f"exp_{self.current_project_id}") or str(self.current_project_id) in project:
                project_runs.append(project)
        
        if not project_runs:
            project_runs = projects
        
        if not project_runs:
            QMessageBox.warning(self, "提示", "没有找到训练项目")
            return
        
        latest_project = sorted(project_runs)[-1]
        project_dir = os.path.join(runs_dir, latest_project)
        
        # 选择目标文件夹
        save_dir = QFileDialog.getExistingDirectory(
            self, "选择导出目标文件夹", 
            os.path.dirname(project_dir)
        )
        
        if not save_dir:
            return
        
        try:
            import shutil
            # 目标文件夹路径
            target_dir = os.path.join(save_dir, latest_project)
            
            # 如果目标文件夹已存在，删除它
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            
            # 复制整个文件夹
            shutil.copytree(project_dir, target_dir)
            QMessageBox.information(self, "成功", f"结果文件夹已导出到 {target_dir}")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"导出出错: {str(e)}")
    
    def update_metrics(self, metrics: Dict):
        """更新指标"""
        for metric, value in metrics.items():
            if metric in self.metric_labels:
                self.metric_labels[metric].setText(f"{value:.4f}")
    
    def read_training_metrics(self, project_dir):
        """从results.csv读取训练指标"""
        results_csv = os.path.join(project_dir, "results.csv")
        
        if not os.path.exists(results_csv):
            # 没有results.csv文件
            return
        
        try:
            import csv
            
            with open(results_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if not rows:
                    return
                
                # 读取最后一行数据
                last_row = rows[-1]
                
                # 提取指标
                precision = float(last_row.get('metrics/precision(B)', 0))
                recall = float(last_row.get('metrics/recall(B)', 0))
                map50 = float(last_row.get('metrics/mAP50(B)', 0))
                map50_95 = float(last_row.get('metrics/mAP50-95(B)', 0))
                
                # 更新指标显示
                if 'mAP50' in self.metric_labels:
                    self.metric_labels['mAP50'].setText(f"{map50:.4f}")
                if 'mAP50_95' in self.metric_labels:
                    self.metric_labels['mAP50_95'].setText(f"{map50_95:.4f}")
                if '精确率' in self.metric_labels:
                    self.metric_labels['精确率'].setText(f"{precision:.4f}")
                if '召回率' in self.metric_labels:
                    self.metric_labels['召回率'].setText(f"{recall:.4f}")
                    
        except Exception as e:
            print(f"读取指标失败: {e}")


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    window = ResultPage()
    window.show()
    sys.exit(app.exec())
