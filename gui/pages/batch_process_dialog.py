# -*- coding: utf-8 -*-
"""
批量处理标注对话框
支持批量删除和批量修改标注类别
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QSpinBox, QComboBox, QListWidget,
    QListWidgetItem, QRadioButton, QButtonGroup, QMessageBox,
    QSplitter, QWidget, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
import os
from typing import List, Dict, Tuple, Optional

from gui.styles import COLORS


class BatchProcessDialog(QDialog):
    """批量处理标注对话框"""
    
    # 信号定义
    process_requested = pyqtSignal(dict)  # 发送处理请求
    
    def __init__(self, parent=None, project_classes=None, total_images=0):
        super().__init__(parent)
        self.setWindowTitle("批量处理标注")
        
        # 获取屏幕分辨率并设置对话框大小
        from PyQt6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        
        # 设置对话框大小为屏幕高度的90%，宽度600
        dialog_height = int(screen_height * 0.9)
        self.setMinimumSize(600, dialog_height)
        self.resize(600, dialog_height)
        
        # 项目类别
        self.project_classes = project_classes or []
        self.total_images = total_images
        
        # 选择的像素点
        self.selected_points = []
        
        # 初始化UI
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        
        # 标题
        title = QLabel("批量处理标注")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        main_layout.addWidget(title)
        
        # 说明文字
        desc = QLabel("1. 在图片上点击选择像素点（可多选）\n2. 设置处理范围\n3. 选择操作类型和类别\n4. 点击执行")
        desc.setStyleSheet("color: gray; margin-bottom: 10px;")
        main_layout.addWidget(desc)
        
        # 步骤1：像素点选择状态
        self.point_group = QGroupBox("步骤1：像素点选择")
        self.point_group.setStyleSheet(self.get_group_style())
        point_layout = QVBoxLayout(self.point_group)
        
        self.point_status = QLabel("未选择像素点")
        self.point_status.setStyleSheet("color: orange;")
        point_layout.addWidget(self.point_status)
        
        self.selected_points_list = QListWidget()
        self.selected_points_list.setMaximumHeight(100)
        point_layout.addWidget(self.selected_points_list)
        
        btn_layout = QHBoxLayout()
        self.btn_clear_points = QPushButton("清除所有点")
        self.btn_clear_points.clicked.connect(self.clear_points)
        self.btn_clear_points.setEnabled(False)
        btn_layout.addWidget(self.btn_clear_points)
        btn_layout.addStretch()
        point_layout.addLayout(btn_layout)
        
        main_layout.addWidget(self.point_group)
        
        # 步骤2：处理范围
        range_group = QGroupBox("步骤2：处理范围")
        range_group.setStyleSheet(self.get_group_style())
        range_layout = QFormLayout(range_group)
        
        self.start_image = QSpinBox()
        self.start_image.setRange(1, max(1, self.total_images))
        self.start_image.setValue(1)
        range_layout.addRow("起始图片:", self.start_image)
        
        self.end_image = QSpinBox()
        self.end_image.setRange(1, max(1, self.total_images))
        self.end_image.setValue(self.total_images)
        range_layout.addRow("结束图片:", self.end_image)
        
        range_info = QLabel(f"共 {self.total_images} 张图片")
        range_info.setStyleSheet("color: gray;")
        range_layout.addRow(range_info)
        
        main_layout.addWidget(range_group)
        
        # 步骤3：操作类型
        op_group = QGroupBox("步骤3：操作类型")
        op_group.setStyleSheet(self.get_group_style())
        op_layout = QVBoxLayout(op_group)
        
        self.op_group = QButtonGroup(self)
        
        self.rbtn_delete = QRadioButton("批量删除")
        self.rbtn_delete.setChecked(True)
        self.rbtn_delete.toggled.connect(self.on_operation_changed)
        self.rbtn_delete.setStyleSheet("""
            QRadioButton {
                color: white;
                font-size: 14px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border-radius: 8px;
                border: 2px solid white;
                background-color: transparent;
            }
            QRadioButton::indicator:checked {
                background-color: white;
                border: 2px solid white;
            }
            QRadioButton::indicator:unchecked {
                background-color: transparent;
                border: 2px solid white;
            }
        """)
        self.op_group.addButton(self.rbtn_delete)
        op_layout.addWidget(self.rbtn_delete)
        
        self.rbtn_modify = QRadioButton("批量修改类别")
        self.rbtn_modify.toggled.connect(self.on_operation_changed)
        self.rbtn_modify.setStyleSheet("""
            QRadioButton {
                color: white;
                font-size: 14px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border-radius: 8px;
                border: 2px solid white;
                background-color: transparent;
            }
            QRadioButton::indicator:checked {
                background-color: white;
                border: 2px solid white;
            }
            QRadioButton::indicator:unchecked {
                background-color: transparent;
                border: 2px solid white;
            }
        """)
        self.op_group.addButton(self.rbtn_modify)
        op_layout.addWidget(self.rbtn_modify)
        
        main_layout.addWidget(op_group)
        
        # 步骤4：类别选择
        self.class_group = QGroupBox("步骤4：类别选择")
        self.class_group.setStyleSheet(self.get_group_style())
        self.class_layout = QVBoxLayout(self.class_group)
        
        # 根据操作类型动态显示不同的类别选择界面
        self.class_stack = QWidget()
        self.class_stack_layout = QVBoxLayout(self.class_stack)
        self.class_stack_layout.setContentsMargins(0, 0, 0, 0)
        
        # 删除模式：多选要删除的类别
        self.delete_class_widget = QWidget()
        delete_class_layout = QVBoxLayout(self.delete_class_widget)
        delete_class_layout.setContentsMargins(0, 0, 0, 0)
        
        delete_label = QLabel("选择要删除的类别（可多选）:")
        delete_class_layout.addWidget(delete_label)
        
        self.delete_class_list = QListWidget()
        self.delete_class_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.populate_class_list(self.delete_class_list)
        delete_class_layout.addWidget(self.delete_class_list)
        
        self.class_stack_layout.addWidget(self.delete_class_widget)
        
        # 修改模式：多选初始类别，单选目标类别
        self.modify_class_widget = QWidget()
        modify_class_layout = QVBoxLayout(self.modify_class_widget)
        modify_class_layout.setContentsMargins(0, 0, 0, 0)
        
        # 初始类别
        source_layout = QVBoxLayout()
        source_label = QLabel("选择初始类别（可多选）:")
        source_layout.addWidget(source_label)
        
        self.source_class_list = QListWidget()
        self.source_class_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.populate_class_list(self.source_class_list)
        source_layout.addWidget(self.source_class_list)
        
        modify_class_layout.addLayout(source_layout)
        
        # 目标类别
        target_layout = QVBoxLayout()
        target_label = QLabel("选择目标类别（单选）:")
        target_layout.addWidget(target_label)
        
        self.target_class_list = QListWidget()
        self.target_class_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.populate_class_list(self.target_class_list)
        target_layout.addWidget(self.target_class_list)
        
        modify_class_layout.addLayout(target_layout)
        
        self.class_stack_layout.addWidget(self.modify_class_widget)
        self.modify_class_widget.hide()
        
        self.class_layout.addWidget(self.class_stack)
        main_layout.addWidget(self.class_group)
        
        # 底部按钮
        button_layout = QHBoxLayout()
        
        self.btn_execute = QPushButton("执行批量处理")
        self.btn_execute.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                font-weight: bold;
                padding: 10px 20px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary']};
            }}
            QPushButton:disabled {{
                background-color: gray;
            }}
        """)
        self.btn_execute.clicked.connect(self.execute_process)
        button_layout.addWidget(self.btn_execute)
        
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_cancel)
        
        main_layout.addLayout(button_layout)
        
        # 初始状态更新
        self.update_execute_button()
    
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
    
    def populate_class_list(self, list_widget: QListWidget):
        """填充类别列表"""
        list_widget.clear()
        for cls in self.project_classes:
            item = QListWidgetItem(f"{cls['id']}: {cls['name']}")
            item.setData(Qt.ItemDataRole.UserRole, cls['id'])
            # 设置颜色
            color = QColor(cls.get('color', '#808080'))
            item.setForeground(color)
            list_widget.addItem(item)
    
    def on_operation_changed(self):
        """操作类型改变时"""
        if self.rbtn_delete.isChecked():
            self.delete_class_widget.show()
            self.modify_class_widget.hide()
        else:
            self.delete_class_widget.hide()
            self.modify_class_widget.show()
    
    def add_point(self, x: int, y: int):
        """添加像素点"""
        self.selected_points.append((x, y))
        self.update_points_display()
        self.update_execute_button()
    
    def clear_points(self):
        """清除所有像素点"""
        self.selected_points.clear()
        self.update_points_display()
        self.update_execute_button()
    
    def update_points_display(self):
        """更新像素点显示"""
        self.selected_points_list.clear()
        for i, (x, y) in enumerate(self.selected_points):
            item = QListWidgetItem(f"点 {i+1}: ({x}, {y})")
            self.selected_points_list.addItem(item)
        
        # 更新状态
        if self.selected_points:
            self.point_status.setText(f"已选择 {len(self.selected_points)} 个像素点")
            self.point_status.setStyleSheet("color: green;")
            self.btn_clear_points.setEnabled(True)
        else:
            self.point_status.setText("未选择像素点")
            self.point_status.setStyleSheet("color: orange;")
            self.btn_clear_points.setEnabled(False)
    
    def update_execute_button(self):
        """更新执行按钮状态"""
        has_points = len(self.selected_points) > 0
        self.btn_execute.setEnabled(has_points)
    
    def execute_process(self):
        """执行批量处理"""
        if not self.selected_points:
            QMessageBox.warning(self, "提示", "请先选择像素点")
            return
        
        # 获取处理范围
        start_idx = self.start_image.value() - 1  # 转换为0-based索引
        end_idx = self.end_image.value() - 1
        
        if start_idx > end_idx:
            QMessageBox.warning(self, "提示", "起始图片不能大于结束图片")
            return
        
        # 构建处理配置
        config = {
            'points': self.selected_points.copy(),
            'start_idx': start_idx,
            'end_idx': end_idx,
            'operation': 'delete' if self.rbtn_delete.isChecked() else 'modify'
        }
        
        if self.rbtn_delete.isChecked():
            # 获取要删除的类别
            selected_items = self.delete_class_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "提示", "请选择要删除的类别")
                return
            config['target_classes'] = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
        else:
            # 获取初始类别和目标类别
            source_items = self.source_class_list.selectedItems()
            if not source_items:
                QMessageBox.warning(self, "提示", "请选择初始类别")
                return
            
            target_items = self.target_class_list.selectedItems()
            if not target_items:
                QMessageBox.warning(self, "提示", "请选择目标类别")
                return
            
            config['source_classes'] = [item.data(Qt.ItemDataRole.UserRole) for item in source_items]
            config['target_class'] = target_items[0].data(Qt.ItemDataRole.UserRole)
        
        # 发送处理请求
        self.process_requested.emit(config)
        self.accept()
    
    def get_selected_points(self) -> List[Tuple[int, int]]:
        """获取选择的像素点"""
        return self.selected_points.copy()
