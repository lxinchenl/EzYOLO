# -*- coding: utf-8 -*-
"""
设置页面
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QFormLayout, QComboBox, QSlider, QSpinBox,
    QFileDialog, QMessageBox, QCheckBox
)
from PyQt6.QtCore import Qt, QSettings, pyqtSignal

from gui.styles import COLORS


class SettingsPage(QWidget):
    """设置页面"""
    
    # 主题变化信号
    theme_changed = pyqtSignal(str)  # 发送新的主题名称
    
    def __init__(self):
        super().__init__()
        self.settings = QSettings("EzYOLO", "Settings")
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        
        # 标题
        title = QLabel("设置")
        title.setObjectName("title")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        main_layout.addWidget(title)
        
        # 主题设置
        theme_group = self.create_theme_group()
        main_layout.addWidget(theme_group)
        
        # 路径设置
        path_group = self.create_path_group()
        main_layout.addWidget(path_group)
        
        # 自动保存设置
        auto_save_group = self.create_auto_save_group()
        main_layout.addWidget(auto_save_group)
        
        # 快捷键设置（预留）
        shortcut_group = self.create_shortcut_group()
        main_layout.addWidget(shortcut_group)
        
        # 底部按钮
        btn_layout = QHBoxLayout()
        
        self.btn_save = QPushButton("💾 保存设置")
        self.btn_save.clicked.connect(self.save_settings)
        btn_layout.addWidget(self.btn_save)
        
        self.btn_reset = QPushButton("🔄 恢复默认")
        self.btn_reset.clicked.connect(self.reset_settings)
        btn_layout.addWidget(self.btn_reset)
        
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)
    
    def create_theme_group(self) -> QGroupBox:
        """创建主题设置组"""
        group = QGroupBox("主题设置")
        
        layout = QFormLayout(group)
        
        # 主题选择（只保留深色主题）
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("深色主题")
        # 固定为深色主题
        self.theme_combo.setCurrentIndex(0)
        # 禁用下拉框，防止用户修改
        self.theme_combo.setEnabled(False)
        layout.addRow("主题:", self.theme_combo)
        
        return group
    
    def create_path_group(self) -> QGroupBox:
        """创建路径设置组"""
        group = QGroupBox("路径设置")
        
        layout = QFormLayout(group)
        
        # 预训练模型路径
        path_layout = QHBoxLayout()
        from pathlib import Path
        app_root = Path(__file__).parent.parent.parent  # 向上三级到EzYOLO根目录
        default_pretrained_path = app_root / "pretrained"
        self.pretrained_path = QLabel(self.settings.value("pretrained_path", str(default_pretrained_path)))
        self.pretrained_path.setStyleSheet("background-color: #252526; padding: 4px; border-radius: 4px;")
        path_layout.addWidget(self.pretrained_path)
        
        btn_browse = QPushButton("浏览")
        btn_browse.clicked.connect(lambda: self.browse_path("pretrained_path", "选择预训练模型目录"))
        path_layout.addWidget(btn_browse)
        
        layout.addRow("预训练模型路径:", path_layout)
        
        return group
    
    def create_auto_save_group(self) -> QGroupBox:
        """创建自动保存设置组"""
        group = QGroupBox("自动保存设置")
        
        layout = QFormLayout(group)
        
        # 自动保存间隔
        self.auto_save_interval = QSpinBox()
        self.auto_save_interval.setRange(1, 60)
        self.auto_save_interval.setValue(int(self.settings.value("auto_save_interval", 5)))
        layout.addRow("自动保存间隔 (分钟):", self.auto_save_interval)
        
        # 启用自动保存
        self.auto_save_enabled = QCheckBox("启用自动保存")
        self.auto_save_enabled.setChecked(self.settings.value("auto_save_enabled", True, type=bool))
        layout.addRow(self.auto_save_enabled)
        
        return group
    
    def create_shortcut_group(self) -> QGroupBox:
        """创建快捷键设置组"""
        group = QGroupBox("快捷键设置")
        
        layout = QFormLayout(group)
        
        # 重置视图快捷键
        shortcut_layout = QHBoxLayout()
        self.reset_view_shortcut = QLabel(self.settings.value("reset_view_shortcut", "R"))
        self.reset_view_shortcut.setStyleSheet("background-color: #252526; padding: 4px; border-radius: 4px;")
        shortcut_layout.addWidget(self.reset_view_shortcut)
        
        btn_set_shortcut = QPushButton("设置")
        btn_set_shortcut.clicked.connect(lambda: self.set_shortcut("reset_view_shortcut", self.reset_view_shortcut))
        shortcut_layout.addWidget(btn_set_shortcut)
        
        layout.addRow("重置视图:", shortcut_layout)
        
        # 矩形工具快捷键
        rect_layout = QHBoxLayout()
        self.rect_tool_shortcut = QLabel(self.settings.value("rect_tool_shortcut", "W"))
        self.rect_tool_shortcut.setStyleSheet("background-color: #252526; padding: 4px; border-radius: 4px;")
        rect_layout.addWidget(self.rect_tool_shortcut)
        
        btn_set_rect = QPushButton("设置")
        btn_set_rect.clicked.connect(lambda: self.set_shortcut("rect_tool_shortcut", self.rect_tool_shortcut))
        rect_layout.addWidget(btn_set_rect)
        
        layout.addRow("矩形工具:", rect_layout)
        
        # 多边形工具快捷键
        poly_layout = QHBoxLayout()
        self.poly_tool_shortcut = QLabel(self.settings.value("poly_tool_shortcut", "P"))
        self.poly_tool_shortcut.setStyleSheet("background-color: #252526; padding: 4px; border-radius: 4px;")
        poly_layout.addWidget(self.poly_tool_shortcut)
        
        btn_set_poly = QPushButton("设置")
        btn_set_poly.clicked.connect(lambda: self.set_shortcut("poly_tool_shortcut", self.poly_tool_shortcut))
        poly_layout.addWidget(btn_set_poly)
        
        layout.addRow("多边形工具:", poly_layout)
        
        # 移动工具快捷键
        move_layout = QHBoxLayout()
        self.move_tool_shortcut = QLabel(self.settings.value("move_tool_shortcut", "V"))
        self.move_tool_shortcut.setStyleSheet("background-color: #252526; padding: 4px; border-radius: 4px;")
        move_layout.addWidget(self.move_tool_shortcut)
        
        btn_set_move = QPushButton("设置")
        btn_set_move.clicked.connect(lambda: self.set_shortcut("move_tool_shortcut", self.move_tool_shortcut))
        move_layout.addWidget(btn_set_move)
        
        layout.addRow("移动工具:", move_layout)
        
        # 上一张图片快捷键
        prev_layout = QHBoxLayout()
        self.prev_image_shortcut = QLabel(self.settings.value("prev_image_shortcut", "A"))
        self.prev_image_shortcut.setStyleSheet("background-color: #252526; padding: 4px; border-radius: 4px;")
        prev_layout.addWidget(self.prev_image_shortcut)
        
        btn_set_prev = QPushButton("设置")
        btn_set_prev.clicked.connect(lambda: self.set_shortcut("prev_image_shortcut", self.prev_image_shortcut))
        prev_layout.addWidget(btn_set_prev)
        
        layout.addRow("上一张图片:", prev_layout)
        
        # 下一张图片快捷键
        next_layout = QHBoxLayout()
        self.next_image_shortcut = QLabel(self.settings.value("next_image_shortcut", "D"))
        self.next_image_shortcut.setStyleSheet("background-color: #252526; padding: 4px; border-radius: 4px;")
        next_layout.addWidget(self.next_image_shortcut)
        
        btn_set_next = QPushButton("设置")
        btn_set_next.clicked.connect(lambda: self.set_shortcut("next_image_shortcut", self.next_image_shortcut))
        next_layout.addWidget(btn_set_next)
        
        layout.addRow("下一张图片:", next_layout)
        
        # 删除标注快捷键
        delete_layout = QHBoxLayout()
        self.delete_shortcut = QLabel(self.settings.value("delete_shortcut", "DELETE"))
        self.delete_shortcut.setStyleSheet("background-color: #252526; padding: 4px; border-radius: 4px;")
        delete_layout.addWidget(self.delete_shortcut)
        
        btn_set_delete = QPushButton("设置")
        btn_set_delete.clicked.connect(lambda: self.set_shortcut("delete_shortcut", self.delete_shortcut))
        delete_layout.addWidget(btn_set_delete)
        
        layout.addRow("删除标注:", delete_layout)
        
        return group
    
    def set_shortcut(self, setting_key: str, label: QLabel):
        """设置快捷键"""
        from PyQt6.QtWidgets import QInputDialog
        
        key, ok = QInputDialog.getText(self, "设置快捷键", f"请输入新的快捷键 (单个字母或数字):")
        if ok and key:
            # 只取第一个字符
            new_key = key.upper()[0]
            self.settings.setValue(setting_key, new_key)
            label.setText(new_key)
            QMessageBox.information(self, "设置成功", f"快捷键已设置为: {new_key}")
    
    def browse_path(self, setting_key: str, dialog_title: str):
        """浏览路径"""
        from PyQt6.QtWidgets import QFileDialog
        
        path = QFileDialog.getExistingDirectory(
            self, dialog_title, 
            self.settings.value(setting_key, ""),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if path:
            if setting_key == "pretrained_path":
                self.pretrained_path.setText(path)
    
    def save_settings(self):
        """保存设置"""
        # 保存主题
        new_theme = self.theme_combo.currentText()
        self.settings.setValue("theme", new_theme)
        
        # 发送主题变化信号
        theme_key = 'light' if new_theme == '浅色主题' else 'dark'
        self.theme_changed.emit(theme_key)
        
        # 保存路径
        self.settings.setValue("pretrained_path", self.pretrained_path.text())
        
        # 保存自动保存设置
        self.settings.setValue("auto_save_interval", self.auto_save_interval.value())
        self.settings.setValue("auto_save_enabled", self.auto_save_enabled.isChecked())
        
        QMessageBox.information(self, "保存成功", "设置已保存！")
    
    def reset_settings(self):
        """恢复默认设置"""
        # 恢复默认值
        self.theme_combo.setCurrentText("深色主题")
        from pathlib import Path
        app_root = Path(__file__).parent.parent.parent  # 向上三级到EzYOLO根目录
        default_pretrained_path = app_root / "pretrained"
        self.pretrained_path.setText(str(default_pretrained_path))
        self.auto_save_interval.setValue(5)
        self.auto_save_enabled.setChecked(True)
        
        # 恢复默认快捷键
        if hasattr(self, 'reset_view_shortcut'):
            self.reset_view_shortcut.setText("R")
        if hasattr(self, 'rect_tool_shortcut'):
            self.rect_tool_shortcut.setText("W")
        if hasattr(self, 'poly_tool_shortcut'):
            self.poly_tool_shortcut.setText("P")
        if hasattr(self, 'move_tool_shortcut'):
            self.move_tool_shortcut.setText("V")
        if hasattr(self, 'prev_image_shortcut'):
            self.prev_image_shortcut.setText("A")
        if hasattr(self, 'next_image_shortcut'):
            self.next_image_shortcut.setText("D")
        if hasattr(self, 'delete_shortcut'):
            self.delete_shortcut.setText("DELETE")
        
        # 保存默认快捷键设置
        self.settings.setValue("reset_view_shortcut", "R")
        self.settings.setValue("rect_tool_shortcut", "W")
        self.settings.setValue("poly_tool_shortcut", "P")
        self.settings.setValue("move_tool_shortcut", "V")
        self.settings.setValue("prev_image_shortcut", "A")
        self.settings.setValue("next_image_shortcut", "D")
        self.settings.setValue("delete_shortcut", "DELETE")
        
        # 发送主题变化信号（默认是深色主题）
        self.theme_changed.emit('dark')
        
        QMessageBox.information(self, "恢复默认", "已恢复默认设置！")
