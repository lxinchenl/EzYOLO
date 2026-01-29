# -*- coding: utf-8 -*-
"""
关于页面
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QFormLayout, QTextEdit
)
from PyQt6.QtCore import Qt


class AboutPage(QWidget):
    """关于页面"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        
        # 标题
        title = QLabel("关于 EzYOLO")
        title.setObjectName("title")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        main_layout.addWidget(title)
        
        # 作者信息
        author_group = self.create_author_group()
        main_layout.addWidget(author_group)
        
        # 联系方式
        contact_group = self.create_contact_group()
        main_layout.addWidget(contact_group)
        
        # 反盗版声明
        anti_piracy_group = self.create_anti_piracy_group()
        main_layout.addWidget(anti_piracy_group)
        
        # 版本信息
        version_group = self.create_version_group()
        main_layout.addWidget(version_group)
        
        # 底部
        main_layout.addStretch()
    
    def create_author_group(self) -> QGroupBox:
        """创建作者信息组"""
        group = QGroupBox("作者信息")
        
        layout = QFormLayout(group)
        
        author_label = QLabel("xinchen")
        author_label.setStyleSheet("background-color: #252526; padding: 4px; border-radius: 4px;")
        layout.addRow("作者:", author_label)
        
        return group
    
    def create_contact_group(self) -> QGroupBox:
        """创建联系方式组"""
        group = QGroupBox("联系方式")
        
        layout = QFormLayout(group)
        
        # 微信
        wechat_label = QLabel("lxc2902689500")
        wechat_label.setStyleSheet("background-color: #252526; padding: 4px; border-radius: 4px;")
        layout.addRow("微信:", wechat_label)
        
        # 邮箱
        email_label = QLabel("liuxinchen0803@qq.com")
        email_label.setStyleSheet("background-color: #252526; padding: 4px; border-radius: 4px;")
        layout.addRow("邮箱:", email_label)
        
        return group
    
    def create_anti_piracy_group(self) -> QGroupBox:
        """创建反盗版声明组"""
        group = QGroupBox("声明")
        
        layout = QVBoxLayout(group)
        
        anti_piracy_text = QTextEdit()
        anti_piracy_text.setReadOnly(True)
        anti_piracy_text.setStyleSheet("background-color: #252526; color: #ffffff; padding: 8px; border-radius: 4px;")
        anti_piracy_text.setText("任何单位或个人未经授权，不得复制、分发、修改、破解本软件。\n\n" 
                               "欢迎反馈交流，期待您的建议和反馈。\n\n" )
        layout.addWidget(anti_piracy_text)
        
        return group
    
    def create_version_group(self) -> QGroupBox:
        """创建版本信息组"""
        group = QGroupBox("版本信息")
        
        layout = QFormLayout(group)
        
        version_label = QLabel("1.0.0")
        version_label.setStyleSheet("background-color: #252526; padding: 4px; border-radius: 4px;")
        layout.addRow("版本:", version_label)
        
        update_label = QLabel("2026-01-29")
        update_label.setStyleSheet("background-color: #252526; padding: 4px; border-radius: 4px;")
        layout.addRow("更新日期:", update_label)
        
        return group