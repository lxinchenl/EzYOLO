# -*- coding: utf-8 -*-
"""
可视化页面
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel


class VisualizePage(QWidget):
    """可视化页面"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题
        title = QLabel("训练可视化")
        title.setObjectName("title")
        layout.addWidget(title)
        
        # 说明文字
        desc = QLabel("查看训练结果和模型性能")
        desc.setObjectName("subtitle")
        layout.addWidget(desc)
        
        layout.addStretch()
