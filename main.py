#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EzYOLO - 本地YOLO全流程训练软件
主程序入口
"""

import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

from gui.main_window import MainWindow


def main():
    """主函数"""
    # 启用高DPI支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # 获取应用根目录
    app_root = Path(__file__).parent
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("EzYOLO")
    app.setApplicationVersion("1.0.0")
    
    # 设置应用图标（使用相对路径）
    icon_path = app_root / "icon.png"
    if icon_path.exists():
        app_icon = QIcon(str(icon_path))
        app.setWindowIcon(app_icon)
    
    # 创建主窗口
    window = MainWindow()
    
    # 为主窗口设置图标
    if 'app_icon' in locals():
        window.setWindowIcon(app_icon)
    
    window.show()
    
    # 运行应用
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
