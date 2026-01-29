# -*- coding: utf-8 -*-
"""
加载对话框组件
显示加载动画和进度信息
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QWidget, QGraphicsOpacityEffect
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QMovie, QPixmap, QColor, QPainter, QPen

from gui.styles import COLORS


class LoadingSpinner(QLabel):
    """加载旋转动画"""
    
    def __init__(self, size: int = 64, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.timer.start(50)  # 每50ms更新一次
        
    def rotate(self):
        """旋转动画"""
        self.angle = (self.angle + 30) % 360
        self.update()
    
    def paintEvent(self, event):
        """绘制旋转的圆环"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制背景圆环
        pen = QPen(QColor(COLORS['border']))
        pen.setWidth(4)
        painter.setPen(pen)
        painter.drawEllipse(8, 8, self.width() - 16, self.height() - 16)
        
        # 绘制旋转的弧
        pen = QPen(QColor(COLORS['primary']))
        pen.setWidth(4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        
        painter.translate(self.width() // 2, self.height() // 2)
        painter.rotate(self.angle)
        painter.translate(-self.width() // 2, -self.height() // 2)
        
        painter.drawArc(8, 8, self.width() - 16, self.height() - 16, 0, 120 * 16)


class LoadingDialog(QDialog):
    """加载对话框"""
    
    def __init__(self, parent=None, title: str = "加载中", message: str = "请稍候..."):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.init_ui(message)
        
        # 设置固定大小
        self.setFixedSize(300, 180)
        
        # 居中显示
        if parent:
            self.move(
                parent.x() + (parent.width() - self.width()) // 2,
                parent.y() + (parent.height() - self.height()) // 2
            )
    
    def init_ui(self, message: str):
        """初始化界面"""
        # 创建主容器
        container = QWidget(self)
        container.setGeometry(0, 0, 300, 180)
        container.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['panel']};
                border: 2px solid {COLORS['border']};
                border-radius: 12px;
            }}
        """)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 加载动画
        self.spinner = LoadingSpinner(64, self)
        layout.addWidget(self.spinner, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # 消息文本
        self.message_label = QLabel(message)
        self.message_label.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 14px;
            font-weight: bold;
        """)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.message_label)
        
        # 进度条（可选）
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS['sidebar']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                text-align: center;
                color: white;
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self.progress_bar)
    
    def set_message(self, message: str):
        """设置消息文本"""
        self.message_label.setText(message)
    
    def set_progress(self, value: int):
        """设置进度"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(value)
    
    def showEvent(self, event):
        """显示事件"""
        super().showEvent(event)
        # 确保对话框在父窗口中央
        if self.parent():
            parent_rect = self.parent().geometry()
            self.move(
                parent_rect.center().x() - self.width() // 2,
                parent_rect.center().y() - self.height() // 2
            )


class LoadingOverlay(QWidget):
    """加载遮罩层 - 用于在控件上方显示加载状态"""
    
    def __init__(self, parent: QWidget, message: str = "加载中..."):
        super().__init__(parent)
        self.setGeometry(parent.rect())
        
        # 半透明背景
        self.setStyleSheet(f"""
            QWidget {{
                background-color: rgba(30, 30, 30, 180);
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 加载动画
        self.spinner = LoadingSpinner(48, self)
        layout.addWidget(self.spinner, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # 消息文本
        self.label = QLabel(message)
        self.label.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 13px;
            margin-top: 12px;
        """)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        
        self.hide()
        
        # 使用定时器确保动画流畅
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update)
    
    def show_loading(self, message: str = None):
        """显示加载遮罩"""
        if message:
            self.label.setText(message)
        self.setGeometry(self.parent().rect())
        self.show()
        self.raise_()
        self.update_timer.start(16)  # 约60fps
    
    def hide_loading(self):
        """隐藏加载遮罩"""
        self.update_timer.stop()
        self.hide()
