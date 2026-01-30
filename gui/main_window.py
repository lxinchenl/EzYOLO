# -*- coding: utf-8 -*-
"""
EzYOLO 主窗口
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QLabel, QPushButton, QFrame
)
from PyQt6.QtCore import Qt, QSettings, QSize, QPoint
from PyQt6.QtGui import QFont, QIcon

from gui.styles import get_full_stylesheet, COLORS
from gui.pages.import_page import ImportPage
from gui.pages.annotate_page import AnnotatePage
from gui.pages.train_page import TrainPage
from gui.pages.result_page import ResultPage
from gui.pages.settings_page import SettingsPage
from gui.pages.about_page import AboutPage


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        
        # 加载设置
        self.settings = QSettings("EzYOLO", "MainWindow")
        
        # 初始化UI
        self.init_ui()
        
        # 加载窗口状态
        self.load_window_state()
    
    def init_ui(self):
        """初始化界面"""
        # 设置窗口标题
        self.setWindowTitle("EzYOLO - 全流程YOLO训练")
        
        # 设置最小尺寸
        self.setMinimumSize(1200, 800)
        
        # 加载当前主题设置
        current_theme = self.settings.value("theme", "深色主题")
        theme_key = 'light' if current_theme == '浅色主题' else 'dark'
        
        # 应用样式表
        self.setStyleSheet(get_full_stylesheet(theme_key))
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建侧边栏
        self.sidebar = self.create_sidebar()
        main_layout.addWidget(self.sidebar)
        
        # 创建内容区域
        self.content_stack = self.create_content_area()
        main_layout.addWidget(self.content_stack)
        
        # 连接主题变化信号
        self.settings_page.theme_changed.connect(self.on_theme_changed)
        
        # 设置布局比例
        main_layout.setStretch(0, 0)  # 侧边栏固定宽度
        main_layout.setStretch(1, 1)  # 内容区域自适应
    
    def create_sidebar(self) -> QWidget:
        """创建侧边栏"""
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(200)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 20, 0, 20)
        layout.setSpacing(4)
        
        # 标题
        title_label = QLabel("EzYOLO")
        title_label.setObjectName("title")
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {COLORS['primary']}; padding: 0 16px;")
        layout.addWidget(title_label)
        
        # 副标题
        subtitle_label = QLabel("YOLO训练工具")
        subtitle_label.setObjectName("subtitle")
        subtitle_label.setStyleSheet("padding: 0 16px; margin-bottom: 20px;")
        layout.addWidget(subtitle_label)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"background-color: {COLORS['border']}; max-height: 1px;")
        layout.addWidget(line)
        
        # 导航按钮组
        self.nav_buttons = []
        
        # 导入按钮
        btn_import = self.create_nav_button("① 导入", 0)
        layout.addWidget(btn_import)
        self.nav_buttons.append(btn_import)
        
        # 标注按钮
        btn_annotate = self.create_nav_button("② 标注", 1)
        layout.addWidget(btn_annotate)
        self.nav_buttons.append(btn_annotate)
        
        # 训练按钮
        btn_train = self.create_nav_button("③ 训练", 2)
        layout.addWidget(btn_train)
        self.nav_buttons.append(btn_train)
        
        # 训练结果按钮
        btn_result = self.create_nav_button("④ 训练结果", 3)
        layout.addWidget(btn_result)
        self.nav_buttons.append(btn_result)
        
        # 添加弹性空间
        layout.addStretch()
        
        # 底部按钮
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setStyleSheet(f"background-color: {COLORS['border']}; max-height: 1px;")
        layout.addWidget(line2)
        
        btn_settings = self.create_nav_button("⚙ 设置", 4)
        layout.addWidget(btn_settings)
        self.nav_buttons.append(btn_settings)
        
        btn_about = self.create_nav_button("ⓘ 关于", 5)
        layout.addWidget(btn_about)
        self.nav_buttons.append(btn_about)
        
        return sidebar
    
    def create_nav_button(self, text: str, index: int) -> QPushButton:
        """创建导航按钮"""
        btn = QPushButton(text)
        btn.setObjectName("nav_button")
        btn.setCheckable(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        
        if index >= 0:
            btn.clicked.connect(lambda checked, idx=index: self.switch_page(idx))
        
        return btn
    
    def create_content_area(self) -> QStackedWidget:
        """创建内容区域"""
        stack = QStackedWidget()
        
        # 导入页面
        self.import_page = ImportPage()
        stack.addWidget(self.import_page)
        
        # 标注页面
        self.annotate_page = AnnotatePage()
        stack.addWidget(self.annotate_page)
        
        # 训练页面
        self.train_page = TrainPage()
        stack.addWidget(self.train_page)
        
        # 训练结果页面
        self.result_page = ResultPage()
        stack.addWidget(self.result_page)
        
        # 设置页面
        self.settings_page = SettingsPage()
        stack.addWidget(self.settings_page)
        
        # 关于页面
        self.about_page = AboutPage()
        stack.addWidget(self.about_page)
        
        # 默认选中第一个页面
        self.nav_buttons[0].setChecked(True)
        
        return stack
    
    def switch_page(self, index: int):
        """切换页面"""
        # 更新按钮状态
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
        
        # 切换页面
        self.content_stack.setCurrentIndex(index)
        
        # 同步项目信息到各页面
        current_project_id = self.import_page.current_project_id
        
        if index == 1:  # 标注页面
            if current_project_id:
                self.annotate_page.set_project(current_project_id)
            else:
                # 如果没有选择项目，提示用户
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(self, "提示", "请先在导入页面选择一个项目")
        
        elif index == 2:  # 训练页面
            if current_project_id:
                self.train_page.set_project(current_project_id)
            else:
                # 如果没有选择项目，提示用户
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(self, "提示", "请先在导入页面选择一个项目")
        
        elif index == 3:  # 训练结果页面
            if current_project_id:
                self.result_page.set_project(current_project_id)
            else:
                # 如果没有选择项目，提示用户
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(self, "提示", "请先在导入页面选择一个项目")
        
        elif index == 4:  # 设置页面
            # 设置页面不需要项目信息
            pass
        
        elif index == 5:  # 关于页面
            # 关于页面不需要项目信息
            pass
    
    def load_window_state(self):
        """加载窗口状态"""
        # 恢复窗口大小
        size = self.settings.value("size", QSize(1400, 900))
        self.resize(size)
        
        # 恢复窗口位置
        pos = self.settings.value("pos", QPoint(100, 100))
        self.move(pos)
        
        # 恢复窗口状态（最大化等）
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
    
    def save_window_state(self):
        """保存窗口状态"""
        self.settings.setValue("size", self.size())
        self.settings.setValue("pos", self.pos())
        self.settings.setValue("windowState", self.saveState())
    
    def on_theme_changed(self, theme_key):
        """处理主题变化（固定为深色主题）"""
        # 始终使用深色主题
        dark_theme_key = 'dark'
        # 应用深色主题样式表
        self.setStyleSheet(get_full_stylesheet(dark_theme_key))
        
        # 保存主题设置为深色主题
        self.settings.setValue("theme", "深色主题")
    
    def closeEvent(self, event):
        """关闭事件"""
        self.save_window_state()
        event.accept()
