# -*- coding: utf-8 -*-
"""
EzYOLO 样式定义
支持深色主题和浅色主题
"""

# 深色主题颜色定义
DARK_COLORS = {
    # 主色调
    'background': '#1E1E1E',
    'sidebar': '#252526',
    'panel': '#2D2D30',
    'selected': '#094771',
    
    # 强调色
    'primary': '#007ACC',
    'success': '#4EC9B0',
    'warning': '#DCDCAA',
    'error': '#F44747',
    
    # 文字色
    'text_primary': '#D4D4D4',
    'text_secondary': '#858585',
    'text_disabled': '#5C5C5C',
    
    # 边框
    'border': '#3E3E42',
    'hover': '#2A2D2E',
}

# 浅色主题颜色定义
LIGHT_COLORS = {
    # 主色调
    'background': '#FFFFFF',
    'sidebar': '#F5F5F5',
    'panel': '#F0F0F0',
    'selected': '#E6F2FB',
    
    # 强调色
    'primary': '#0078D7',
    'success': '#107C10',
    'warning': '#FFB900',
    'error': '#D13438',
    
    # 文字色
    'text_primary': '#333333',
    'text_secondary': '#666666',
    'text_disabled': '#999999',
    
    # 边框
    'border': '#E0E0E0',
    'hover': '#E8E8E8',
}

# 默认使用深色主题
COLORS = DARK_COLORS

def generate_stylesheet(colors):
    """根据颜色生成样式表"""
    stylesheet = ""
    
    # 主窗口样式
    stylesheet += """
QMainWindow {
    background-color: %s;
    color: %s;
}

QWidget {
    background-color: %s;
    color: %s;
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
}

QFrame {
    background-color: %s;
    border: 1px solid %s;
    border-radius: 4px;
}
"""
    
    # 侧边栏样式
    stylesheet += """
QWidget#sidebar {
    background-color: %s;
    border-right: 1px solid %s;
}

QPushButton#nav_button {
    background-color: transparent;
    color: %s;
    border: none;
    padding: 12px 16px;
    text-align: left;
    font-size: 14px;
}

QPushButton#nav_button:hover {
    background-color: %s;
}

QPushButton#nav_button:checked {
    background-color: %s;
    color: %s;
}

QPushButton#nav_button:disabled {
    color: %s;
}
"""
    
    # 按钮样式
    stylesheet += """
QPushButton {
    background-color: %s;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-size: 13px;
}

QPushButton:hover {
    background-color: %s;
    opacity: 0.9;
}

QPushButton:pressed {
    background-color: %s;
    opacity: 0.8;
}

QPushButton:disabled {
    background-color: %s;
    color: %s;
}

QPushButton#secondary {
    background-color: %s;
    color: %s;
    border: 1px solid %s;
}

QPushButton#secondary:hover {
    background-color: %s;
}
"""
    
    # 输入框样式
    stylesheet += """
QLineEdit {
    background-color: %s;
    color: %s;
    border: 1px solid %s;
    padding: 6px 10px;
    border-radius: 4px;
}

QLineEdit:focus {
    border: 1px solid %s;
}

QLineEdit:disabled {
    background-color: %s;
    color: %s;
}

QComboBox {
    background-color: %s;
    color: %s;
    border: 1px solid %s;
    padding: 6px 10px;
    border-radius: 4px;
}

QComboBox:focus {
    border: 1px solid %s;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: %s;
    color: %s;
    border: 1px solid %s;
    selection-background-color: %s;
}
"""
    
    # 标签样式
    stylesheet += """
QLabel {
    color: %s;
    font-size: 13px;
}

QLabel#title {
    font-size: 16px;
    font-weight: bold;
    color: %s;
}

QLabel#subtitle {
    font-size: 12px;
    color: %s;
}
"""
    
    # 滚动条样式
    stylesheet += """
QScrollBar:vertical {
    background-color: %s;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: %s;
    border-radius: 6px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: %s;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: %s;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background-color: %s;
    border-radius: 6px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background-color: %s;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0px;
}
"""
    
    # 列表样式
    stylesheet += """
QListWidget {
    background-color: %s;
    color: %s;
    border: 1px solid %s;
    border-radius: 4px;
    outline: none;
}

QListWidget::item {
    padding: 8px 12px;
    border-bottom: 1px solid %s;
}

QListWidget::item:selected {
    background-color: %s;
    color: %s;
}

QListWidget::item:hover {
    background-color: %s;
}
"""
    
    # 分组框样式
    stylesheet += """
QGroupBox {
    background-color: %s;
    color: %s;
    border: 1px solid %s;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: %s;
}
"""
    
    # 进度条样式
    stylesheet += """
QProgressBar {
    background-color: %s;
    border: 1px solid %s;
    border-radius: 4px;
    text-align: center;
    color: %s;
}

QProgressBar::chunk {
    background-color: %s;
    border-radius: 3px;
}
"""
    
    # 标签页样式
    stylesheet += """
QTabWidget::pane {
    background-color: %s;
    border: 1px solid %s;
    border-radius: 4px;
    top: -1px;
}

QTabBar::tab {
    background-color: %s;
    color: %s;
    padding: 8px 16px;
    border: 1px solid %s;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background-color: %s;
    color: %s;
    border-bottom: 1px solid %s;
}

QTabBar::tab:hover:!selected {
    background-color: %s;
    color: %s;
}
"""
    
    # 替换颜色值
    color_values = [
        # 主窗口样式
        colors['background'],
        colors['text_primary'],
        colors['background'],
        colors['text_primary'],
        colors['panel'],
        colors['border'],
        
        # 侧边栏样式
        colors['sidebar'],
        colors['border'],
        colors['text_primary'],
        colors['hover'],
        colors['selected'],
        colors['text_primary'],
        colors['text_disabled'],
        
        # 按钮样式
        colors['primary'],
        colors['primary'],
        colors['primary'],
        colors['text_disabled'],
        colors['text_secondary'],
        colors['panel'],
        colors['text_primary'],
        colors['border'],
        colors['hover'],
        
        # 输入框样式
        colors['panel'],
        colors['text_primary'],
        colors['border'],
        colors['primary'],
        colors['sidebar'],
        colors['text_disabled'],
        colors['panel'],
        colors['text_primary'],
        colors['border'],
        colors['primary'],
        colors['panel'],
        colors['text_primary'],
        colors['border'],
        colors['selected'],
        
        # 标签样式
        colors['text_primary'],
        colors['text_primary'],
        colors['text_secondary'],
        
        # 滚动条样式
        colors['sidebar'],
        colors['border'],
        colors['text_secondary'],
        colors['sidebar'],
        colors['border'],
        colors['text_secondary'],
        
        # 列表样式
        colors['panel'],
        colors['text_primary'],
        colors['border'],
        colors['border'],
        colors['selected'],
        colors['text_primary'],
        colors['hover'],
        
        # 分组框样式
        colors['panel'],
        colors['text_primary'],
        colors['border'],
        colors['text_secondary'],
        
        # 进度条样式
        colors['sidebar'],
        colors['border'],
        colors['text_primary'],
        colors['primary'],
        
        # 标签页样式
        colors['panel'],
        colors['border'],
        colors['sidebar'],
        colors['text_secondary'],
        colors['border'],
        colors['panel'],
        colors['text_primary'],
        colors['panel'],
        colors['hover'],
        colors['text_primary']
    ]
    
    return stylesheet % tuple(color_values)

# 完整样式表
def get_full_stylesheet(theme='dark'):
    """获取完整样式表
    
    Args:
        theme: 主题名称，'dark' 或 'light'
    """
    if theme == 'light':
        return generate_stylesheet(LIGHT_COLORS)
    else:
        return generate_stylesheet(DARK_COLORS)
