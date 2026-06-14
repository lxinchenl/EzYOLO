# -*- coding: utf-8 -*-
"""图片分组选择对话框"""

from typing import List, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QMessageBox, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt

from gui.styles import COLORS
from models.database import db

UNGROUPED_SENTINEL = 0


class GroupSelectDialog(QDialog):
    """选择已有分组、新建分组，或移出分组。"""

    def __init__(
        self,
        parent,
        project_id: int,
        title: str = "选择分组",
        allow_ungroup: bool = False,
        allow_skip: bool = False,
    ):
        super().__init__(parent)
        self.project_id = project_id
        self.allow_ungroup = allow_ungroup
        self.allow_skip = allow_skip
        self._result_group_id: Optional[int] = None
        self._cancelled = True

        self.setWindowTitle(title)
        self.setMinimumWidth(360)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['background']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
            }}
            QLineEdit, QComboBox {{
                background-color: {COLORS['sidebar']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 6px 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        hint = QLabel("选择已有分组，或输入新分组名称：")
        layout.addWidget(hint)

        self.group_combo = QComboBox()
        self._refresh_groups()
        layout.addWidget(self.group_combo)

        new_row = QHBoxLayout()
        new_row.addWidget(QLabel("新建分组:"))
        self.new_group_input = QLineEdit()
        self.new_group_input.setPlaceholderText("输入自定义分组名称")
        new_row.addWidget(self.new_group_input, 1)
        layout.addLayout(new_row)

        if allow_ungroup:
            self.mode_group = QButtonGroup(self)
            self.radio_existing = QRadioButton("使用上方选择的分组")
            self.radio_new = QRadioButton("使用新建分组名称")
            self.radio_ungroup = QRadioButton("移出分组（变为未分组）")
            self.radio_existing.setChecked(True)
            for radio in (self.radio_existing, self.radio_new, self.radio_ungroup):
                self.mode_group.addButton(radio)
                layout.addWidget(radio)
        else:
            self.mode_group = QButtonGroup(self)
            self.radio_existing = QRadioButton("使用已有分组")
            self.radio_new = QRadioButton("使用新建分组")
            self.radio_existing.setChecked(True)
            for radio in (self.radio_existing, self.radio_new):
                self.mode_group.addButton(radio)
                layout.addWidget(radio)
            self.radio_ungroup = None

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        if allow_skip:
            skip_btn = QPushButton("不分组")
            skip_btn.setObjectName("secondary")
            skip_btn.clicked.connect(self._on_skip)
            btn_row.addWidget(skip_btn)
        cancel_btn = QPushButton("取消")
        cancel_btn.setObjectName("secondary")
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self._on_accept)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

    def _refresh_groups(self):
        self.group_combo.clear()
        groups = db.get_project_image_groups(self.project_id)
        counts = db.get_group_image_counts(self.project_id)
        ungrouped_count = counts.get(None, 0)
        self.group_combo.addItem(f"未分组 ({ungrouped_count})", UNGROUPED_SENTINEL)
        for group in groups:
            count = counts.get(group['id'], 0)
            self.group_combo.addItem(f"{group['name']} ({count})", group['id'])

    def _on_skip(self):
        self._result_group_id = None
        self._cancelled = False
        self.accept()

    def _on_accept(self):
        if self.radio_ungroup and self.radio_ungroup.isChecked():
            self._result_group_id = None
            self._cancelled = False
            self.accept()
            return

        if self.radio_new.isChecked():
            name = self.new_group_input.text().strip()
            if not name:
                QMessageBox.warning(self, "提示", "请输入新分组名称")
                return
            try:
                self._result_group_id = db.create_image_group(self.project_id, name)
            except ValueError as e:
                QMessageBox.warning(self, "提示", str(e))
                return
        else:
            group_id = self.group_combo.currentData()
            if group_id == UNGROUPED_SENTINEL:
                self._result_group_id = None
            else:
                self._result_group_id = group_id

        self._cancelled = False
        self.accept()

    def was_cancelled(self) -> bool:
        return self._cancelled

    def get_selected_group_id(self) -> Optional[int]:
        """返回分组 ID；None 表示未分组。"""
        return self._result_group_id


def ask_import_group(parent, project_id: int) -> tuple[bool, Optional[int]]:
    """导入前询问是否放入分组。

    Returns:
        (proceed, group_id)
        - proceed=False: 用户取消导入
        - proceed=True, group_id=None: 导入但不分组
        - proceed=True, group_id=int: 导入到指定分组
    """
    reply = QMessageBox.question(
        parent,
        "导入分组",
        "是否将本次导入的图片放入分组？\n（选择“否”则导入为未分组）",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.No,
    )
    if reply == QMessageBox.StandardButton.Cancel:
        return False, None
    if reply == QMessageBox.StandardButton.No:
        return True, None

    dialog = GroupSelectDialog(
        parent,
        project_id,
        title="选择导入分组",
        allow_ungroup=False,
        allow_skip=False,
    )
    if dialog.exec() != QDialog.DialogCode.Accepted or dialog.was_cancelled():
        return False, None
    return True, dialog.get_selected_group_id()
