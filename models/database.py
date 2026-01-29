# -*- coding: utf-8 -*-
"""
数据库管理模块
使用SQLite作为本地数据库
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from contextlib import contextmanager


class Database:
    """数据库管理类"""
    
    def __init__(self, db_path: str = None):
        """
        初始化数据库
        
        Args:
            db_path: 数据库文件路径，默认为项目目录下的data/EzYOLO.db
        """
        if db_path is None:
            # 默认存储在软件所在目录
            current_dir = Path(__file__).parent.parent
            data_dir = current_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "EzYOLO.db")
        else:
            self.db_path = db_path
        
        # 初始化数据库
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_database(self):
        """初始化数据库表结构"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 创建项目表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    type TEXT DEFAULT 'detection',
                    classes TEXT DEFAULT '[]',
                    status TEXT DEFAULT 'created',
                    storage_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建数据集表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    type TEXT DEFAULT 'train',
                    image_count INTEGER DEFAULT 0,
                    annotation_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
            """)
            
            # 创建图像表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    dataset_id INTEGER,
                    filename TEXT NOT NULL,
                    original_path TEXT,
                    storage_path TEXT,
                    width INTEGER,
                    height INTEGER,
                    size INTEGER,
                    format TEXT,
                    status TEXT DEFAULT 'pending',
                    annotated_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE SET NULL
                )
            """)
            
            # 创建标注表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER NOT NULL,
                    project_id INTEGER NOT NULL,
                    class_id INTEGER DEFAULT 0,
                    class_name TEXT,
                    type TEXT DEFAULT 'bbox',
                    data TEXT NOT NULL,
                    attributes TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
            """)
            
            # 创建训练任务表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    model_version TEXT DEFAULT 'v8',
                    model_type TEXT DEFAULT 'n',
                    task_type TEXT DEFAULT 'detect',
                    config TEXT DEFAULT '{}',
                    status TEXT DEFAULT 'pending',
                    progress INTEGER DEFAULT 0,
                    current_epoch INTEGER DEFAULT 0,
                    total_epochs INTEGER DEFAULT 100,
                    metrics TEXT DEFAULT '{}',
                    weights_path TEXT,
                    log_path TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                )
            """)
            
            # 创建训练指标历史表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL,
                    epoch INTEGER NOT NULL,
                    metrics TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES training_jobs(id) ON DELETE CASCADE
                )
            """)
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_project ON images(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_annotations_image ON annotations(image_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_jobs_project ON training_jobs(project_id)")
            
            conn.commit()
    
    # ==================== 项目操作 ====================
    
    def create_project(self, name: str, description: str = "", 
                       project_type: str = "detection", 
                       classes: List[Dict] = None) -> int:
        """
        创建新项目
        
        Args:
            name: 项目名称
            description: 项目描述
            project_type: 项目类型 (detection/segmentation/classification)
            classes: 类别列表 [{"id": 0, "name": "person", "color": "#FF0000"}]
            
        Returns:
            项目ID
        """
        if classes is None:
            classes = []
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # 项目存储路径也放在软件所在目录
            current_dir = Path(__file__).parent.parent
            storage_path = current_dir / "projects" / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            storage_path.mkdir(parents=True, exist_ok=True)
            
            cursor.execute("""
                INSERT INTO projects (name, description, type, classes, storage_path)
                VALUES (?, ?, ?, ?, ?)
            """, (
                name, 
                description, 
                project_type, 
                json.dumps(classes, ensure_ascii=False),
                str(storage_path)
            ))
            return cursor.lastrowid
    
    def get_project(self, project_id: int) -> Optional[Dict]:
        """获取项目信息"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def get_all_projects(self) -> List[Dict]:
        """获取所有项目"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects ORDER BY updated_at DESC")
            return [dict(row) for row in cursor.fetchall()]
    
    def update_project(self, project_id: int, **kwargs) -> bool:
        """更新项目信息"""
        allowed_fields = ['name', 'description', 'type', 'classes', 'status', 'storage_path']
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return False
        
        # 处理classes字段
        if 'classes' in updates and isinstance(updates['classes'], list):
            updates['classes'] = json.dumps(updates['classes'], ensure_ascii=False)
        
        updates['updated_at'] = datetime.now().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values()) + [project_id]
            cursor.execute(f"UPDATE projects SET {set_clause} WHERE id = ?", values)
            return cursor.rowcount > 0
    
    def delete_project(self, project_id: int) -> bool:
        """删除项目"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            return cursor.rowcount > 0
    
    # ==================== 图像操作 ====================
    
    def add_image(self, project_id: int, filename: str, storage_path: str,
                  width: int = None, height: int = None, size: int = None,
                  image_format: str = None, original_path: str = None,
                  dataset_id: int = None) -> int:
        """添加图像记录"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO images 
                (project_id, dataset_id, filename, original_path, storage_path, 
                 width, height, size, format)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (project_id, dataset_id, filename, original_path, storage_path,
                  width, height, size, image_format))
            return cursor.lastrowid
    
    def get_project_images(self, project_id: int, status: str = None) -> List[Dict]:
        """获取项目下的所有图像"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    "SELECT * FROM images WHERE project_id = ? AND status = ? ORDER BY created_at",
                    (project_id, status)
                )
            else:
                cursor.execute(
                    "SELECT * FROM images WHERE project_id = ? ORDER BY created_at",
                    (project_id,)
                )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_image(self, image_id: int) -> Optional[Dict]:
        """获取单个图像信息"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM images WHERE id = ?",
                (image_id,)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def delete_image_annotations(self, image_id: int) -> bool:
        """删除图像的所有标注"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM annotations WHERE image_id = ?",
                (image_id,)
            )
            return cursor.rowcount > 0
    
    def update_image_status(self, image_id: int, status: str) -> bool:
        """更新图像状态"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            annotated_at = datetime.now().isoformat() if status == 'annotated' else None
            cursor.execute(
                "UPDATE images SET status = ?, annotated_at = ? WHERE id = ?",
                (status, annotated_at, image_id)
            )
            return cursor.rowcount > 0
    
    def delete_image(self, image_id: int) -> bool:
        """删除图像"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 首先删除相关的标注
            cursor.execute("DELETE FROM annotations WHERE image_id = ?", (image_id,))
            
            # 然后删除图像记录
            cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
            
            return cursor.rowcount > 0
    
    # ==================== 标注操作 ====================
    
    def add_annotation(self, image_id: int, project_id: int, class_id: int,
                       class_name: str, annotation_type: str, data: Dict,
                       attributes: Dict = None) -> int:
        """
        添加标注
        
        Args:
            image_id: 图像ID
            project_id: 项目ID
            class_id: 类别ID
            class_name: 类别名称
            annotation_type: 标注类型 (bbox/polygon/keypoint)
            data: 标注数据，如 {"x": 10, "y": 20, "width": 100, "height": 100}
            attributes: 额外属性
        """
        if attributes is None:
            attributes = {}
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO annotations 
                (image_id, project_id, class_id, class_name, type, data, attributes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (image_id, project_id, class_id, class_name, annotation_type,
                  json.dumps(data), json.dumps(attributes)))
            
            # 更新图像状态
            cursor.execute(
                "UPDATE images SET status = 'annotated', annotated_at = ? WHERE id = ?",
                (datetime.now().isoformat(), image_id)
            )
            
            return cursor.lastrowid
    
    def get_image_annotations(self, image_id: int) -> List[Dict]:
        """获取图像的所有标注"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM annotations WHERE image_id = ?", (image_id,))
            rows = cursor.fetchall()
            annotations = []
            for row in rows:
                ann = dict(row)
                ann['data'] = json.loads(ann['data'])
                ann['attributes'] = json.loads(ann['attributes'])
                annotations.append(ann)
            return annotations
    
    def delete_annotation(self, annotation_id: int) -> bool:
        """删除标注"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
            return cursor.rowcount > 0
    
    # ==================== 训练任务操作 ====================
    
    def create_training_job(self, project_id: int, name: str,
                           model_version: str = 'v8', model_type: str = 'n',
                           task_type: str = 'detect', config: Dict = None) -> int:
        """创建训练任务"""
        if config is None:
            config = {}
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_jobs 
                (project_id, name, model_version, model_type, task_type, config)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (project_id, name, model_version, model_type, task_type,
                  json.dumps(config)))
            return cursor.lastrowid
    
    def update_training_status(self, job_id: int, status: str,
                               progress: int = None, metrics: Dict = None) -> bool:
        """更新训练状态"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            updates = ["status = ?"]
            values = [status]
            
            if progress is not None:
                updates.append("progress = ?")
                values.append(progress)
            
            if metrics is not None:
                updates.append("metrics = ?")
                values.append(json.dumps(metrics))
            
            if status == 'running' and progress == 0:
                updates.append("started_at = ?")
                values.append(datetime.now().isoformat())
            elif status in ['completed', 'failed']:
                updates.append("completed_at = ?")
                values.append(datetime.now().isoformat())
            
            values.append(job_id)
            
            cursor.execute(
                f"UPDATE training_jobs SET {', '.join(updates)} WHERE id = ?",
                values
            )
            return cursor.rowcount > 0
    
    def get_project_training_jobs(self, project_id: int) -> List[Dict]:
        """获取项目的所有训练任务"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM training_jobs WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,)
            )
            rows = cursor.fetchall()
            jobs = []
            for row in rows:
                job = dict(row)
                job['config'] = json.loads(job['config'])
                job['metrics'] = json.loads(job['metrics'])
                jobs.append(job)
            return jobs


# 全局数据库实例
db = Database()
