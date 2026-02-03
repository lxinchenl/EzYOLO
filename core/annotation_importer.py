# -*- coding: utf-8 -*-
"""
标注导入器
支持YOLO、COCO、VOC等格式的标注导入
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os

from models.database import db


class AnnotationImporter:
    """标注导入器"""
    
    def __init__(self, project_id: int):
        """
        初始化标注导入器
        
        Args:
            project_id: 项目ID
        """
        self.project_id = project_id
        self.project = db.get_project(project_id)
        if not self.project:
            raise ValueError(f"项目 {project_id} 不存在")
        
        # 获取项目类别信息
        self.classes = json.loads(self.project['classes']) if self.project['classes'] else []
        self.class_map = {cls['name']: cls['id'] for cls in self.classes}
    
    def import_yolo_annotations(self, labels_dir: str, images_dir: str = None, overwrite: bool = False) -> Tuple[int, int]:
        """
        导入YOLO格式的标注
        
        Args:
            labels_dir: YOLO标签文件夹路径
            images_dir: 对应的图像文件夹路径（可选）
            overwrite: 是否覆盖已有的标注
            
        Returns:
            (成功导入数量, 跳过数量)
        """
        labels_dir = Path(labels_dir)
        if not labels_dir.exists():
            raise ValueError(f"标签文件夹不存在: {labels_dir}")
        
        if images_dir:
            images_dir = Path(images_dir)
        
        imported = 0
        skipped = 0
        
        # 获取所有txt文件
        txt_files = list(labels_dir.glob("*.txt"))
        
        for txt_file in txt_files:
            try:
                # 查找对应的图像文件
                image_file = self._find_corresponding_image(txt_file, images_dir)
                if not image_file:
                    print(f"未找到对应的图像文件: {txt_file}")
                    skipped += 1
                    continue
                
                # 获取图像信息
                image_info = self._get_image_info(str(image_file))
                if not image_info:
                    skipped += 1
                    continue
                
                # 查找数据库中的图像记录
                image_record = self._find_image_record(str(image_file))
                if not image_record:
                    # 如果图像不存在，先导入图像
                    image_record = self._import_image_if_needed(str(image_file))
                    if not image_record:
                        skipped += 1
                        continue
                
                # 检查是否已经有标注
                existing_annotations = db.get_image_annotations(image_record['id'])
                if existing_annotations and not overwrite:
                    # 如果已经有标注且不覆盖，跳过
                    skipped += 1
                    continue
                
                # 如果需要覆盖，先删除所有原标注
                if existing_annotations and overwrite:
                    db.delete_image_annotations(image_record['id'])
                
                # 读取YOLO标注
                annotations = self._parse_yolo_file(txt_file, image_info)
                
                # 导入标注
                for ann in annotations:
                    # 根据任务类型设置标注类型
                    project_task = self.project.get('type', 'detect')
                    if project_task == 'segment' and 'points' in ann['data']:
                        annotation_type = 'polygon'
                    elif project_task == 'pose' and 'keypoints' in ann['data']:
                        annotation_type = 'keypoint'
                    elif project_task == 'obb' and 'angle' in ann['data']:
                        annotation_type = 'obb'
                    else:
                        annotation_type = 'bbox'
                    
                    db.add_annotation(
                        image_id=image_record['id'],
                        project_id=self.project_id,
                        class_id=ann['class_id'],
                        class_name=ann['class_name'],
                        annotation_type=annotation_type,
                        data=ann['data']
                    )
                
                imported += len(annotations)
                
            except Exception as e:
                print(f"导入YOLO标注失败 {txt_file}: {e}")
                skipped += 1
        
        return imported, skipped
    
    def import_coco_annotations(self, coco_file: str, overwrite: bool = False) -> Tuple[int, int]:
        """
        导入COCO格式的标注
        
        Args:
            coco_file: COCO JSON文件路径
            overwrite: 是否覆盖已有的标注
            
        Returns:
            (成功导入数量, 跳过数量)
        """
        coco_file = Path(coco_file)
        if not coco_file.exists():
            raise ValueError(f"COCO文件不存在: {coco_file}")
        
        try:
            with open(coco_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
        except Exception as e:
            raise ValueError(f"读取COCO文件失败: {e}")
        
        imported = 0
        skipped = 0
        
        # 解析COCO数据
        images = {img['id']: img for img in coco_data.get('images', [])}
        categories = {cat['id']: cat for cat in coco_data.get('categories', [])}
        annotations = coco_data.get('annotations', [])
        
        for ann in annotations:
            try:
                # 获取图像信息
                image_id = ann.get('image_id')
                if image_id not in images:
                    skipped += 1
                    continue
                
                image_info = images[image_id]
                
                # 查找数据库中的图像记录
                image_record = self._find_image_record_by_filename(image_info['file_name'])
                if not image_record:
                    # 如果图像不存在，先导入图像
                    image_record = self._import_image_if_needed(image_info['file_name'])
                    if not image_record:
                        skipped += 1
                        continue
                
                # 检查是否已经有标注
                existing_annotations = db.get_image_annotations(image_record['id'])
                if existing_annotations and not overwrite:
                    # 如果已经有标注且不覆盖，跳过
                    skipped += 1
                    continue
                
                # 如果需要覆盖，先删除所有原标注
                if existing_annotations and overwrite:
                    db.delete_image_annotations(image_record['id'])
                
                # 获取类别信息
                category_id = ann.get('category_id')
                if category_id not in categories:
                    skipped += 1
                    continue
                
                category = categories[category_id]
                
                # 解析标注数据
                if 'bbox' in ann:
                    # 边界框标注
                    bbox = ann['bbox']  # [x, y, width, height]
                    data = {
                        'x': bbox[0],
                        'y': bbox[1],
                        'width': bbox[2],
                        'height': bbox[3]
                    }
                    
                    db.add_annotation(
                        image_id=image_record['id'],
                        project_id=self.project_id,
                        class_id=category_id,
                        class_name=category['name'],
                        annotation_type='bbox',
                        data=data
                    )
                    
                    imported += 1
                
                if 'segmentation' in ann:
                    # 分割标注
                    segmentation = ann['segmentation']
                    if isinstance(segmentation, list):
                        # 多边形格式
                        data = {'points': segmentation}
                        
                        db.add_annotation(
                            image_id=image_record['id'],
                            project_id=self.project_id,
                            class_id=category_id,
                            class_name=category['name'],
                            annotation_type='polygon',
                            data=data
                        )
                        
                        imported += 1
                
            except Exception as e:
                print(f"导入COCO标注失败: {e}")
                skipped += 1
        
        return imported, skipped
    
    def import_voc_annotations(self, voc_dir: str, overwrite: bool = False) -> Tuple[int, int]:
        """
        导入Pascal VOC格式的标注
        
        Args:
            voc_dir: VOC标注文件夹路径
            overwrite: 是否覆盖已有的标注
            
        Returns:
            (成功导入数量, 跳过数量)
        """
        voc_dir = Path(voc_dir)
        if not voc_dir.exists():
            raise ValueError(f"VOC文件夹不存在: {voc_dir}")
        
        imported = 0
        skipped = 0
        
        # 获取所有XML文件
        xml_files = list(voc_dir.glob("*.xml"))
        
        for xml_file in xml_files:
            try:
                # 解析XML文件
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # 获取图像文件名
                filename_elem = root.find('filename')
                if filename_elem is None:
                    skipped += 1
                    continue
                
                filename = filename_elem.text
                
                # 查找对应的图像文件
                image_record = self._find_image_record_by_filename(filename)
                if not image_record:
                    # 如果图像不存在，先导入图像
                    image_record = self._import_image_if_needed(filename)
                    if not image_record:
                        skipped += 1
                        continue
                
                # 检查是否已经有标注
                existing_annotations = db.get_image_annotations(image_record['id'])
                if existing_annotations and not overwrite:
                    # 如果已经有标注且不覆盖，跳过
                    skipped += 1
                    continue
                
                # 如果需要覆盖，先删除所有原标注
                if existing_annotations and overwrite:
                    db.delete_image_annotations(image_record['id'])
                
                # 获取图像尺寸
                size_elem = root.find('size')
                width = int(size_elem.find('width').text) if size_elem is not None else 0
                height = int(size_elem.find('height').text) if size_elem is not None else 0
                
                # 解析标注对象
                for obj in root.findall('object'):
                    try:
                        # 获取类别信息
                        name_elem = obj.find('name')
                        if name_elem is None:
                            continue
                        
                        class_name = name_elem.text
                        class_id = self.class_map.get(class_name, 0)
                        
                        # 获取边界框信息
                        bbox_elem = obj.find('bndbox')
                        if bbox_elem is None:
                            continue
                        
                        xmin = int(bbox_elem.find('xmin').text)
                        ymin = int(bbox_elem.find('ymin').text)
                        xmax = int(bbox_elem.find('xmax').text)
                        ymax = int(bbox_elem.find('ymax').text)
                        
                        # 转换为YOLO格式
                        data = {
                            'x': xmin,
                            'y': ymin,
                            'width': xmax - xmin,
                            'height': ymax - ymin
                        }
                        
                        db.add_annotation(
                            image_id=image_record['id'],
                            project_id=self.project_id,
                            class_id=class_id,
                            class_name=class_name,
                            annotation_type='bbox',
                            data=data
                        )
                        
                        imported += 1
                        
                    except Exception as e:
                        print(f"解析VOC对象失败: {e}")
                        skipped += 1
                
            except Exception as e:
                print(f"导入VOC标注失败 {xml_file}: {e}")
                skipped += 1
        
        return imported, skipped
    
    def _parse_yolo_file(self, txt_file: Path, image_info: Dict) -> List[Dict]:
        """
        解析YOLO标注文件
        
        Args:
            txt_file: YOLO标签文件路径
            image_info: 图像信息
            
        Returns:
            标注列表
        """
        annotations = []
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 1:
                    continue
                
                # 解析类别ID
                class_id = int(parts[0])
                class_name = self._get_class_name_by_id(class_id)
                
                # 转换为像素坐标
                img_width = image_info['width']
                img_height = image_info['height']
                
                project_task = self.project.get('type', 'detect')
                
                if project_task == 'classify':
                    # 分类任务：只需要类别ID
                    annotations.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'data': {
                            'class_id': class_id
                        }
                    })
                elif len(parts) >= 5:
                    # 解析基本的边界框信息
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    if project_task == 'segment' and len(parts) > 5:
                        # 分割任务：处理多边形点
                        points = []
                        for i in range(5, len(parts), 2):
                            if i + 1 < len(parts):
                                px = float(parts[i]) * img_width
                                py = float(parts[i + 1]) * img_height
                                points.append({'x': px, 'y': py})
                        
                        annotations.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'data': {
                                'points': points
                            }
                        })
                    elif project_task == 'pose' and len(parts) > 5:
                        # 姿态估计任务：处理关键点
                        keypoints = []
                        for i in range(5, len(parts), 3):
                            if i + 2 < len(parts):
                                kp_x = float(parts[i]) * img_width
                                kp_y = float(parts[i + 1]) * img_height
                                kp_v = float(parts[i + 2])
                                keypoints.append({
                                    'x': kp_x,
                                    'y': kp_y,
                                    'v': kp_v
                                })
                        
                        annotations.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'data': {
                                'x': (x_center - width / 2) * img_width,
                                'y': (y_center - height / 2) * img_height,
                                'width': width * img_width,
                                'height': height * img_height,
                                'keypoints': keypoints
                            }
                        })
                    elif project_task == 'obb' and len(parts) > 5:
                        # 旋转目标检测任务：处理角度
                        angle = float(parts[5]) if len(parts) > 5 else 0.0
                        
                        annotations.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'data': {
                                'x': (x_center - width / 2) * img_width,
                                'y': (y_center - height / 2) * img_height,
                                'width': width * img_width,
                                'height': height * img_height,
                                'angle': angle
                            }
                        })
                    else:
                        # 检测任务：处理边界框
                        x = (x_center - width / 2) * img_width
                        y = (y_center - height / 2) * img_height
                        w = width * img_width
                        h = height * img_height
                        
                        annotations.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'data': {
                                'x': x,
                                'y': y,
                                'width': w,
                                'height': h
                            }
                        })
            
        except Exception as e:
            print(f"解析YOLO文件失败 {txt_file}: {e}")
        
        return annotations
    
    def _find_corresponding_image(self, txt_file: Path, images_dir: Path = None) -> Optional[Path]:
        """
        查找对应的图像文件
        
        Args:
            txt_file: YOLO标签文件路径
            images_dir: 图像文件夹路径
            
        Returns:
            对应的图像文件路径，未找到返回None
        """
        # 支持的图像格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # 尝试相同目录
        base_name = txt_file.stem
        for ext in image_extensions:
            image_file = txt_file.parent / f"{base_name}{ext}"
            if image_file.exists():
                return image_file
        
        # 如果指定了图像目录，尝试在图像目录中查找
        if images_dir:
            for ext in image_extensions:
                image_file = images_dir / f"{base_name}{ext}"
                if image_file.exists():
                    return image_file
        
        return None
    
    def _get_image_info(self, image_path: str) -> Optional[Dict]:
        """
        获取图像信息
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像信息字典，失败返回None
        """
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
                return {
                    'width': width,
                    'height': height,
                    'format': img.format.lower() if img.format else 'unknown'
                }
        except Exception as e:
            print(f"获取图像信息失败 {image_path}: {e}")
            return None
    
    def _find_image_record(self, image_path: str) -> Optional[Dict]:
        """
        查找数据库中的图像记录
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像记录，未找到返回None
        """
        filename = Path(image_path).name
        images = db.get_project_images(self.project_id)
        
        for image in images:
            if image['filename'] == filename or image['original_path'] == image_path:
                return image
        
        return None
    
    def _find_image_record_by_filename(self, filename: str) -> Optional[Dict]:
        """
        根据文件名查找数据库中的图像记录
        
        Args:
            filename: 图像文件名
            
        Returns:
            图像记录，未找到返回None
        """
        images = db.get_project_images(self.project_id)
        
        for image in images:
            if image['filename'] == filename:
                return image
        
        return None
    
    def _import_image_if_needed(self, image_path: str) -> Optional[Dict]:
        """
        如果需要，导入图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像记录，失败返回None
        """
        # 这里需要调用图像导入功能
        # TODO: 实现图像导入逻辑
        return None
    
    def _get_class_name_by_id(self, class_id: int) -> str:
        """
        根据类别ID获取类别名称
        
        Args:
            class_id: 类别ID
            
        Returns:
            类别名称
        """
        for cls in self.classes:
            if cls['id'] == class_id:
                return cls['name']
        return f"class_{class_id}"