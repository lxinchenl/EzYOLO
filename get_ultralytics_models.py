# -*- coding: utf-8 -*-
"""
获取Ultralytics支持的所有YOLO模型信息
自动检测所有可用的预训练模型
"""

import json
import os
import re


def get_available_models_from_hub():
    """从Ultralytics Hub获取所有可用模型列表"""
    models_info = {}
    
    try:
        from ultralytics import YOLO
        from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS
        import torch
        
        print("✓ 检测到Ultralytics库")
        print(f"✓ 发现 {len(GITHUB_ASSETS_STEMS)} 个预训练模型")
        
        # 解析模型名称
        # 支持的格式: yolov8n, yolov8n-seg, yolov8n-cls, yolov8n-pose, yolov8n-obb, yolov8n-world
        pattern = r'^(yolo[v]?\d+)([a-z]+)?(-[a-z]+)?$'
        
        for model_stem in sorted(GITHUB_ASSETS_STEMS):
            match = re.match(pattern, model_stem)
            if not match:
                continue
            
            base_name = match.group(1)  # yolov8, yolo11, etc.
            size = match.group(2) or 'n'  # n, s, m, l, x, tiny, etc.
            task_suffix = match.group(3) or ''  # -seg, -cls, -pose, etc.
            
            # 转换版本名称为标准格式
            if base_name.startswith('yolov'):
                version = f"YOLO{base_name[4:].upper()}"  # yolov8 -> YOLOv8
            elif base_name.startswith('yolo'):
                version_num = base_name[4:]
                version = f"YOLOv{version_num}" if version_num.isdigit() else f"YOLO{version_num.upper()}"
            else:
                continue
            
            # 初始化版本字典
            if version not in models_info:
                models_info[version] = {}
            
            # 任务类型
            task = 'detect'
            if '-seg' in task_suffix:
                task = 'segment'
            elif '-cls' in task_suffix:
                task = 'classify'
            elif '-pose' in task_suffix:
                task = 'pose'
            elif '-obb' in task_suffix:
                task = 'obb'
            elif '-world' in task_suffix:
                task = 'world'
            
            # 初始化型号字典
            if size not in models_info[version]:
                models_info[version][size] = {
                    'tasks': {},
                    'filename': f"{model_stem}.pt"
                }
            
            # 记录任务类型
            models_info[version][size]['tasks'][task] = f"{model_stem}.pt"
            
        return models_info
        
    except ImportError as e:
        print(f"✗ 未检测到Ultralytics库: {e}")
        print("请安装: pip install ultralytics")
        return None


def test_model_loading(models_info: dict, max_test: int = 20):
    """测试加载部分模型获取真实参数"""
    tested_models = {}
    test_count = 0
    
    try:
        from ultralytics import YOLO
        import torch
        
        print("\n" + "=" * 60)
        print("测试加载模型获取真实参数...")
        print("=" * 60)
        
        for version, sizes in models_info.items():
            tested_models[version] = {}
            
            for size, info in sizes.items():
                if test_count >= max_test:
                    break
                
                # 优先测试detect任务
                filename = info['tasks'].get('detect') or list(info['tasks'].values())[0]
                model_name = filename.replace('.pt', '')
                
                try:
                    print(f"\n测试 {version} {size} ({model_name})...")
                    
                    # 加载模型
                    model = YOLO(filename)
                    
                    # 获取模型信息
                    model_info_data = {
                        'filename': filename,
                        'tasks': list(info['tasks'].keys()),
                        'loaded': True
                    }
                    
                    # 获取参数量
                    try:
                        total_params = sum(p.numel() for p in model.model.parameters())
                        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
                        model_info_data['parameters'] = {
                            'total': total_params,
                            'trainable': trainable_params,
                            'total_m': round(total_params / 1e6, 2)
                        }
                    except Exception as e:
                        model_info_data['parameters'] = {'error': str(e)}
                    
                    # 获取层数和GFLOPs
                    try:
                        from ultralytics.utils.torch_utils import model_info_for_loggers
                        # 从模型字符串解析
                        summary = str(model.model)
                        lines = summary.split('\n')
                        for line in lines:
                            if 'layers' in line.lower() and 'parameters' in line.lower():
                                # 解析类似: Model summary: 225 layers, 11126745 parameters, 11126745 gradients, 28.9 GFLOPs
                                match = re.search(r'(\d+)\s*layers.*?([\d.]+)\s*GFLOPs', line, re.IGNORECASE)
                                if match:
                                    model_info_data['layers'] = int(match.group(1))
                                    model_info_data['gflops'] = float(match.group(2))
                                break
                    except Exception as e:
                        model_info_data['layers_error'] = str(e)
                    
                    # 获取类别信息
                    try:
                        if hasattr(model, 'names'):
                            model_info_data['num_classes'] = len(model.names)
                            model_info_data['class_names'] = list(model.names.values())[:10]  # 只保存前10个
                    except:
                        pass
                    
                    tested_models[version][size] = model_info_data
                    print(f"  ✓ 成功: {model_info_data.get('parameters', {}).get('total_m', '-')}M params, "
                          f"{model_info_data.get('gflops', '-')} GFLOPs")
                    
                    # 删除模型释放内存
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    test_count += 1
                    
                except Exception as e:
                    tested_models[version][size] = {
                        'filename': filename,
                        'tasks': list(info['tasks'].keys()),
                        'loaded': False,
                        'error': str(e)
                    }
                    print(f"  ✗ 失败: {e}")
        
        return tested_models
        
    except ImportError:
        print("✗ 无法加载ultralytics进行测试")
        return {}


def generate_model_info_file(detected_models: dict, tested_models: dict):
    """生成模型信息Python文件"""
    
    # 合并检测到的和测试过的模型信息
    merged_info = {}
    
    for version, sizes in detected_models.items():
        merged_info[version] = {}
        for size, info in sizes.items():
            merged_info[version][size] = {
                'filename': info['filename'],
                'tasks': list(info['tasks'].keys()),
            }
            
            # 添加测试信息（如果有）
            if version in tested_models and size in tested_models[version]:
                test_data = tested_models[version][size]
                if 'parameters' in test_data:
                    merged_info[version][size]['parameters'] = test_data['parameters']
                if 'gflops' in test_data:
                    merged_info[version][size]['gflops'] = test_data['gflops']
                if 'layers' in test_data:
                    merged_info[version][size]['layers'] = test_data['layers']
                if 'num_classes' in test_data:
                    merged_info[version][size]['num_classes'] = test_data['num_classes']
    
    # 生成代码
    code = '''# -*- coding: utf-8 -*-
"""
YOLO模型信息
自动生成的模型信息文件 - 包含所有Ultralytics支持的模型
"""

# 所有检测到的模型信息
ULTRALYTICS_MODELS = '''
    
    code += json.dumps(merged_info, indent=4, ensure_ascii=False)
    
    code += '''


# 按版本组织的模型
VERSIONS = {
'''
    
    for version in sorted(merged_info.keys()):
        sizes = list(merged_info[version].keys())
        code += f'    "{version}": {sizes},\n'
    
    code += '''}


# 型号名称映射（中文）
SIZE_NAMES = {
    "n": "nano (超轻量)",
    "s": "small (轻量)",
    "m": "medium (中等)",
    "l": "large (大)",
    "x": "xlarge (超大)",
    "tiny": "tiny (超轻量)",
    "t": "tiny (超轻量)",
    "c": "compact (紧凑)",
    "e": "extended (扩展)",
    "b": "balanced (平衡)",
}


# 任务类型映射
TASK_NAMES = {
    "detect": "目标检测",
    "segment": "实例分割",
    "classify": "图像分类",
    "pose": "姿态估计",
    "obb": "旋转框检测",
    "world": "开放词汇检测",
}


def get_model_info(version: str, size: str, task: str = "detect") -> dict:
    """
    获取模型信息
    
    Args:
        version: 版本名称，如 "YOLOv8"
        size: 型号，如 "n", "s", "m", "l", "x"
        task: 任务类型，如 "detect", "segment", "classify"
    
    Returns:
        模型信息字典
    """
    version_data = ULTRALYTICS_MODELS.get(version, {})
    size_data = version_data.get(size, {})
    
    # 获取对应任务的文件名
    tasks = size_data.get('tasks', [])
    filename = None
    
    if task in tasks:
        # 构建文件名
        base = f"{version.lower().replace('v', '')}{size}"
        if task != 'detect':
            base += f"-{task}"
        filename = f"{base}.pt"
    else:
        # 使用第一个可用任务
        if tasks:
            filename = size_data.get('filename')
    
    info = {
        'version': version,
        'size': size,
        'task': task,
        'filename': filename,
        'available_tasks': tasks,
    }
    
    # 添加测试信息
    if 'parameters' in size_data:
        info['parameters'] = size_data['parameters']
    if 'gflops' in size_data:
        info['gflops'] = size_data['gflops']
    if 'layers' in size_data:
        info['layers'] = size_data['layers']
    
    return info


def format_model_info(info: dict) -> str:
    """格式化模型信息显示"""
    lines = [
        f"版本: {info.get('version', '-')}",
        f"型号: {info.get('size', '-')}",
        f"任务: {TASK_NAMES.get(info.get('task', ''), info.get('task', '-'))}",
    ]
    
    if 'parameters' in info:
        params = info['parameters']
        lines.append(f"参数量: {params.get('total_m', '-')}M")
    
    if 'gflops' in info:
        lines.append(f"FLOPs: {info['gflops']} GFLOPs")
    
    if 'layers' in info:
        lines.append(f"层数: {info['layers']}")
    
    lines.append(f"文件名: {info.get('filename', '-')}")
    
    return "\\n".join(lines)


def list_all_models():
    """列出所有可用模型"""
    print("=" * 60)
    print("Ultralytics YOLO 模型列表")
    print("=" * 60)
    
    for version in sorted(ULTRALYTICS_MODELS.keys()):
        print(f"\\n{version}:")
        sizes = ULTRALYTICS_MODELS[version]
        for size in sorted(sizes.keys()):
            info = sizes[size]
            tasks = ", ".join([TASK_NAMES.get(t, t) for t in info.get('tasks', [])])
            params = info.get('parameters', {}).get('total_m', '-')
            gflops = info.get('gflops', '-')
            print(f"  [{size:5s}] {tasks:20s} | 参数量: {params:>6s}M | FLOPs: {gflops:>6s}G")


if __name__ == "__main__":
    list_all_models()
'''
    
    return code


def main():
    """主函数"""
    print("=" * 60)
    print("Ultralytics YOLO模型信息获取工具")
    print("=" * 60)
    
    # 步骤1: 从Hub获取所有模型
    print("\n[1/3] 从Ultralytics Hub获取模型列表...")
    detected_models = get_available_models_from_hub()
    
    if detected_models is None:
        print("\n✗ 无法获取模型列表，请确保已安装ultralytics")
        return
    
    print(f"\n✓ 检测到 {len(detected_models)} 个版本")
    total_models = sum(len(sizes) for sizes in detected_models.values())
    print(f"✓ 共 {total_models} 个模型")
    
    # 显示检测到的模型
    print("\n" + "=" * 60)
    print("检测到的模型:")
    print("=" * 60)
    for version, sizes in sorted(detected_models.items()):
        print(f"\n{version}:")
        for size, info in sorted(sizes.items()):
            tasks = ", ".join(info['tasks'].keys())
            print(f"  - {size:5s}: {tasks}")
    
    # 步骤2: 测试加载部分模型
    print("\n[2/3] 测试加载模型获取真实参数...")
    tested_models = test_model_loading(detected_models, max_test=30)
    
    # 步骤3: 生成代码文件
    print("\n[3/3] 生成模型信息代码文件...")
    code = generate_model_info_file(detected_models, tested_models)
    
    # 保存文件
    output_file = "model_info.py"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(code)
    
    print(f"\n✓ 模型信息已保存到: {os.path.abspath(output_file)}")
    print(f"✓ 文件大小: {os.path.getsize(output_file)} bytes")
    
    # 使用说明
    print("\n" + "=" * 60)
    print("使用说明:")
    print("=" * 60)
    print("1. 导入: from model_info import ULTRALYTICS_MODELS, get_model_info")
    print("2. 获取信息: info = get_model_info('YOLOv8', 'n', 'detect')")
    print("3. 格式化: print(format_model_info(info))")
    print("4. 列全部: python model_info.py")


if __name__ == "__main__":
    main()
