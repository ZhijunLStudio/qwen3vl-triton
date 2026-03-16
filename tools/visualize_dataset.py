#!/usr/bin/env python3
"""
数据集可视化工具
将数据集中的问题和其他图像、信息叠加绘制并保存
"""
import os
from pathlib import Path
from datasets import load_from_disk
from PIL import Image, ImageDraw, ImageFont


def visualize_dataset(
    dataset_path: str = "./data",
    output_dir: str = "./output_visualized",
    num_samples: int = None,
    start_index: int = 0,
    font_size: int = 20
):
    """
    可视化数据集（图像 + 问题 + 其他信息）
    
    参数:
        dataset_path: 数据集路径
        output_dir: 输出目录
        num_samples: 要可视化的样本数（None表示全部）
        start_index: 起始索引
        font_size: 字体大小
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"正在加载数据集: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    if "validation" in dataset:
        dataset = dataset["validation"]
    
    total = len(dataset)
    if num_samples is None:
        num_samples = total - start_index
    
    end_index = min(start_index + num_samples, total)
    
    print(f"总样本数: {total}")
    print(f"将可视化: {end_index - start_index} 张 (索引 {start_index} 到 {end_index-1})")
    print(f"输出目录: {output_path}")
    print("-" * 50)
    
    saved_count = 0
    failed_count = 0
    
    for i in range(start_index, end_index):
        item = dataset[i]
        
        image = item.get("image")
        question_id = item.get("question_id", i)
        question = item.get("question", "")
        image_id = item.get("image_id", "")
        image_width = item.get("image_width", 0)
        image_height = item.get("image_height", 0)
        image_classes = item.get("image_classes", [])
        set_name = item.get("set_name", "")
        
        if image is None:
            print(f"[警告] 样本 {i} 没有图像")
            failed_count += 1
            continue
        
        try:
            if isinstance(image, Image.Image):
                img = image.copy()
            else:
                img = Image.open(image).copy()
            
            draw = ImageDraw.Draw(img)
            
            # 计算画布高度（添加文本区域）
            text_lines = []
            text_lines.append(f"Question ID: {question_id}")
            text_lines.append(f"Image ID: {image_id}")
            text_lines.append(f"Image Size: {image_width} x {image_height}")
            text_lines.append(f"Set: {set_name}")
            if image_classes:
                classes_str = ", ".join(image_classes[:5])
                if len(image_classes) > 5:
                    classes_str += f" ... (+{len(image_classes)-5})"
                text_lines.append(f"Classes: {classes_str}")
            text_lines.append("-" * 30)
            text_lines.append(f"Q: {question}")
            
            # 计算文本区域高度
            line_height = font_size + 10
            text_area_height = len(text_lines) * line_height + 20
            
            # 创建新画布（图像 + 文本区域）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            new_img = Image.new('RGB', (img.width, img.height + text_area_height), 'white')
            new_img.paste(img, (0, 0))
            
            draw = ImageDraw.Draw(new_img)
            
            # 尝试加载字体（如果失败使用默认）
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            
            # 绘制文本（白色背景 + 黑色文字）
            y_offset = img.height + 10
            for line in text_lines:
                draw.text((10, y_offset), line, fill='black', font=font)
                y_offset += line_height
            
            # 保存
            safe_image_id = str(image_id).replace("/", "_").replace(":", "_")[:50]
            filename = f"{question_id:05d}_{safe_image_id}.jpg"
            filepath = output_path / filename
            new_img.save(filepath, "JPEG", quality=95)
            
            saved_count += 1
            
            if (i - start_index + 1) % 100 == 0 or (i - start_index + 1) == (end_index - start_index):
                print(f"进度: {i - start_index + 1}/{end_index - start_index} (已保存 {saved_count} 张)")
                
        except Exception as e:
            print(f"[错误] 处理样本 {i} 失败: {e}")
            failed_count += 1
    
    print("-" * 50)
    print(f"完成！")
    print(f"成功保存: {saved_count} 张")
    print(f"失败: {failed_count} 张")
    print(f"输出目录: {output_path.absolute()}")


def show_sample_info(dataset_path: str = "./data", num_samples: int = 5):
    """
    显示数据集样本的详细信息
    """
    print(f"正在加载数据集: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    if "validation" in dataset:
        dataset = dataset["validation"]
    
    print(f"\n总样本数: {len(dataset)}")
    print("\n" + "=" * 60)
    print("数据集字段说明:")
    print("=" * 60)
    
    # 获取第一个样本的字段
    sample = dataset[0]
    
    print("\n可用字段:")
    for key, value in sample.items():
        value_type = type(value).__name__
        if hasattr(value, '__len__') and not isinstance(value, str):
            value_preview = f"{value_type}, len={len(value)}"
        else:
            value_preview = str(value)[:100] if value else "None"
        
        print(f"  - {key}: {value_preview}")
    
    print("\n" + "=" * 60)
    print("示例数据 (前3条):")
    print("=" * 60)
    
    for i in range(min(3, len(dataset))):
        item = dataset[i]
        print(f"\n--- 样本 {i} ---")
        for key, value in item.items():
            if key == "image":
                print(f"  {key}: <Image>")
            elif isinstance(value, list):
                print(f"  {key}: {value[:3]}..." if len(value) > 3 else f"  {key}: {value}")
            elif len(str(value)) > 100:
                print(f"  {key}: {str(value)[:100]}...")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="可视化数据集（图像 + 问题 + 其他信息）")
    parser.add_argument("--dataset-path", type=str, default="./data", help="数据集路径")
    parser.add_argument("--output-dir", type=str, default="./output_visualized", help="输出目录")
    parser.add_argument("--num-samples", type=int, default=None, help="要可视化的样本数")
    parser.add_argument("--start-index", type=int, default=0, help="起始索引")
    parser.add_argument("--font-size", type=int, default=20, help="字体大小")
    parser.add_argument("--show-info", action="store_true", help="仅显示数据集信息")
    
    args = parser.parse_args()
    
    if args.show_info:
        show_sample_info(args.dataset_path)
    else:
        visualize_dataset(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            start_index=args.start_index,
            font_size=args.font_size
        )
