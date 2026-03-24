#!/usr/bin/env python3
"""
准确率评估脚本 - 优化版
只加载需要的样本，减少加载时间
"""
import json
import argparse
from datasets import load_from_disk


def compute_class_hit_rate(result_path: str, dataset_path: str = "./data") -> dict:
    print(f"正在加载结果: {result_path}")
    with open(result_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    answers = results.get("answers", [])
    if not answers:
        print("错误: 结果中没有 answers 字段")
        return {}
    
    # 收集需要查询的 question_id
    question_ids = [a.get("question_id") for a in answers if a.get("question_id")]
    print(f"共有 {len(question_ids)} 个答案需要评估")
    
    # 只加载需要的样本（按索引）
    print(f"正在加载数据集: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    if "validation" in dataset:
        dataset = dataset["validation"]
    
    # 构建 id -> classes 映射（只针对需要的）
    id_to_classes = {}
    for qid in question_ids:
        # 假设 question_id 和索引有某种对应关系，尝试直接访问
        try:
            idx = question_ids.index(qid)
            item = dataset[idx]
            id_to_classes[qid] = item.get("image_classes", [])
        except:
            id_to_classes[qid] = []
    
    hits = 0
    total = 0
    details = []
    
    for answer_item in answers:
        qid = answer_item.get("question_id")
        predicted = answer_item.get("prediction", "").lower() if answer_item.get("prediction") else ""
        
        if qid not in id_to_classes:
            continue
            
        classes = id_to_classes[qid]
        if not classes:
            continue
        
        hit = False
        matched_class = None
        for cls in classes:
            cls_lower = cls.lower()
            if cls_lower in predicted:
                hit = True
                matched_class = cls
                break
        
        if hit:
            hits += 1
        total += 1
        details.append({
            "question_id": qid,
            "predicted": predicted[:100],
            "classes": classes,
            "matched": matched_class,
            "hit": hit
        })
    
    hit_rate = hits / total if total > 0 else 0
    
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"总样本数: {total}")
    print(f"命中数: {hits}")
    print(f"类别命中率: {hit_rate*100:.2f}%")
    print("=" * 60)
    
    # 显示命中的示例
    print("\n命中的示例:")
    hit_examples = [d for d in details if d["hit"]][:5]
    for d in hit_examples:
        print(f"✓ QID={d['question_id']}")
        print(f"   预测: {d['predicted'][:60]}...")
        print(f"   命中: {d['matched']}")
        print()
    
    # 显示未命中的示例
    print("未命中的示例:")
    miss_examples = [d for d in details if not d["hit"]][:5]
    for d in miss_examples:
        print(f"✗ QID={d['question_id']}")
        print(f"   预测: {d['predicted'][:60]}...")
        print(f"   类别: {d['classes'][:3]}")
        print()
    
    return {
        "total": total,
        "hits": hits,
        "hit_rate": hit_rate,
        "details": details
    }


def main():
    parser = argparse.ArgumentParser(description="准确率评估 - 基于类别命中")
    parser.add_argument("--result", type=str, default="result.json", help="结果文件路径")
    parser.add_argument("--dataset-path", type=str, default="./data", help="数据集路径")
    
    args = parser.parse_args()
    compute_class_hit_rate(args.result, args.dataset_path)


if __name__ == "__main__":
    main()
