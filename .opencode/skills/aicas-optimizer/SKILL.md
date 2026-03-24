---
name: "aicas-optimizer"
description: "AICAS 2026 VLM模型自动优化器 - 基于现有evaluation_wrapper.py持续迭代优化TTFT和吞吐量"
---

# AICAS 2026 VLM 自动优化器

## 目标
- **TTFT**: 降至 ~30ms (当前约65ms)
- **吞吐量**: 提升至 >500 tokens/sec (当前约65 tokens/sec)
- **准确率**: 保持现有水平，变化不超过5%

## 评分权重
- 准确率: 40%
- TTFT提升率: 30%
- 吞吐量提升率: 30%

## 当前基准性能（初始状态）
运行 benchmark 后的结果：
- **TTFT**: 65.47 ms
- **吞吐量**: 65.43 tokens/sec
- **类别命中率**: 35% (7/20) - 通过 compute_accuracy.py 计算
- **加权总分**: 待计算

## 加权总分计算
每次优化后计算：
- 加权总分 = 准确率×0.4 + TTFT提升率×0.3 + 吞吐量提升率×0.3
- TTFT提升率 = (基准TTFT - 当前TTFT) / 基准TTFT
- 吞吐量提升率 = (当前吞吐量 - 基准吞吐量) / 基准吞吐量

## 现有基础
1. **evaluation_wrapper.py** - 已有的优化版本（基于 evaluation_wrapper-origin.py）
2. **benchmark.py** - 性能测试工具
3. **profile_decode.py** - 算子耗时分析工具
4. **compute_accuracy.py** - 准确率（类别命中率）计算脚本

## 准确率（类别命中率）计算方法
运行完 benchmark 后，使用 compute_accuracy.py 计算：

```bash
source /data/lizhijun/anaconda3/bin/activate torch && python compute_accuracy.py --result result.json
```

逻辑：检查模型生成的答案是否包含该图片的 image_classes 类别标签中的任意一个。

## 优化迭代流程

### 阶段 0: 初始评估（必须先执行）
1. 运行基准测试获取当前性能
   ```bash
   cd /data/lizhijun/work/AICAS/AICASGC
   source /data/lizhijun/anaconda3/bin/activate torch
   export CUDA_VISIBLE_DEVICES=7
   python benchmark.py --model-path ./Qwen3-VL-2B-Instruct --dataset-path ./data --output result.json --num-samples 20
   ```
2. 计算类别命中率
   ```bash
   cd /data/lizhijun/work/AICAS/AICASGC
   source /data/lizhijun/anaconda3/bin/activate torch
   python compute_accuracy.py --result result.json
   ```
3. 记录初始指标: TTFT, 吞吐量, 类别命中率

### 阶段 1: 开始优化
基于当前性能瓶颈，自主选择优化方向：
- Vision Encoder 优化（INT4量化、融合等）
- Attention 算子优化（Flash Attention、算子融合等）
- Decode 阶段优化（INT4量化、KV Cache优化等）
- 系统级优化（torch.compile、TF32等）

优化代码直接写入 evaluation_wrapper.py 中。

### 阶段 2: 验证优化效果
每次优化后：
1. 运行 benchmark 测试性能
2. 计算类别命中率
3. 计算加权总分

### 阶段 3: Git 提交
如果加权总分相比上次有提升，执行：
```bash
cd /data/lizhijun/work/AICAS/AICASGC
git add ./
git commit -m "优化: TTFT=Xms, 吞吐量=Y tokens/s, 命中率=Z%, 加权总分=W"
git push origin main
```

## 终止条件
只有满足以下条件才停止优化：
- TTFT ≤ 30ms 且 吞吐量 ≥ 500 tokens/sec

## 测试命令
```bash
cd /data/lizhijun/work/AICAS/AICASGC
source /data/lizhijun/anaconda3/bin/activate torch
export CUDA_VISIBLE_DEVICES=7
python benchmark.py --model-path ./Qwen3-VL-2B-Instruct --dataset-path ./data --output result.json --num-samples 20
```

## 准确率计算命令
```bash
cd /data/lizhijun/work/AICAS/AICASGC
source /data/lizhijun/anaconda3/bin/activate torch
python compute_accuracy.py --result result.json
```

## 关键文件
- `evaluation_wrapper-origin.py` - 原始版本（不修改）
- `evaluation_wrapper.py` - **必须修改的优化文件**
- `benchmark.py` - 性能测试
- `compute_accuracy.py` - 准确率计算
- `result.json` - 测试结果

## 注意事项
1. **必须修改 evaluation_wrapper.py**
2. **遇到错误修复后，继续下一步，不要停**
3. 每次只修改一个优化点，便于定位问题
4. 准确率优先，不能牺牲太多精度
5. 使用第8张卡: CUDA_VISIBLE_DEVICES=7
6. 测试样本数: 20 用于快速验证
7. 每次优化后都要运行 benchmark + compute_accuracy 验证性能
