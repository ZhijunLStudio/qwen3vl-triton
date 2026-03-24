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

## 现有基础
1. **evaluation_wrapper.py** - 已有的优化版本（基于 evaluation_wrapper-origin.py）
2. **benchmark.py** - 性能测试工具
3. **profile_decode.py** - 算子耗时分析工具
4. **compute_accuracy.py** - 准确率（类别命中率）计算脚本
5. 已有 INT4 量化 Vision Encoder 的基础实现

## 准确率（类别命中率）计算方法
运行完 benchmark 后，使用 compute_accuracy.py 计算：

```bash
source /data/lizhijun/anaconda3/bin/activate torch && python compute_accuracy.py --result result.json
```

逻辑：检查模型生成的答案是否包含该图片的 image_classes 类别标签中的任意一个。

## 加权总分计算
每次优化后计算：
- 加权总分 = 准确率×0.4 + TTFT提升率×0.3 + 吞吐量提升率×0.3
- TTFT提升率 = (基准TTFT - 当前TTFT) / 基准TTFT
- 吞吐量提升率 = (当前吞吐量 - 基准吞吐量) / 基准吞吐量

## 代码整理规范
优化后的代码必须放在 `optimizer/` 目录下，保持整洁：
```
optimizer/
├── __init__.py
├── vision_encoder.py    # Vision Encoder 优化（INT4量化、融合等）
├── attention.py          # Attention 算子优化（Triton内核等）
├── decode.py            # Decode 阶段优化
├── utils.py             # 通用工具函数
└── config.py            # 优化配置
```

**重要**: 不要修改外层的 `evaluation_wrapper-origin.py`，所有优化在 `optimizer/` 目录下进行，最终通过修改 `evaluation_wrapper.py` 来使用。

## 优化迭代流程

### 阶段 0: 初始评估（必须先执行）
1. 运行基准测试获取当前性能
   ```bash
   source /data/lizhijun/anaconda3/bin/activate torch && export CUDA_VISIBLE_DEVICES=7 && python benchmark.py --model-path ./Qwen3-VL-2B-Instruct --dataset-path ./data --output result.json --num-samples 20
   ```
2. 计算类别命中率
   ```bash
   source /data/lizhijun/anaconda3/bin/activate torch && python compute_accuracy.py --result result.json
   ```
3. 记录初始指标: TTFT, 吞吐量, 类别命中率
4. 创建 optimization_log.json 记录初始状态

### 阶段 1: Vision Encoder 优化（TTFT）
**优先级**: 高

在 `optimizer/vision_encoder.py` 中实现：
1. INT4 量化 Vision Encoder（已在 evaluation_wrapper.py 中有基础）
2. 融合 Patch Embedding 层
3. 优化 Attention 计算（使用 Flash Attention）

检查点:
- TTFT 是否降低?
- 类别命中率是否保持?

### 阶段 2: Prefill 阶段优化（TTFT）
**优先级**: 高

在 `optimizer/attention.py` 中实现：
1. Flash Attention 启用（确保模型配置中启用）
2. QKV 融合
3. 算子融合（RMSNorm + Attention）

检查点:
- TTFT 是否降低到 30ms 左右?
- 是否引入精度损失?

### 阶段 3: Decode 阶段优化（吞吐量）
**优先级**: 高

在 `optimizer/decode.py` 中实现：
1. INT4 量化 LLM 层
2. 算子融合（Triton 内核）
3. KV Cache 优化
4. 动态批处理

检查点:
- 吞吐量是否超过 500 tokens/sec?
- TTFT 是否受影响?

### 阶段 4: 系统级优化
**优先级**: 中

1. torch.compile 优化
2. TF32 启用
3. CUDA Graph 优化
4. 内存优化

## 测试命令
```bash
source /data/lizhijun/anaconda3/bin/activate torch && export CUDA_VISIBLE_DEVICES=7 && python benchmark.py --model-path ./Qwen3-VL-2B-Instruct --dataset-path ./data --output result.json --num-samples 20
```

## 性能分析命令
```bash
source /data/lizhijun/anaconda3/bin/activate torch && export CUDA_VISIBLE_DEVICES=7 && python profile_decode.py
```

## 准确率计算命令
```bash
source /data/lizhijun/anaconda3/bin/activate torch && python compute_accuracy.py --result result.json
```

## Git 提交和推送（每次有提升时执行）
当加权总分相比上次有提升时，执行：

```bash
git add ./
git commit -m "优化: TTFT=Xms, 吞吐量=Y tokens/s, 命中率=Z%"
git push origin main
```

注意：如果 push 失败，可能需要先启动代理：
```bash
proxy  # 启用代理
git push origin main
unproxy  # 关闭代理
```

## 优化记录
每次优化后，必须更新 `optimization_log.json`，记录：
- 优化内容描述
- TTFT 变化
- 吞吐量变化
- 类别命中率变化
- 加权总分
- 是否满足终止条件

## 终止条件
满足以下任一条件停止优化:
1. 连续 3 次优化无性能提升
2. 类别命中率下降超过 5%
3. TTFT ≤ 30ms 且 吞吐量 ≥ 500 tokens/sec
4. 达到最大迭代次数 (20次)

## 关键文件
- `evaluation_wrapper-origin.py` - 原始版本（不修改）
- `evaluation_wrapper.py` - 已有优化版本（在此基础上修改）
- `optimizer/` - 新增优化代码目录
- `benchmark.py` - 性能测试
- `profile_decode.py` - 性能分析
- `compute_accuracy.py` - 准确率计算
- `result.json` - 测试结果
- `optimization_log.json` - 优化记录

## 注意事项
1. **基于现有代码**: 在 evaluation_wrapper.py 基础上优化，不要从头开始
2. **保持代码整洁**: 新代码放 optimizer/ 目录
3. **每次只修改一个点**: 便于定位问题
4. **准确率优先**: 不能为了速度牺牲太多精度
5. **使用第8张卡**: CUDA_VISIBLE_DEVICES=7
6. **测试样本数**: 20 用于快速验证
7. **每次优化后都要运行 benchmark + compute_accuracy**: 验证性能
8. **Git操作**: 每次加权总分提升时自动 commit 和 push
