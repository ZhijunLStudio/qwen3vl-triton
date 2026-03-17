import torch
import torch.nn as nn

# ====================================================================
# 1. 我们要手搓的 INT4 算子 (纯 PyTorch 慢速版，仅用于验证逻辑和形状)
# ====================================================================
class PurePyTorchINT4Linear(nn.Module):
    def __init__(self, in_features, out_features, group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        # 注册三个魔法张量：打包后的权重、缩放因子、打包后的零点
        self.register_buffer('qweight', torch.empty((in_features // 2, out_features), dtype=torch.int8))
        self.register_buffer('scales', torch.empty((in_features // group_size, out_features), dtype=torch.float16))
        self.register_buffer('qzeros', torch.empty(((in_features // group_size) // 2, out_features), dtype=torch.int8))

    def forward(self, x):
        """
        这就是 HuggingFace 调用的入口！
        我们会在这里把 INT4 临时解包成 FP16，然后用原生 matmul 算。
        虽然慢，但逻辑绝对正确，形状绝对匹配！
        """
        device = x.device
        
        # --- 第 1 步：解包权重 ---
        # 准备一个空的 FP16 张量装解包后的数据
        qweight_unpacked = torch.zeros((self.in_features, self.out_features), device=device, dtype=torch.float16)
        
        # 用位运算拆解低 4 位和高 4 位
        qweight_unpacked[0::2, :] = (self.qweight & 0x0F).to(torch.float16)
        qweight_unpacked[1::2, :] = ((self.qweight >> 4) & 0x0F).to(torch.float16)
        
        # --- 第 2 步：解包 Zero-points ---
        zeros_unpacked = torch.zeros((self.in_features // self.group_size, self.out_features), device=device, dtype=torch.float16)
        zeros_unpacked[0::2, :] = (self.qzeros & 0x0F).to(torch.float16)
        zeros_unpacked[1::2, :] = ((self.qzeros >> 4) & 0x0F).to(torch.float16)
        
        # --- 第 3 步：反量化 (De-quantization) ---
        # 复制 scales 和 zeros 匹配 in_features 维度
        scales_expanded = self.scales.repeat_interleave(self.group_size, dim=0)
        zeros_expanded = zeros_unpacked.repeat_interleave(self.group_size, dim=0)
        
        # 公式: FP16_W = (INT4_W - Zero) * Scale
        W_fp16 = (qweight_unpacked - zeros_expanded) * scales_expanded
        
        # --- 第 4 步：矩阵乘法！---
        # x 的形状是 [..., in_features], W_fp16 是 [in_features, out_features]
        # 返回形状 [..., out_features]
        return torch.matmul(x, W_fp16)

# ====================================================================
# 2. 沙盒模拟测试
# ====================================================================
def run_sandbox():
    print("="*60)
    print("🔍 INT4 算子替换沙盒实验")
    print("="*60)
    
    # 模拟 Qwen3-VL 某一层的特征维度
    in_features = 2048
    out_features = 2048
    seq_len = 10 # 假设只有 10 个文本 Token
    
    # 1. 这是一个原生 HuggingFace 里的 Linear 层
    standard_linear = nn.Linear(in_features, out_features, bias=False).half().cuda()
    
    # 模拟前向传播的输入数据
    dummy_input = torch.randn((1, seq_len, in_features), dtype=torch.float16, device="cuda")
    
    print("\n[原生状态]")
    print(f"原生算子类型: {type(standard_linear)}")
    print(f"原生权重形状: {standard_linear.weight.shape} (FP16)")
    
    out_standard = standard_linear(dummy_input)
    print(f"✅ HuggingFace 得到输出形状: {out_standard.shape}")
    
    # 2. 我们初始化一个自定义的 INT4 算子
    # 注意：在真实代码中，这里会被填入 extract_w4a16_packed.py 提取出来的字典数据
    my_int4_operator = PurePyTorchINT4Linear(in_features, out_features).cuda()
    
    print("\n[替换状态]")
    print(f"自定义算子类型: {type(my_int4_operator)}")
    print(f"INT8 打包权重形状: {my_int4_operator.qweight.shape}  <--- 体积只有原来 1/4！")
    print(f"Scale 形状: {my_int4_operator.scales.shape}")
    
    # 3. 模拟 Monkey Patch 替换过程
    # 假设有个 DummyModel，里面有个属性叫 q_proj
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = standard_linear
    
    model = DummyModel()
    
    # 【核心动作】：替换它！
    setattr(model, 'q_proj', my_int4_operator)
    
    # Hugging Face 不知情，继续调用
    out_int4 = model.q_proj(dummy_input)
    
    print(f"✅ HuggingFace 得到输出形状: {out_int4.shape}")
    print("\n🎉 结论：只要维度一致，Hugging Face 根本不在乎底层的运算逻辑！")

if __name__ == "__main__":
    run_sandbox()
