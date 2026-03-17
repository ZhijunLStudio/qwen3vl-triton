import torch
from transformers import AutoModelForImageTextToText
from tqdm import tqdm

def pack_int4_to_int8(qweight_int4):
    """
    核心魔法：把 INT4 压进 INT8！
    假设输入形状是 [in_features, out_features]
    由于 2 个 INT4 拼成 1 个 INT8，输出形状变为 [in_features // 2, out_features]
    """
    # 确保是 0-15 之间的无符号整数
    qweight_int4 = qweight_int4.to(torch.uint8) & 0x0F
    
    # 按照 in_features 维度，两两分组
    # q_low 是低 4 位，q_high 是高 4 位
    q_low = qweight_int4[0::2, :]
    q_high = qweight_int4[1::2, :]
    
    # 拼装：(高位 << 4) | (低位)
    packed_qweight = (q_high << 4) | q_low
    return packed_qweight.to(torch.int8)

# 修改 extract_w4a16_packed.py 中的提取逻辑
def extract_w4a16_grouped_scales():
    print("="*60)
    print("🚀 启动离线 W4A16 (INT4) 分组量化与【算子融合】打包压榨...")
    print("="*60)
    
    model_path = "./Qwen3-VL-2B-Instruct"
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu"
    )
    
    group_size = 128
    quant_dict = {}
    layers = model.model.language_model.layers
    
    def quantize_and_pack(W, group_size):
        """把之前写在循环里的量化逻辑抽成函数"""
        in_features, out_features = W.shape
        W_grouped = W.reshape(-1, group_size, out_features)
        
        W_max = W_grouped.max(dim=1, keepdim=True)[0]
        W_min = W_grouped.min(dim=1, keepdim=True)[0]
        
        max_int = 15.0
        scales = (W_max - W_min) / max_int
        scales = torch.clamp(scales, min=1e-5)
        
        zeros = -torch.round(W_min / scales)
        zeros = torch.clamp(zeros, 0, max_int)
        
        W_q = torch.round(W_grouped / scales + zeros)
        W_q = torch.clamp(W_q, 0, max_int).to(torch.uint8)
        
        W_q = W_q.reshape(in_features, out_features)
        scales = scales.reshape(-1, out_features).to(torch.float16)
        zeros = zeros.reshape(-1, out_features)
        
        return pack_int4_to_int8(W_q), scales, pack_int4_to_int8(zeros)

    for idx, layer in enumerate(tqdm(layers, desc="Fusion & Quantization")):
        # ---------------------------------------------------------
        # 1. 融合 QKV (注意：Qwen 是 GQA，Q, K, V 的 out_features 可能不同)
        # ---------------------------------------------------------
        W_q = layer.self_attn.q_proj.weight.data.t()
        W_k = layer.self_attn.k_proj.weight.data.t()
        W_v = layer.self_attn.v_proj.weight.data.t()
        
        # 记录维度，推理拆分时要用
        q_dim, k_dim, v_dim = W_q.shape[1], W_k.shape[1], W_v.shape[1]
        
        # 物理拼接！dim=1 因为我们已经 t() 转置成了 [in, out]
        W_qkv = torch.cat([W_q, W_k, W_v], dim=1).contiguous()
        
        q_packed, q_scales, q_zeros = quantize_and_pack(W_qkv, group_size)
        quant_dict[f"layers.{idx}.self_attn.qkv_proj"] = {
            'qweight': q_packed, 'scales': q_scales, 'qzeros': q_zeros,
            'dims': (q_dim, k_dim, v_dim) # 记录维度信息
        }
        
        # o_proj 保持不变
        W_o = layer.self_attn.o_proj.weight.data.t().contiguous()
        o_packed, o_scales, o_zeros = quantize_and_pack(W_o, group_size)
        quant_dict[f"layers.{idx}.self_attn.o_proj"] = {
            'qweight': o_packed, 'scales': o_scales, 'qzeros': o_zeros
        }

        # ---------------------------------------------------------
        # 2. 融合 MLP (Gate & Up)
        # ---------------------------------------------------------
        W_gate = layer.mlp.gate_proj.weight.data.t()
        W_up = layer.mlp.up_proj.weight.data.t()
        
        gate_dim, up_dim = W_gate.shape[1], W_up.shape[1]
        
        W_gate_up = torch.cat([W_gate, W_up], dim=1).contiguous()
        gu_packed, gu_scales, gu_zeros = quantize_and_pack(W_gate_up, group_size)
        
        quant_dict[f"layers.{idx}.mlp.gate_up_proj"] = {
            'qweight': gu_packed, 'scales': gu_scales, 'qzeros': gu_zeros,
            'dims': (gate_dim, up_dim)
        }
        
        # down_proj 保持不变
        W_down = layer.mlp.down_proj.weight.data.t().contiguous()
        d_packed, d_scales, d_zeros = quantize_and_pack(W_down, group_size)
        quant_dict[f"layers.{idx}.mlp.down_proj"] = {
            'qweight': d_packed, 'scales': d_scales, 'qzeros': d_zeros
        }

    torch.save(quant_dict, "qwen3_2b_int4_fused_packed.pt")
    print("\n✅ INT4 融合打包完成！Launch 次数即将暴降！")


if __name__ == "__main__":
    extract_w4a16_grouped_scales()
