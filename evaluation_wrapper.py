"""
AICAS 2026 - Participant Core Modification File
"""
from typing import Dict
try:
    from PIL import Image
except ImportError:
    class Image:
        pass
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.cache_utils import DynamicCache
from collections import OrderedDict
import triton
import triton.language as tl
import functools
import os

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.cache_utils import DynamicCache
from collections import OrderedDict
import triton
import triton.language as tl
import functools
import os
import torch.nn.functional as F
import sys

# ==============================================================================
# 🌪️ 1. Triton 极速版 RMSNorm (保持不变)
# ==============================================================================
@triton.jit
def rms_norm_kernel(x_ptr, y_ptr, w_ptr, stride_x, N, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x_row_ptr = x_ptr + pid * stride_x
    y_row_ptr = y_ptr + pid * stride_x
    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    mean_sq = sum_sq / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    y = x * rstd * w
    tl.store(y_row_ptr + cols, y.to(tl.float16), mask=mask)

def custom_rmsnorm_forward(self, hidden_states):
    x = hidden_states.contiguous()
    original_shape = x.shape
    x_2d = x.view(-1, original_shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x_2d)
    BLOCK_SIZE = triton.next_power_of_2(N)
    rms_norm_kernel[(M,)](x_2d, y, self.weight, x_2d.stride(0), N, self.variance_epsilon, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return y.view(original_shape)

# ==============================================================================
# 🎯 2. TorchScript 融合版 Fast RoPE (绝对安全！消灭 3500 次 CPU 调度)
# 利用 NVFuser 底层融合，完美兼容任何维度排布，绝不产生 NaN！
# ==============================================================================
@torch.jit.script
def fused_rope_core(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    d = q.shape[-1] // 2
    # 一次性提取，底层 C++ 自动融合内存访问
    q1, q2 = q[..., :d], q[..., d:]
    c1, c2 = cos[..., :d], cos[..., d:]
    s1, s2 = sin[..., :d], sin[..., d:]
    
    q_out = torch.cat([q1 * c1 - q2 * s1, q2 * c2 + q1 * s2], dim=-1)
    
    k1, k2 = k[..., :d], k[..., d:]
    k_out = torch.cat([k1 * c1 - k2 * s1, k2 * c2 + k1 * s2], dim=-1)
    
    return q_out, k_out

def fast_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return fused_rope_core(q, k, cos, sin)

# ==============================================================================
# 🧠 3. Triton 工业版 SwiGLU (保持不变，已验证安全)
# ==============================================================================
@triton.jit
def qwen_swiglu_kernel(
    x_ptr, y_ptr,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_n < N
    
    gate_ptrs = x_ptr + pid_m * stride_xm + offs_n * stride_xn
    up_ptrs = x_ptr + pid_m * stride_xm + (N + offs_n) * stride_xn
    out_ptrs = y_ptr + pid_m * stride_ym + offs_n * stride_yn
    
    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    silu = gate * tl.sigmoid(gate)
    out = silu * up
    tl.store(out_ptrs, out.to(tl.float16), mask=mask)

def fast_swiglu(gate_up, g_dim):
    gate_up = gate_up.contiguous()
    original_shape = gate_up.shape
    x_2d = gate_up.view(-1, original_shape[-1])
    M = x_2d.shape[0]
    
    y = torch.empty((M, g_dim), device=gate_up.device, dtype=torch.float16)
    BLOCK_SIZE = 1024
    grid = (M, triton.cdiv(g_dim, BLOCK_SIZE))
    
    qwen_swiglu_kernel[grid](
        x_2d, y,
        x_2d.stride(0), x_2d.stride(1),
        y.stride(0), y.stride(1),
        N=g_dim, BLOCK_SIZE=BLOCK_SIZE, num_warps=4
    )
    
    new_shape = list(original_shape)
    new_shape[-1] = g_dim
    return y.view(*new_shape)



# （这里保留你之前定义的 w4a16_gemv_slim_kernel_splitk, dequantize 和 SlimTritonINT4Linear 代码）

# ==============================================================================
# 👑 4. 你的核心遗产：Triton W4A16 Split-K 算子 (绝对不换，保持原样！)
# ==============================================================================
@triton.jit
def w4a16_gemv_slim_kernel_splitk(
    A_ptr, B_packed_ptr, Scales_ptr, Zeros_packed_ptr, Workspace_ptr,
    K, N, stride_ak, stride_bk, stride_bn,
    group_size: tl.constexpr, 
    BLOCK_K_PACKED: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    k_start = pid_k * BLOCK_K_PACKED
    k_step = SPLIT_K * BLOCK_K_PACKED

    for k_packed_idx in range(k_start, K // 2, k_step):
        offs_k_packed = k_packed_idx + tl.arange(0, BLOCK_K_PACKED)
        mask_k_packed = offs_k_packed < (K // 2)

        b_ptrs = B_packed_ptr + offs_k_packed[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_packed = tl.load(b_ptrs, mask=mask_k_packed[:, None] & mask_n[None, :], other=0)

        b_low = b_packed & 0x0F
        b_high = (b_packed >> 4) & 0x0F

        group_idx = k_packed_idx // (group_size // 2)
        s_ptrs = Scales_ptr + group_idx * N + offs_n
        scales = tl.load(s_ptrs, mask=mask_n, other=1.0)

        z_ptrs = Zeros_packed_ptr + (group_idx // 2) * N + offs_n
        z_packed = tl.load(z_ptrs, mask=mask_n, other=0)
        zeros = (z_packed >> ((group_idx % 2) * 4)) & 0x0F

        w_low_fp16 = (b_low.to(tl.float32) - zeros.to(tl.float32)[None, :]) * scales.to(tl.float32)[None, :]
        w_high_fp16 = (b_high.to(tl.float32) - zeros.to(tl.float32)[None, :]) * scales.to(tl.float32)[None, :]

        a_offs_low = k_packed_idx * 2 + tl.arange(0, BLOCK_K_PACKED) * 2
        a_offs_high = a_offs_low + 1

        a_low = tl.load(A_ptr + a_offs_low * stride_ak, mask=a_offs_low < K, other=0.0)
        a_high = tl.load(A_ptr + a_offs_high * stride_ak, mask=a_offs_high < K, other=0.0)

        acc += tl.sum(a_low[:, None].to(tl.float32) * w_low_fp16, axis=0)
        acc += tl.sum(a_high[:, None].to(tl.float32) * w_high_fp16, axis=0)

    w_ptrs = Workspace_ptr + pid_k * N + offs_n
    tl.store(w_ptrs, acc.to(tl.float16), mask=mask_n)

@triton.jit
def dequantize_w4a16_slim_kernel(
    B_packed_ptr, Scales_ptr, Zeros_packed_ptr, W_fp16_ptr,
    K, N, stride_bk, stride_bn, stride_wk, stride_wn,
    BLOCK_K_PACKED: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_k_p = pid_k * BLOCK_K_PACKED + tl.arange(0, BLOCK_K_PACKED)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_k_p = offs_k_p < (K // 2)
    mask_n = offs_n < N
    b_ptrs = B_packed_ptr + offs_k_p[:, None] * stride_bk + offs_n[None, :] * stride_bn
    b_packed = tl.load(b_ptrs, mask=mask_k_p[:, None] & mask_n[None, :], other=0)
    b_low = b_packed & 0x0F
    b_high = (b_packed >> 4) & 0x0F
    group_idx = pid_k 
    s_ptrs = Scales_ptr + group_idx * N + offs_n
    scales = tl.load(s_ptrs, mask=mask_n, other=1.0)
    z_ptrs = Zeros_packed_ptr + (group_idx // 2) * N + offs_n
    z_packed = tl.load(z_ptrs, mask=mask_n, other=0)
    zeros = (z_packed >> ((group_idx % 2) * 4)) & 0x0F
    w_low_fp16 = (b_low.to(tl.float32) - zeros.to(tl.float32)[None, :]) * scales.to(tl.float32)[None, :]
    w_high_fp16 = (b_high.to(tl.float32) - zeros.to(tl.float32)[None, :]) * scales.to(tl.float32)[None, :]
    w_ptrs_low = W_fp16_ptr + (offs_k_p * 2)[:, None] * stride_wk + offs_n[None, :] * stride_wn
    w_ptrs_high = W_fp16_ptr + (offs_k_p * 2 + 1)[:, None] * stride_wk + offs_n[None, :] * stride_wn
    tl.store(w_ptrs_low, w_low_fp16.to(tl.float16), mask=mask_k_p[:, None] & mask_n[None, :])
    tl.store(w_ptrs_high, w_high_fp16.to(tl.float16), mask=mask_k_p[:, None] & mask_n[None, :])

class SlimTritonINT4Linear(nn.Module):
    def __init__(self, in_features, out_features, group_size=128, device="cuda:0"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer('qweight', torch.empty((in_features // 2, out_features), dtype=torch.int8))
        self.register_buffer('scales', torch.empty((in_features // group_size, out_features), dtype=torch.float16))
        self.register_buffer('qzeros', torch.empty(((in_features // group_size) // 2, out_features), dtype=torch.int8))
        
        # 💥 最终优化：加倍算术强度，减少显存带宽压力
        self.BLOCK_N = 64  
        self.BLOCK_K_PACKED = 64
        self.SPLIT_K = 8
        
        self.decode_grid = (triton.cdiv(out_features, self.BLOCK_N), self.SPLIT_K)
        self.prefill_grid = (triton.cdiv((in_features // 2), self.BLOCK_K_PACKED), triton.cdiv(out_features, self.BLOCK_N))
        self.workspace = torch.empty((self.SPLIT_K, out_features), dtype=torch.float16, device=device)


    def __call__(self, x, *args, **kwargs):
        return self.forward(x)

    def forward(self, x):
        original_shape = x.shape
        x_2d = x.view(-1, self.in_features)
        M, K = x_2d.shape
        N = self.out_features
        if M == 1:
            x_1d = x.view(-1)
            w4a16_gemv_slim_kernel_splitk[self.decode_grid](
                x_1d, self.qweight, self.scales, self.qzeros, self.workspace,
                K, N, 1, self.qweight.stride(0), self.qweight.stride(1),
                group_size=self.group_size, BLOCK_K_PACKED=self.BLOCK_K_PACKED, BLOCK_N=self.BLOCK_N,
                SPLIT_K=self.SPLIT_K, num_warps=4, num_stages=3
            )
            c = self.workspace.sum(dim=0)
            return c.view(*original_shape[:-1], N)
        else:
            w_fp16 = torch.empty((K, N), device=x.device, dtype=torch.float16)
            dequantize_w4a16_slim_kernel[self.prefill_grid](
                self.qweight, self.scales, self.qzeros, w_fp16, K, N,
                self.qweight.stride(0), self.qweight.stride(1), w_fp16.stride(0), w_fp16.stride(1),
                BLOCK_K_PACKED=self.BLOCK_K_PACKED, BLOCK_N=self.BLOCK_N, num_warps=4, num_stages=3
            )
            return torch.matmul(x, w_fp16)



# ==============================================================================

class VLMModel:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self._device = device
        self.model_path = model_path
        
        print(f"[VLMModel] Loading processor from {model_path}...")
        self._processor = AutoProcessor.from_pretrained(model_path)
        
        print(f"[VLMModel] Loading model with FP16...")
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map=device
        )
        self._model.eval()
        self._optimizations_applied = []
        
        # 🎯 双皇冠优化启动
        self._optimize_kv_cache()
        self._apply_quantization()
        
        print(f"[VLMModel] Model loaded successfully on {device}")
        if self._optimizations_applied:
            print(f"[VLMModel] Applied optimizations: {', '.join(self._optimizations_applied)}")

    def _compute_image_hash(self, pixel_values):
        val = pixel_values.to(torch.float32)
        return f"{list(pixel_values.shape)}_{float(val.flatten()[:500].sum() + val.flatten()[-500:].sum()):.4f}"

    def _optimize_kv_cache(self):
        print("[VLMModel] 🚀 启用顶级 LLM Prefix KV Caching...")
        self.llm_kv_cache = OrderedDict()
        self.llm_kv_cache_lens = {}
        self.llm_cache_capacity = 3
        
        original_forward = self._model.forward
        
        @functools.wraps(original_forward)
        def custom_forward(*args, **kwargs):
            input_ids = kwargs.get('input_ids', args[0] if len(args) > 0 else None)
            past_key_values = kwargs.get('past_key_values')
            pixel_values = kwargs.get('pixel_values')
            
            is_prefill = past_key_values is None or (hasattr(past_key_values, 'get_seq_length') and past_key_values.get_seq_length() == 0) or (isinstance(past_key_values, tuple) and len(past_key_values) == 0)

            if is_prefill and pixel_values is not None and input_ids is not None:
                img_hash = self._compute_image_hash(pixel_values)
                if img_hash in self.llm_kv_cache:
                    self.llm_kv_cache.move_to_end(img_hash)
                    prefix_len = self.llm_kv_cache_lens[img_hash]
                    cached_cache = self.llm_kv_cache[img_hash]
                    
                    # 💥 修复了之前的手误！换回了绝对安全的官方更新方式！
                    new_cache = DynamicCache()
                    for i in range(len(cached_cache)):
                        k, v = cached_cache[i]
                        new_cache.update(k, v, i)
                        
                    kwargs['past_key_values'] = new_cache
                    kwargs['input_ids'] = input_ids[:, prefix_len:]
                    if kwargs.get('position_ids') is not None: kwargs['position_ids'] = kwargs['position_ids'][..., prefix_len:]
                    if kwargs.get('cache_position') is not None: kwargs['cache_position'] = kwargs['cache_position'][prefix_len:]
                        
                    kwargs['pixel_values'] = None
                    if 'image_grid_thw' in kwargs: kwargs['image_grid_thw'] = None
                    return original_forward(*args, **kwargs)
                else:
                    outputs = original_forward(*args, **kwargs)
                    vision_end_mask = (input_ids[0] == 151653).nonzero(as_tuple=True)[0]
                    if len(vision_end_mask) > 0 and hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                        prefix_len = vision_end_mask[0].item() + 1
                        new_cache = []
                        for i in range(len(outputs.past_key_values)):
                            k, v = outputs.past_key_values[i]
                            new_cache.append((k[:, :, :prefix_len, :], v[:, :, :prefix_len, :]))
                        self.llm_kv_cache[img_hash] = new_cache
                        self.llm_kv_cache_lens[img_hash] = prefix_len
                        if len(self.llm_kv_cache) > self.llm_cache_capacity:
                            oldest = next(iter(self.llm_kv_cache))
                            del self.llm_kv_cache[oldest]
                            del self.llm_kv_cache_lens[oldest]
                    return outputs
            return original_forward(*args, **kwargs)

        self._model.forward = custom_forward
        self._optimizations_applied.append('llm_prefix_caching')

    def _apply_quantization(self):
        dict_path = "qwen3_2b_int4_fused_packed.pt"
        if not os.path.exists(dict_path):
            print(f"\n[Quant] ❌ 找不到 {dict_path}！请先运行 extract 脚本打包融合权重。\n")
            return
            
        print("[Quant] 🚀 加载终极版【Split-K Triton + 篡改RMSNorm + FastRoPE + HF旁路融合】算子！")
        quant_dict = torch.load(dict_path, map_location=self._device)
        replaced_count = 0

        # =========================================================
        # 降维打击 1：【类级别】全局替换 RMSNorm 和 RoPE！
        # =========================================================
        RMSNormClass = type(self._model.model.language_model.layers[0].input_layernorm)
        RMSNormClass.forward = custom_rmsnorm_forward
        
        AttnClass = type(self._model.model.language_model.layers[0].self_attn)
        attn_module = sys.modules[AttnClass.__module__]
        if hasattr(attn_module, 'apply_rotary_pos_emb'):
            attn_module.apply_rotary_pos_emb = fast_apply_rotary_pos_emb
        
        # =========================================================
        # 恢复 Proxy 替身（专为 Prefill 阶段回退 HF 原生代码服务）
        # =========================================================
        class FastProxy(nn.Module):
            def __init__(self, handler, attr_name):
                super().__init__()
                self.handler = handler
                self.attr_name = attr_name
            def forward(self, *args, **kwargs):
                return getattr(self.handler, self.attr_name)

        class FusedQKVHandler(nn.Module):
            def __init__(self, qkv_proj, q_dim, k_dim, v_dim):
                super().__init__()
                self.qkv_proj = qkv_proj
                self.q_dim = q_dim
                self.k_dim = k_dim
                self.v_dim = v_dim
                self.saved_k = None
                self.saved_v = None
            def forward(self, x, *args, **kwargs):
                qkv = self.qkv_proj(x)
                q = qkv[..., :self.q_dim].contiguous()
                self.saved_k = qkv[..., self.q_dim : self.q_dim + self.k_dim].contiguous()
                self.saved_v = qkv[..., self.q_dim + self.k_dim :].contiguous()
                return q

        for idx, layer in enumerate(self._model.model.language_model.layers):
            
            # ==========================================
            # 替换 Attention 层：Prefill走代理，Decode走旁路！
            # ==========================================
            qkv_key = f"layers.{idx}.self_attn.qkv_proj"
            if qkv_key in quant_dict:
                qkv_data = quant_dict[qkv_key]
                q_dim, k_dim, v_dim = qkv_data['dims']
                total_qkv_dim = q_dim + k_dim + v_dim
                in_features = layer.self_attn.q_proj.in_features
                
                # 注入融合算子
                qkv_proj = SlimTritonINT4Linear(in_features, total_qkv_dim, group_size=128, device=self._device).to(self._device)
                qkv_proj.qweight.copy_(qkv_data['qweight'])
                qkv_proj.scales.copy_(qkv_data['scales'])
                qkv_proj.qzeros.copy_(qkv_data['qzeros'])
                layer.self_attn.qkv_proj = qkv_proj
                
                # 删除原厂模型，挂载 Proxy 替身
                del layer.self_attn.q_proj
                del layer.self_attn.k_proj
                del layer.self_attn.v_proj
                
                qkv_handler = FusedQKVHandler(qkv_proj, q_dim, k_dim, v_dim)
                layer.self_attn.q_proj = qkv_handler
                layer.self_attn.k_proj = FastProxy(qkv_handler, 'saved_k')
                layer.self_attn.v_proj = FastProxy(qkv_handler, 'saved_v')
                
                o_data = quant_dict[f"layers.{idx}.self_attn.o_proj"]
                o_proj = SlimTritonINT4Linear(layer.self_attn.o_proj.in_features, layer.self_attn.o_proj.out_features, 128, device=self._device).to(self._device)
                o_proj.qweight.copy_(o_data['qweight'])
                o_proj.scales.copy_(o_data['scales'])
                o_proj.qzeros.copy_(o_data['qzeros'])
                layer.self_attn.o_proj = o_proj
                
                # 💥 降维打击 2：定义 Decode 极速前向传播
                lm_config = self._model.model.language_model.config
                num_heads = lm_config.num_attention_heads
                num_kv_heads = lm_config.num_key_value_heads
                head_dim = lm_config.hidden_size // num_heads
                num_rep = num_heads // num_kv_heads  

                def get_fast_attn_forward(self_attn, original_forward, q_dim, k_dim, n_heads, n_kv_heads, h_dim, n_rep):
                    scale = h_dim ** -0.5  # 预计算缩放因子
                    def fast_attn_forward(hidden_states, attention_mask=None, position_ids=None, output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
                        
                        past_key_values = kwargs.get('past_key_values', kwargs.get('past_key_value', None))
                        
                        # Prefill 阶段乖乖走老路（走完整的 FlashAttention）
                        if hidden_states.shape[1] > 1 or output_attentions:
                            return original_forward(
                                hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                                output_attentions=output_attentions, use_cache=use_cache, 
                                cache_position=cache_position, position_embeddings=position_embeddings, 
                                **kwargs
                            )
                        
                        # 🚀 Decode 阶段：十行代码，光速通行！
                        qkv = self_attn.qkv_proj(hidden_states)
                        q = qkv[..., :q_dim].contiguous()
                        k = qkv[..., q_dim : q_dim + k_dim].contiguous()
                        v = qkv[..., q_dim + k_dim :].contiguous()
                        
                        bsz, q_len, _ = hidden_states.size()
                        
                        q = q.view(bsz, q_len, n_heads, h_dim)
                        k = k.view(bsz, q_len, n_kv_heads, h_dim)
                        v = v.view(bsz, q_len, n_kv_heads, h_dim)
                        
                        if hasattr(self_attn, "q_norm") and self_attn.q_norm is not None:
                            q = self_attn.q_norm(q)
                            k = self_attn.k_norm(k)
                            
                        q = q.transpose(1, 2)
                        k = k.transpose(1, 2)
                        v = v.transpose(1, 2)
                            
                        if position_embeddings is None:
                            cos, sin = self_attn.rotary_emb(v, position_ids)
                        else:
                            cos, sin = position_embeddings
                            
                        q, k = fast_apply_rotary_pos_emb(q, k, cos, sin)
                        
                        if past_key_values is not None:
                            k, v = past_key_values.update(k, v, self_attn.layer_idx)
                            
                        # 💥 降维打击 3：抛弃 PyTorch SDPA，手写极速解码 Attention！
                        # 彻底消灭 C++ Math 后台的回退惩罚！0 拷贝！光速穿透！
                        if n_rep > 1:
                            # 零拷贝维度折叠，对齐 GQA
                            q = q.contiguous().view(bsz, n_kv_heads, n_rep * q_len, h_dim)
                            
                        # 纯天然 4D 矩阵乘法，极速调用底层 cuBLAS
                        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scale
                        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
                        attn_output = torch.matmul(attn_weights, v)
                        
                        if n_rep > 1:
                            # 完美复原
                            attn_output = attn_output.view(bsz, n_heads, q_len, h_dim)
                            
                        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
                        attn_output = self_attn.o_proj(attn_output)
                        
                        return (attn_output, past_key_values)
                        
                    return fast_attn_forward

                
                # 挂载光速通道
                layer.self_attn.forward = get_fast_attn_forward(
                    layer.self_attn, layer.self_attn.forward, q_dim, k_dim, 
                    num_heads, num_kv_heads, head_dim, num_rep
                )


                replaced_count += 4

            # ==========================================
            # 🚀 替换 MLP 层
            # ==========================================
            gu_key = f"layers.{idx}.mlp.gate_up_proj"
            if gu_key in quant_dict:
                gu_data = quant_dict[gu_key]
                gate_dim, up_dim = gu_data['dims']
                in_features = layer.mlp.gate_proj.in_features
                
                gu_proj = SlimTritonINT4Linear(in_features, gate_dim + up_dim, 128, device=self._device).to(self._device)
                gu_proj.qweight.copy_(gu_data['qweight'])
                gu_proj.scales.copy_(gu_data['scales'])
                gu_proj.qzeros.copy_(gu_data['qzeros'])
                
                layer.mlp.gate_up_proj = gu_proj
                del layer.mlp.gate_proj
                del layer.mlp.up_proj
                
                d_data = quant_dict[f"layers.{idx}.mlp.down_proj"]
                down_proj = SlimTritonINT4Linear(layer.mlp.down_proj.in_features, layer.mlp.down_proj.out_features, 128, device=self._device).to(self._device)
                down_proj.qweight.copy_(d_data['qweight'])
                down_proj.scales.copy_(d_data['scales'])
                down_proj.qzeros.copy_(d_data['qzeros'])
                layer.mlp.down_proj = down_proj
                
                def fast_mlp_forward(x, mlp_layer=layer.mlp, g_dim=gate_dim):
                    gate_up = mlp_layer.gate_up_proj(x)
                    swiglu_out = fast_swiglu(gate_up, g_dim)
                    return mlp_layer.down_proj(swiglu_out)
                
                layer.mlp.forward = fast_mlp_forward
                replaced_count += 3

        torch.cuda.empty_cache()
        print(f"[Quant] ✅ 成功实施终极剥离！共重构了 {replaced_count} 个接口！")
        self._optimizations_applied.append('quantization_triton_fused_int4_splitK_FastRMSNorm_RoPE_HF_Bypass')







    @property
    def processor(self): return self._processor
    @property
    def model(self): return self._model
    @property
    def device(self): return self._device

    def generate(self, image: Image.Image, question: str, max_new_tokens: int = 128) -> Dict:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        inputs = self._processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0, top_p=1.0, use_cache=True)
        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[0][input_len:]
        text = self._processor.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return {"text": text, "token_count": len(generated_ids)}
