"""
AICAS 2026 - Participant Core Modification File

Participants should modify the VLMModel class to implement optimizations.

Note:
- Benchmark directly calls self.model.generate() for performance testing.
- Your optimizations should modify self.model or its operators in __init__ via Monkey Patch.
- The generate() method is optional and mainly for debugging.
"""
from typing import Dict
try:
    from PIL import Image
except ImportError:
    # For testing without PIL
    class Image:
        pass
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


class VLMModel:
    """
    Participant optimization class - modify this to implement optimizations.
    
    Optimization Architecture:
    - Split optimizations into separate methods for isolation and testing
    - Enable/disable each optimization independently in __init__
    - Each optimization method can be tested individually
    
    Important Notes:
    1. Benchmark directly calls self.model.generate() for performance testing.
    2. Your optimizations should modify self.model or its operators via Monkey Patch.
    3. All optimizations are applied in __init__ by calling optimization methods.
    """
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
        """
        Initialize model and apply optimizations.
        
        Args:
            model_path: Qwen3-VL-2B-Instruct model path
            device: CUDA device, e.g., "cuda:0"
        """
        self._device = device
        self.model_path = model_path
        
        # Load processor
        print(f"[VLMModel] Loading processor from {model_path}...")
        self._processor = AutoProcessor.from_pretrained(model_path)
        
        # Load model
        print(f"[VLMModel] Loading model with FP16...")
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device
        )
        # print(self._model)
        # import sys
        # sys.exit(0) # 打印完直接退出，不跑测试了
        self._model.eval()
        
        # Track applied optimizations
        self._optimizations_applied = []
        
        
        # ================================================================
        # Participant Optimization Area - Enable/disable optimizations here
        # Uncomment the optimization methods you want to apply
        # ================================================================
        
        # 1. Vision Encoder Acceleration
        # self._optimize_vision_encoder()
        
        # 2. KV Cache Management
        self._optimize_kv_cache()
        
        # 3. Cross-modal Connector Optimization
        # self._optimize_cross_modal_connector()
        
        # 4. Flash Attention Optimization
        self._enable_flash_attention()
        
        # 5. Quantization
        # self._apply_quantization()
        
        # Optional: Explore model structure before optimization
        # self._explore_model_structure()        
        # ================================================================
        
        print(f"[VLMModel] Model loaded successfully on {device}")
        if self._optimizations_applied:
            print(f"[VLMModel] Applied optimizations: {', '.join(self._optimizations_applied)}")
    
    # ================================================================
    # Optimization Methods - Implement your optimizations here
    # ================================================================
    
    def _explore_model_structure(self):
        """
        Helper method to explore model structure.
        
        Use this to understand the model architecture before implementing optimizations.
        This helps identify where to apply monkey patches.
        """
        print("=" * 60)
        print("Model Structure Exploration")
        print("=" * 60)
        
        # Explore vision model structure
        if hasattr(self._model, 'vision_model'):
            print(f"Vision Model: {type(self._model.vision_model)}")
            if hasattr(self._model.vision_model, 'encoder'):
                if hasattr(self._model.vision_model.encoder, 'layers'):
                    print(f"  Vision Encoder Layers: {len(self._model.vision_model.encoder.layers)}")
                    # Show first layer structure
                    if len(self._model.vision_model.encoder.layers) > 0:
                        print(f"  First Layer Type: {type(self._model.vision_model.encoder.layers[0])}")
        else:
            print("Vision Model: Not found (model structure may differ)")
        
        # Explore language model structure
        if hasattr(self._model, 'model'):
            print(f"Language Model: {type(self._model.model)}")
            if hasattr(self._model.model, 'layers'):
                print(f"  Language Model Layers: {len(self._model.model.layers)}")
        else:
            print("Language Model: Not found (model structure may differ)")
        
        # Explore cross-modal components
        cross_modal_attrs = ['connector', 'cross_attn', 'cross_attention', 'proj', 'projector']
        found_components = []
        for attr in cross_modal_attrs:
            if hasattr(self._model, attr):
                found_components.append(attr)
        if found_components:
            print(f"Cross-modal Components: {', '.join(found_components)}")
        else:
            print("Cross-modal Components: Explore manually (structure may vary)")
        
        print("=" * 60)
        print("Tip: Use print(self._model) to see full model structure")
        print("=" * 60)
    
    def _optimize_vision_encoder(self):
        """
        Optimize Vision Encoder for high-resolution image inputs.
        
        Optimization Directions:
        1. Patch embedding convolution optimization
        2. Vision Transformer attention mechanism optimization
        3. Layer normalization optimization
        4. Memory-efficient image processing
        
        Implementation Steps:
        1. Inspect model structure: call self._explore_model_structure()
        2. Identify bottlenecks using profiling tools (PyTorch Profiler, nsys, etc.)
        3. Implement optimized operators (Triton/CUDA kernels)
        4. Replace original operators via monkey patch
        
        Target Components:
        - self._model.vision_model (if exists)
        - Vision encoder layers and attention mechanisms
        - Convolution operations in patch embedding
        """
        # TODO: Implement your Vision Encoder optimization here
        # 
        # Example workflow:
        # 1. from your_optimization import optimized_attention, optimized_conv
        # 2. Inspect: print(self._model.vision_model) to find target layers
        # 3. Replace: layer.self_attn.forward = optimized_attention
        # 4. Test: Run benchmark to verify improvement
        
        if 'vision_encoder' not in self._optimizations_applied:
            self._optimizations_applied.append('vision_encoder')
    
    
    
    
    def _optimize_kv_cache(self):
        """
        战役 1：静态 KV Cache + 消灭 aten::copy_
        
        使用 StaticCache 替代默认的 DynamicCache，预分配固定大小的内存区域，
        避免动态扩容导致的 aten::copy_ 操作和内存碎片。
        """
        from transformers.cache_utils import StaticCache
        import types
        
        print("[VLMModel] Applying Static KV Cache optimization...")
        
        # 1. 确保启用 use_cache
        self._model.config.use_cache = True
        if hasattr(self._model.config, 'pad_token_id'):
            if self._model.config.pad_token_id is None:
                self._model.config.pad_token_id = self._model.config.eos_token_id
        
        # 2. 计算合理的最大缓存长度（输入 + 生成）
        # 根据 benchmark.py，MAX_NEW_TOKENS = 128，输入通常 < 2048
        max_cache_len = 4096  # 留出充足余量
        
        # 3. Monkey Patch 模型的 generate 方法，自动注入 StaticCache
        original_generate = self._model.generate
        
        def static_cache_generate(self, *args, **kwargs):
            # 如果用户没有提供 past_key_values，则使用 StaticCache
            if "past_key_values" not in kwargs or kwargs["past_key_values"] is None:
                static_cache = StaticCache(
                    config=self.config,
                    max_cache_len=max_cache_len,
                    device=self.device if hasattr(self, 'device') else "cuda:0"
                )
                kwargs["past_key_values"] = static_cache
            return original_generate(*args, **kwargs)
        
        # 绑定到模型
        self._model.generate = types.MethodType(static_cache_generate, self._model)
        
        if 'static_kv_cache' not in self._optimizations_applied:
            self._optimizations_applied.append('static_kv_cache')
    
    def _optimize_cross_modal_connector(self):
        """
        Optimize Cross-modal Connector computation efficiency.
        
        Optimization Directions:
        1. Cross-attention mechanism optimization
        2. Vision-to-language projection optimization
        3. Multi-modal fusion layer efficiency
        4. Feature alignment and transformation optimization
        
        Implementation Steps:
        1. Identify cross-modal components using self._explore_model_structure()
        2. Profile cross-modal operations to find bottlenecks
        3. Implement optimized cross-attention or projection kernels
        4. Replace original operations via monkey patch
        
        Note: Qwen3-VL's cross-modal structure may vary.
        Use model exploration to identify actual component names and locations.
        """
        # TODO: Implement your Cross-modal Connector optimization here
        # 
        # Example workflow:
        # 1. Explore: self._explore_model_structure() to find connector components
        # 2. from your_optimization import optimized_cross_attention
        # 3. Identify: Inspect model to find cross-attention layers
        # 4. Replace: connector.cross_attention.forward = optimized_cross_attention
        # 5. Test: Verify accuracy and performance improvements
        
        if 'cross_modal' not in self._optimizations_applied:
            self._optimizations_applied.append('cross_modal')
            
            
    def _enable_flash_attention(self):
        """
        手写算子替换：踢开 HuggingFace 包装器，直调 Flash Attention 2 底层 C++ 接口
        专注优化 Vision Encoder（拉低 TTFT 的核心）
        """
        import torch
        import types
        from flash_attn import flash_attn_varlen_func
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision
        
        print("[VLMModel] Applying Hardcore Flash Attention to Vision Encoder...")

        def custom_vision_flash_attn_forward(
            self_attn, # 原本的 self
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb = None,
            position_embeddings = None,
            **kwargs,
        ) -> torch.Tensor:
            seq_length = hidden_states.shape[0]
            
            # 1. 干净利落地切分 Q, K, V
            # 出来的形状完美契合 FA2: (seq_length, num_heads, head_dim)
            query_states, key_states, value_states = (
                self_attn.qkv(hidden_states)
                .reshape(seq_length, 3, self_attn.num_heads, -1)
                .permute(1, 0, 2, 3)
                .unbind(0)
            )
            
            # 2. 注入旋转位置编码 (RoPE)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
            
            # ==========================================================
            # 3. 核心魔改：砍掉多余的 unsqueeze，直调底层 Kernel！
            # ==========================================================
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            
            # Flash Attention varlen 接口强制要求 cu_seqlens 必须是 int32 类型
            cu_seqlens_int32 = cu_seqlens.to(torch.int32)
            
            # 毫无中间商赚差价，直接把数据怼进 GPU 的 Tensor Core！
            attn_output = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_int32,
                cu_seqlens_k=cu_seqlens_int32,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                softmax_scale=self_attn.scaling,
                causal=False, # 视觉模块是全局注意力，不是因果注意力
            )
            
            # 4. 最终投影输出
            attn_output = attn_output.view(seq_length, -1)
            attn_output = self_attn.proj(attn_output)
            
            return attn_output

        # 暴力替换！把 Qwen 视觉模块里 24 层 Transformer 的注意力逻辑，全换成我们写的！
        if hasattr(self._model.model, 'visual'):
            for block in self._model.model.visual.blocks:
                block.attn.forward = types.MethodType(custom_vision_flash_attn_forward, block.attn)
            self._optimizations_applied.append('hardcore_vision_flash_attn')
        else:
            print("[Warning] Could not find visual model to patch Flash Attention.")

    
    # def _enable_flash_attention(self):
    #     """
    #     Enable or implement Flash Attention optimization.
        
    #     Implementation Approaches:
        
    #     Approach 1: Enable PyTorch's Built-in Flash Attention (Simple)
    #         - Uses torch.backends.cuda.enable_flash_sdp(True)
    #         - Easy to enable but limited customization
    #         - May not work for all attention patterns in Qwen3-VL
        
    #     Approach 2: Implement Custom Flash Attention (Advanced, Recommended)
    #         - Write custom Triton/CUDA kernels for attention computation
    #         - Replace torch.nn.functional.scaled_dot_product_attention
    #         - Full control over attention computation and memory layout
    #         - Better performance potential but requires more implementation effort
        
    #     Recommended: Implement Approach 2 for better performance gains.
    #     Use profiling to identify which attention operations benefit most from optimization.
    #     """
    #     # TODO: Choose and implement your Flash Attention approach
        
    #     # Approach 1: Simple (enable PyTorch built-in)
    #     # torch.backends.cuda.enable_flash_sdp(True)
        
    #     # Approach 2: Advanced (custom implementation - recommended)
    #     # from your_optimization import custom_flash_attention
    #     # torch.nn.functional.scaled_dot_product_attention = custom_flash_attention
    #     # 
    #     # Or replace at layer level:
    #     # for layer in self._model.model.layers:
    #     #     layer.self_attn.forward = custom_attention_with_flash
        
    #     torch.backends.cuda.enable_flash_sdp(True)
    #     self._optimizations_applied.append('flash_attention')

        
    #     # if 'flash_attention' not in self._optimizations_applied:
    #     #     self._optimizations_applied.append('flash_attention')
    
    def _apply_quantization(self):
        """
        Apply quantization to reduce model size and speed up inference.
        
        Optimization Directions:
        1. INT8 quantization (8-bit integer)
        2. FP8 quantization (8-bit floating point)
        3. Mixed precision quantization
        4. Dynamic vs static quantization
        
        Implementation Steps:
        1. Choose quantization strategy based on accuracy/performance trade-off
        2. Use quantization libraries (BitsAndBytes, TensorRT, etc.)
        3. Calibrate quantized model on validation data
        4. Verify accuracy preservation
        
        Note: Quantization may require reloading the model with quantization config.
        Consider applying quantization before other optimizations if model reload is needed.
        """
        # TODO: Implement your quantization here
        # 
        # Example workflow:
        # 1. from transformers import BitsAndBytesConfig
        # 2. quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        # 3. Note: May need to reload model with quantization config
        # 4. Test: Verify accuracy and performance improvements
        
        if 'quantization' not in self._optimizations_applied:
            self._optimizations_applied.append('quantization')
    
    # Required properties for benchmark
    @property
    def processor(self):
        """
        Required by benchmark for input processing.
        
        Benchmark uses this to prepare inputs with unified tokenizer.
        """
        return self._processor
    
    @property
    def model(self):
        """
        Required by benchmark for direct model.generate() calls.
        
        Benchmark directly calls self.model.generate() for performance testing.
        Your optimizations should modify this model object or its operators.
        """
        return self._model
    
    @property
    def device(self):
        """
        Required by benchmark for device information.
        """
        return self._device
    
    def generate(
        self, 
        image: Image.Image, 
        question: str, 
        max_new_tokens: int = 128
    ) -> Dict:
        """
        Generate answer (optional method, mainly for debugging).
        
        Note: Benchmark uses self.model.generate() directly for performance testing.
        This method is provided for convenience and debugging purposes.
        
        Args:
            image: PIL Image object
            question: Question text
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Dict: {
                "text": str,        # Generated text answer
                "token_count": int  # Generated token count
            }
        """
        # Build Qwen3-VL message format
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }]
        
        # Process inputs
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self._device)
        
        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                use_cache=True
            )
        
        # Extract generated tokens (remove input part)
        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[0][input_len:]
        
        # Decode
        text = self._processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return {
            "text": text,
            "token_count": len(generated_ids)
        }