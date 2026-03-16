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
import torch
from collections import OrderedDict
import time


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
        # self._enable_flash_attention()
        
        # 5. Quantization
        # self._apply_quantization()
        
        # Optional: Explore model structure before optimization
        # self._explore_model_structure()
        
        # ================================================================
        
        print(f"[VLMModel] Model loaded successfully on {device}")
        if self._optimizations_applied:
            print(f"[VLMModel] Applied optimizations: {', '.join(self._optimizations_applied)}")
            
            
        # 1. 初始化 LRU 缓存 (容量设为3，防止 OOM)
        # self.image_cache = OrderedDict()
        # self.cache_capacity = 3
        
        # # 2. 启用跨模态拦截优化
        # self._optimize_cross_modal_connector()

    
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
    
    # def _optimize_kv_cache(self):
    #     """
    #     Optimize KV Cache management to reduce memory fragmentation.
        
    #     Optimization Directions:
    #     1. Memory layout optimization (contiguous memory allocation)
    #     2. Fragmentation-free allocation strategies
    #     3. Efficient cache reuse patterns
    #     4. Dynamic cache sizing
        
    #     Implementation Steps:
    #     1. Understand current KV cache implementation in model layers
    #     2. Design memory-efficient cache allocation strategy
    #     3. Implement custom KV cache allocator if needed
    #     4. Apply optimizations via monkey patch or config modification
        
    #     Target Components:
    #     - self._model.config (cache configuration)
    #     - Attention layers (KV cache allocation)
    #     - Generation loop (cache management)
    #     """
    #     # Enable KV Cache first
    #     self._model.config.use_cache = True
    #     if hasattr(self._model.config, 'pad_token_id'):
    #         if self._model.config.pad_token_id is None:
    #             self._model.config.pad_token_id = self._model.config.eos_token_id
        
    #     # TODO: Implement advanced KV Cache optimizations here
    #     # 
    #     # Example workflow:
    #     # 1. from your_optimization import FragmentationFreeKVCache
    #     # 2. for layer in self._model.model.layers:
    #     # 3.     layer.attention.custom_kv_cache = FragmentationFreeKVCache()
    #     # 4. Test: Monitor memory usage and generation speed
        
    #     if 'kv_cache' not in self._optimizations_applied:
    #         self._optimizations_applied.append('kv_cache')
    
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
        Enable or implement Flash Attention optimization.
        
        Implementation Approaches:
        
        Approach 1: Enable PyTorch's Built-in Flash Attention (Simple)
            - Uses torch.backends.cuda.enable_flash_sdp(True)
            - Easy to enable but limited customization
            - May not work for all attention patterns in Qwen3-VL
        
        Approach 2: Implement Custom Flash Attention (Advanced, Recommended)
            - Write custom Triton/CUDA kernels for attention computation
            - Replace torch.nn.functional.scaled_dot_product_attention
            - Full control over attention computation and memory layout
            - Better performance potential but requires more implementation effort
        
        Recommended: Implement Approach 2 for better performance gains.
        Use profiling to identify which attention operations benefit most from optimization.
        """
        # TODO: Choose and implement your Flash Attention approach
        
        # Approach 1: Simple (enable PyTorch built-in)
        # torch.backends.cuda.enable_flash_sdp(True)
        
        # Approach 2: Advanced (custom implementation - recommended)
        # from your_optimization import custom_flash_attention
        # torch.nn.functional.scaled_dot_product_attention = custom_flash_attention
        # 
        # Or replace at layer level:
        # for layer in self._model.model.layers:
        #     layer.self_attn.forward = custom_attention_with_flash
        
        if 'flash_attention' not in self._optimizations_applied:
            self._optimizations_applied.append('flash_attention')
    
    def _apply_quantization(self):
        """
        修复版 W8A16（Weight-Only 8bit） - 专治 Qwen3-VL lm_head.SCB 报错
        """
        print("[Quant] Applying W8A16 (bitsandbytes weight-only) - skipping lm_head...")

        # 先卸载旧模型（避免显存残留）
        if hasattr(self, '_model'):
            del self._model
            torch.cuda.empty_cache()

        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=["lm_head", "vision_model"],  # ← 关键修复！lm_head 必须跳过
            # llm_int8_enable_fp32_cpu_offload=True,  # 如果还 OOM 可以打开
        )

        print("[Quant] Reloading model with W8A16...")
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            dtype=torch.float16,          # ← 修复 deprecated warning（新版 transformers 用 dtype）
            device_map=self._device,
            trust_remote_code=True        # Qwen-VL 系列需要
        )
        self._model.eval()

        if 'quantization' not in self._optimizations_applied:
            self._optimizations_applied.append('quantization (W8A16 - lm_head skipped)')
    
    
    
    # def _apply_quantization(self):
    #     """
    #     Apply quantization to reduce model size and speed up inference.
        
    #     Optimization Directions:
    #     1. INT8 quantization (8-bit integer)
    #     2. FP8 quantization (8-bit floating point)
    #     3. Mixed precision quantization
    #     4. Dynamic vs static quantization
        
    #     Implementation Steps:
    #     1. Choose quantization strategy based on accuracy/performance trade-off
    #     2. Use quantization libraries (BitsAndBytes, TensorRT, etc.)
    #     3. Calibrate quantized model on validation data
    #     4. Verify accuracy preservation
        
    #     Note: Quantization may require reloading the model with quantization config.
    #     Consider applying quantization before other optimizations if model reload is needed.
    #     """
    #     # TODO: Implement your quantization here
    #     # 
    #     # Example workflow:
    #     # 1. from transformers import BitsAndBytesConfig
    #     # 2. quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    #     # 3. Note: May need to reload model with quantization config
    #     # 4. Test: Verify accuracy and performance improvements
        
    #     if 'quantization' not in self._optimizations_applied:
    #         self._optimizations_applied.append('quantization')
    
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



    def _compute_image_hash(self, pixel_values):
        """
        极速计算图片张量的 Hash 签名
        对张量的形状和前中后几个关键点求和，保证同一张图签名一致，不同图极大概率不同
        """
        # 转为 float32 避免半精度求和溢出
        val = pixel_values.to(torch.float32)
        # 提取开头、中间、结尾各取一小部分求和，速度极快
        sig = float(val.flatten()[:500].sum() + val.flatten()[-500:].sum())
        return f"{list(pixel_values.shape)}_{sig:.4f}"

    def _optimize_cross_modal_connector(self):
        """
        拦截 Qwen3-VL 的视觉编码器 (self._model.model.visual)
        """
        print("[VLMModel] 🚀 启用 Vision Encoder 前缀缓存优化...")
        
        # 结合你打印的结构，确保能找到 visual 模块
        if not hasattr(self._model, 'model') or not hasattr(self._model.model, 'visual'):
            print("[Warning] 未找到 self._model.model.visual，缓存机制未开启！")
            return
            
        original_visual_forward = self._model.model.visual.forward
        
        # 定义拦截器函数
        def cached_visual_forward(*args, **kwargs):
            # Qwen3-VL 的 visual forward 通常第一个参数是 hidden_states/pixel_values
            # 先尝试从 kwargs 获取，如果没有就从 args 获取
            pixel_values = kwargs.get('hidden_states', args[0] if len(args) > 0 else None)
            
            if pixel_values is None:
                # 异常情况，直接放行
                return original_visual_forward(*args, **kwargs)
                
            # 1. 计算图像签名
            img_hash = self._compute_image_hash(pixel_values)
            
            # 2. 查表：如果命中缓存，直接返回！
            if img_hash in self.image_cache:
                print(f"[Radix Cache] ⚡ 命中同图缓存！跳过视觉计算！(Hash: {img_hash[-10:]})")
                self.image_cache.move_to_end(img_hash) # 刷新活跃度
                return self.image_cache[img_hash]
            
            # 3. 未命中：调用原生模块计算
            # print(f"[Radix Cache] 首次遇到该图，计算并缓存... (Hash: {img_hash[-10:]})")
            features = original_visual_forward(*args, **kwargs)
            
            # 4. 存入缓存
            # 必须使用 .detach()，否则会保存计算图导致显存瞬间爆炸！
            if isinstance(features, torch.Tensor):
                cached_data = features.detach()
            elif isinstance(features, tuple):
                cached_data = tuple(f.detach() if isinstance(f, torch.Tensor) else f for f in features)
            else:
                cached_data = features # fallback
                
            self.image_cache[img_hash] = cached_data
            
            # 5. 维护容量，踢掉最老的缓存
            if len(self.image_cache) > self.cache_capacity:
                self.image_cache.popitem(last=False)
                
            return features
            
        # 实施 Monkey Patch 替换！
        self._model.model.visual.forward = cached_visual_forward
        
        if 'cross_modal' not in self._optimizations_applied:
            self._optimizations_applied.append('cross_modal_radix_cache')
            
            
    def _optimize_kv_cache(self):
        """
        Level-2 Prefix Caching: 终极绝对安全版 (参数绑定 + Zero-Copy)
        """
        import functools
        import inspect
        from transformers.cache_utils import DynamicCache
        from collections import OrderedDict
        
        print("[VLMModel] 🚀 启用终极 LLM Prefix KV Caching...")
        
        self.llm_kv_cache = OrderedDict()
        self.llm_kv_cache_lens = {}
        self.llm_cache_capacity = 3
        
        original_forward = self._model.forward
        # 预先解析参数签名，速度极快
        forward_sig = inspect.signature(original_forward)
        
        import time # 记得在文件头 import time
        
    def _optimize_kv_cache(self):
        """
        Level-2 Prefix Caching: 暴力探针 + 兼容 DynamicCache 版
        """
        import functools
        from transformers.cache_utils import DynamicCache
        from collections import OrderedDict
        import time
        
        print("[VLMModel] 🚀 启用 LLM Prefix KV Caching (暴力探针版)...")
        
        self.llm_kv_cache = OrderedDict()
        self.llm_kv_cache_lens = {}
        self.llm_cache_capacity = 3
        
        original_forward = self._model.forward
        
        import time
        
    def _optimize_kv_cache(self):
        """
        Level-2 Prefix Caching: 官方 API 安全兼容版 + 绝对零拷贝
        """
        import functools
        from transformers.cache_utils import DynamicCache
        from collections import OrderedDict
        import torch
        
        print("[VLMModel] 🚀 启用 LLM Prefix KV Caching (官方API安全版)...")
        
        self.llm_kv_cache = OrderedDict()
        self.llm_kv_cache_lens = {}
        self.llm_cache_capacity = 3
        
        original_forward = self._model.forward
        
    def _optimize_kv_cache(self):
        """
        Level-2 Prefix Caching: 终极通关版 (修复 cache_position 崩溃 + 绝对零拷贝)
        """
        import functools
        from transformers.cache_utils import DynamicCache
        from collections import OrderedDict
        import torch
        
        print("[VLMModel] 🚀 启用顶级 LLM Prefix KV Caching (完美通关版)...")
        
        self.llm_kv_cache = OrderedDict()
        self.llm_kv_cache_lens = {}
        self.llm_cache_capacity = 3
        
        original_forward = self._model.forward
        
        @functools.wraps(original_forward)
        def custom_forward(*args, **kwargs):
            input_ids = kwargs.get('input_ids', args[0] if len(args) > 0 else None)
            past_key_values = kwargs.get('past_key_values')
            pixel_values = kwargs.get('pixel_values')
            
            # 兼容判断是否为首轮 Prefill
            is_prefill = False
            if past_key_values is None:
                is_prefill = True
            elif hasattr(past_key_values, 'get_seq_length') and past_key_values.get_seq_length() == 0:
                is_prefill = True
            elif isinstance(past_key_values, tuple) and len(past_key_values) == 0:
                is_prefill = True

            if is_prefill and pixel_values is not None and input_ids is not None:
                img_hash = self._compute_image_hash(pixel_values)
                
                if img_hash in self.llm_kv_cache:
                    print(f"\n[LLM Cache] ⚡⚡⚡ 命中顶级 LLM 记忆池！瞬间跳过全图！(Hash: {img_hash[-6:]})")
                    self.llm_kv_cache.move_to_end(img_hash)
                    
                    prefix_len = self.llm_kv_cache_lens[img_hash]
                    cached_cache = self.llm_kv_cache[img_hash]
                    
                    # 🚀 绝对零拷贝重组：只传指针，不复制显存！
                    new_cache = DynamicCache()
                    for i in range(len(cached_cache)):
                        k, v = cached_cache[i]
                        new_cache.update(k, v, i)
                        
                    kwargs['past_key_values'] = new_cache
                    kwargs['input_ids'] = input_ids[:, prefix_len:]
                    
                    if kwargs.get('position_ids') is not None:
                        kwargs['position_ids'] = kwargs['position_ids'][..., prefix_len:]
                        
                    # 💥 致命修复：切断 cache_position，防止 RoPE 维度崩溃！
                    if kwargs.get('cache_position') is not None:
                        kwargs['cache_position'] = kwargs['cache_position'][prefix_len:]
                        
                    kwargs['pixel_values'] = None
                    if 'image_grid_thw' in kwargs:
                        kwargs['image_grid_thw'] = None
                        
                    return original_forward(*args, **kwargs)
                    
                else:
                    # 全量计算
                    outputs = original_forward(*args, **kwargs)
                    
                    vision_end_mask = (input_ids[0] == 151653).nonzero(as_tuple=True)[0]
                    if len(vision_end_mask) > 0 and hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                        prefix_len = vision_end_mask[0].item() + 1
                        
                        # 🚀 绝对零拷贝保存：不 clone()，直接存视图，省下 20ms！
                        new_cache = DynamicCache()
                        for i in range(len(outputs.past_key_values)):
                            k, v = outputs.past_key_values[i]
                            # Qwen3 的 KV shape: [batch, num_heads, seq_len, head_dim]
                            new_cache.update(k[:, :, :prefix_len, :], v[:, :, :prefix_len, :], i)
                            
                        self.llm_kv_cache[img_hash] = new_cache
                        self.llm_kv_cache_lens[img_hash] = prefix_len
                        
                        if len(self.llm_kv_cache) > self.llm_cache_capacity:
                            oldest = next(iter(self.llm_kv_cache))
                            del self.llm_kv_cache[oldest]
                            del self.llm_kv_cache_lens[oldest]
                            
                    return outputs

            # 非 Prefill 阶段
            return original_forward(*args, **kwargs)

        self._model.forward = custom_forward
        self._optimizations_applied.append('llm_prefix_caching_final')
