import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

def probe_model_internals():
    print("="*60)
    print("🚀 Qwen3-VL 底层探针启动...")
    print("="*60)

    # 1. 加载模型
    model_path = "./Qwen3-VL-2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="cuda:0"
    )

    # 2. 构造一个极其简单的测试样例
    # 弄一张极小的图（比如 28x28），这样 token 少，打印出来不会刷屏
    test_image = Image.new('RGB', (28, 28), color='red')
    question = "图片里是什么？"
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": test_image},
            {"type": "text", "text": question}
        ]
    }]

    # 3. 探针 A：Token 边界探测
    print("\n" + "="*60)
    print("🔍 探针 A：Token 结构与边界探测")
    print("="*60)
    
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to("cuda:0")

    input_ids = inputs.input_ids[0]
    total_len = len(input_ids)
    print(f"输入总 Token 长度: {total_len}")
    
    # 我们把 Token 逐个打印出来，寻找规律
    print("\nToken 序列大解剖 (展示前 20 个 和 后 20 个):")
    for i in range(min(20, total_len)):
        token_str = processor.tokenizer.decode([input_ids[i]])
        print(f"Index {i:4d} | ID {input_ids[i]:6d} | Token: {repr(token_str)}")
        
    print("...... (中间省略) ......")
    
    for i in range(max(20, total_len - 20), total_len):
        token_str = processor.tokenizer.decode([input_ids[i]])
        print(f"Index {i:4d} | ID {input_ids[i]:6d} | Token: {repr(token_str)}")

    # 寻找特殊视觉 Token 的 ID (比如 <|image_pad|> 或 <|vision_end|>)
    # 看看文本 "图片里是什么？" 的第一个字出现在哪个 Index。这决定了我们要缓存多长！

    # 4. 探针 B：KV Cache 形状与对象探测
    print("\n" + "="*60)
    print("🔍 探针 B：KV Cache 结构与维度探测")
    print("="*60)
    
    # 跑一次完整的 Prefill 前向传播（不开梯度）
    with torch.no_grad():
        outputs = model(
            **inputs,
            use_cache=True, 
            output_hidden_states=False
        )

    kv_cache = outputs.past_key_values
    print(f"KV Cache 的 Python 类型: {type(kv_cache)}")
    
    # 判断格式并打印维度
    if isinstance(kv_cache, tuple):
        print(f"层数: {len(kv_cache)} 层")
        print(f"第一层 K 矩阵的类型: {type(kv_cache[0][0])}")
        print(f"第一层 K 矩阵的维度: {kv_cache[0][0].shape}  <--- 重点看这个！")
        print(f"第一层 V 矩阵的维度: {kv_cache[0][1].shape}")
    else:
        # 可能是 transformers 4.36+ 引入的 Cache/DynamicCache 对象
        print(f"当前使用的是新版 Cache 对象。")
        if hasattr(kv_cache, 'key_cache'):
            print(f"第一层 K 矩阵的维度: {kv_cache.key_cache[0].shape}  <--- 重点看这个！")
            
    # 5. 显存测试
    kv_size_mb = 0
    if isinstance(kv_cache, tuple):
        for layer in kv_cache:
            kv_size_mb += layer[0].element_size() * layer[0].nelement() / (1024*1024)
            kv_size_mb += layer[1].element_size() * layer[1].nelement() / (1024*1024)
    print(f"\n⚠️ 警告：当前这张极小图片的 KV Cache 占用了 {kv_size_mb:.2f} MB 显存！")
    print("如果是一张 800x800 的原图，这个体积会暴涨数十倍。")

if __name__ == "__main__":
    probe_model_internals()
