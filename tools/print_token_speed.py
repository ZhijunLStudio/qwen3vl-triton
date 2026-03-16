import time
import torch
from evaluation_wrapper import VLMModel
from PIL import Image

def test_kv_cache_slowdown():
    model_wrapper = VLMModel(model_path="./Qwen3-VL-2B-Instruct", device="cuda:0")
    test_image = Image.new('RGB', (224, 224), color='white')
    question = "请写一篇500字的长文描述这张图片。" # 诱导模型生成长文本
    
    # 提取底层模型和输入
    processor = model_wrapper.processor
    model = model_wrapper.model
    inputs = processor.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": test_image}, {"type": "text", "text": question}]}], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda:0")
    
    # 手动写一个简易的 Generation Loop，用来掐表
    past_key_values = None
    input_ids = inputs.input_ids
    
    print("开始逐个 Token 生成测试...")
    for i in range(100): # 生成100个token看趋势
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values, # 传入上一轮的 KV Cache
                use_cache=True
            )
        
        torch.cuda.synchronize()
        step_time = (time.perf_counter() - start_time) * 1000 # 转成毫秒
        
        # 拿到生成的 token 和新的 KV Cache
        next_token = outputs.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        input_ids = next_token
        past_key_values = outputs.past_key_values
        
        # 每 10 个 Token 打印一次，观察时间是否呈线性增长
        if (i+1) % 10 == 0:
            # 顺便监控一下显存占用
            mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"Token {i+1:3d} | 耗时: {step_time:.2f} ms | 显存占用: {mem_mb:.2f} MB")

if __name__ == "__main__":
    test_kv_cache_slowdown()
