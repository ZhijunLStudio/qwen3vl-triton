import torch
from PIL import Image
from torch.profiler import profile, record_function, ProfilerActivity
from evaluation_wrapper import VLMModel

def run_decode_profiling():
    print("="*60)
    print("🚀 启动专门针对 Decode 阶段 (Throughput) 的高精度 Profiler...")
    print("="*60)

    # 1. 加载我们带有 INT4 Triton 和 KV Cache 优化的模型
    model_wrapper = VLMModel(model_path="./Qwen3-VL-2B-Instruct", device="cuda:0")
    processor = model_wrapper.processor
    model = model_wrapper.model
    device = model_wrapper.device

    # 2. 准备一张测试图和问题
    test_image = Image.new('RGB', (224, 224), color='white')
    question = "描述图片。"
    messages = [{"role": "user", "content": [{"type": "image", "image": test_image}, {"type": "text", "text": question}]}]
    
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(device)

    print("\n[阶段 1] 执行 Prefill (跳过这部分的 Profiling)...")
    with torch.no_grad():
        outputs = model(
            **inputs,
            use_cache=True,
            output_hidden_states=False
        )
    
    # 拿到 Prefill 产生的 KV Cache 和下一个要输入的 Token
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    
    print("\n[阶段 2] GPU 预热 Decode...")
    # 预热几轮，排除启动开销
    with torch.no_grad():
        for _ in range(3):
            out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            next_token = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            past_key_values = out.past_key_values

    print("\n[阶段 3] 📸 开启 Profiler，精准抓取 5 次单字生成的耗时！")
    torch.cuda.synchronize()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=True # 开启调用栈追踪，揪出是谁调用的！
    ) as prof:
        with torch.no_grad():
            with record_function("VLM_Pure_Decode_Step"):
                # 连续生成 5 个 Token，放大问题
                for _ in range(5):
                    out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
                    next_token = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                    past_key_values = out.past_key_values

    torch.cuda.synchronize()
    
    print("\n" + "="*60)
    print("🔥 Decode 阶段 GPU 耗时排名前 20 的算子 (Kernel) 🔥")
    print("="*60)
    # 按 CUDA 耗时排序打印
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    print("\n" + "="*60)
    print("🧠 Decode 阶段 CPU 发起调用排名前 20 的算子 (寻找 Launch Overhead 瓶颈) 🧠")
    print("="*60)
    # 按 CPU 耗时排序打印
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

if __name__ == "__main__":
    run_decode_profiling()
