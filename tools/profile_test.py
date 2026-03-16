import torch
from PIL import Image
from torch.profiler import profile, record_function, ProfilerActivity
from evaluation_wrapper import VLMModel

def run_profiling():
    print("1. 加载模型...")
    # 这里会自动调用你在 evaluation_wrapper.py 里写的替换逻辑
    model_wrapper = VLMModel(model_path="./Qwen3-VL-2B-Instruct", device="cuda:0")
    
    # 创建一张普通的测试图
    test_image = Image.new('RGB', (800, 800), color='white')
    question = "详细描述这张图片。"
    
    print("2. GPU 预热 (必须做，否则测出来的全是启动开销)...")
    model_wrapper.generate(test_image, question, max_new_tokens=2)
    torch.cuda.synchronize()
    
    print("3. 开始底层算子抓包 Profiling...")
    # 启动 Profiler，监控 CPU 和 CUDA (GPU)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=False
    ) as prof:
        with record_function("VLM_Prefill_and_Decode"):
            # 只生成 1 个 Token，精准测试 TTFT 阶段的算子
            model_wrapper.generate(test_image, question, max_new_tokens=1)
            
    print("\n" + "="*60)
    print("🔥 GPU 耗时排名前 15 的底层算子 (Kernel) 🔥")
    print("="*60)
    # 按 CUDA 耗时排序，打印前 15 名
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

if __name__ == "__main__":
    run_profiling()
