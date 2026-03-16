import inspect
from transformers import AutoModelForImageTextToText

print("加载模型中...")
model = AutoModelForImageTextToText.from_pretrained("./Qwen3-VL-2B-Instruct", device_map="cpu")

# 1. 打印原生 forward 的签名
sig_original = inspect.signature(model.forward)
print("\n✅ 原生 forward 的参数签名:")
print(sig_original)

# 2. 模拟我们之前的错误拦截
def bad_custom_forward(*args, **kwargs):
    pass
model.forward = bad_custom_forward
sig_bad = inspect.signature(model.forward)
print("\n❌ 错误拦截后的参数签名 (导致报错的元凶):")
print(sig_bad)

# 3. 正确的拦截方式 (使用 functools.wraps)
import functools
# 重新加载干净的模型
model = AutoModelForImageTextToText.from_pretrained("./Qwen3-VL-2B-Instruct", device_map="cpu")
original_forward = model.forward

@functools.wraps(original_forward)
def good_custom_forward(*args, **kwargs):
    pass
model.forward = good_custom_forward
sig_good = inspect.signature(model.forward)
print("\n🎯 functools.wraps 修复后的参数签名 (完美伪装):")
print(sig_good)
