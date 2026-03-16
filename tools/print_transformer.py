# import inspect
# from transformers.models.qwen3_vl.modeling_qwen3_vl import (
#     Qwen3VLModel, 
#     Qwen3VLTextAttention, 
#     Qwen3VLTextMLP
# )

# print("============= 1. 核心枢纽：Qwen3VLModel.forward =============")
# print(inspect.getsource(Qwen3VLModel.forward))

# print("\n============= 2. 文本注意力：Qwen3VLTextAttention.forward =============")
# print(inspect.getsource(Qwen3VLTextAttention.forward))

# print("\n============= 3. 文本MLP：Qwen3VLTextMLP.forward =============")
# print(inspect.getsource(Qwen3VLTextMLP.forward))



from transformers import AutoModelForImageTextToText
model = AutoModelForImageTextToText.from_pretrained("./Qwen3-VL-2B-Instruct", torch_dtype="auto")
print(model)
