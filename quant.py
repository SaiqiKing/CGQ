from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch

model_path = '/root/autodl-tmp/qwen/Qwen2-7B'   # Qwen-1_8B  Qwen2-0.5B  Qwen2-1.5B
quant_path = '/root/autodl-tmp/qwen/Qwen2-7B-ooo-4bit'

quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
if ("Qwen1" in str(model_path) or "Qwen2" in str(model_path)):
    # model = AutoAWQForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2", **{"low_cpu_mem_usage": True, "use_cache": False})
    model = AutoAWQForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, **{"low_cpu_mem_usage": True, "use_cache": False})
else:
    # model = AutoAWQForCausalLM.from_pretrained(model_path, bf16=False, fp16=True, use_flash_attn=True, **{"low_cpu_mem_usage": True, "use_cache": False})
    model = AutoAWQForCausalLM.from_pretrained(model_path, bf16=False, fp16=True, use_flash_attn=False, **{"low_cpu_mem_usage": True, "use_cache": False})

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
