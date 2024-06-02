# Reload tokenizer and model
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import setup_chat_format
base_model = "mistralai/Mistral-7B-v0.3"
new_model = "Orpo-Mad-Max-Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    offload_buffers=True,
    device_map="cpu",
)
model, tokenizer = setup_chat_format(model, tokenizer)

# Merge adapter with base model
model = PeftModel.from_pretrained(model, "./results/checkpoint-330", offload_buffers=True)
model = model.merge_and_unload()
model.save_pretrained("./results")
tokenizer.save_pretrained("./results")
model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)