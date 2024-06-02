import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model from the saved directory
model_path = "./results"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = model.to(device)


# Function to generate a response
def generate_response(input_text, max_length=100):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


# Example usage
input_text = "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant"
response = generate_response(input_text)
print(response)
