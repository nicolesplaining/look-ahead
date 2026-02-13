from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Your prompt
prompt = "A rhyming couplet:\nA riptide pulled him off the sand,\n"

# Tokenize and generate 
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=16, num_return_sequences=10, do_sample=True)

# Decode and print all samples
for i, output in enumerate(outputs):
    print(f"\n--- Sample {i+1} ---")
    print(tokenizer.decode(output, skip_special_tokens=True))
