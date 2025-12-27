import torch
from transformers import pipeline

# 1. Setup the model ID
model_id = "meta-llama/Llama-3.2-1B"

# 2. Initialize the text-generation pipeline
# We use bfloat16 for better efficiency on modern GPUs
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 3. Define a simple prompt
# Note: Since this is the base model (not 'Instruct'), 
# it will try to "complete" your text rather than "answer" a question.
prompt = "The future of artificial intelligence is"

# 4. Generate text
print("\n--- Generating Response ---\n")
output = pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.7)

# 5. Print the result
print(output[0]['generated_text'])