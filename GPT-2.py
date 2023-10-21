# Import the necessary libraries
import torch
import transformers

# Load the pre-trained GPT-2 model and tokenizer
model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")

# Define a function to generate text given a prompt
def generate_text(prompt, max_length=100, temperature=0.9, top_k=50, top_p=0.95):
  # Encode the prompt into tokens
  input_ids = tokenizer.encode(prompt, return_tensors="pt")
  # Generate text using the model
  output = model.generate(input_ids, max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p)
  # Decode the output into text
  text = tokenizer.decode(output[0], skip_special_tokens=True)
  # Return the text
  return text

# Test the function with a sample prompt
prompt = "AI is"
text = generate_text(prompt)
print(text)
