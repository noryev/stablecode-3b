import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os

# Fetch the Hugging Face token from the environment
hf_token = os.getenv("HF_TOKEN")

# Load the tokenizer and model for stablecode with the token

# Look to see how others do this- Are they pulling the model from Huggingface like this or are they doing it another way. 
# If anything replicate what is already being done.

tokenizer = AutoTokenizer.from_pretrained(
    "stabilityai/stablecode-instruct-alpha-3b", 
    trust_remote_code=True, 
    use_auth_token=hf_token  # Use the token for authentication
)
model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/stablecode-instruct-alpha-3b",
    trust_remote_code=True,
    torch_dtype="auto",  # Let the model manage tensor types automatically
    use_auth_token=hf_token  # Use the token for authentication
)
model.cuda()  # Move model to GPU if CUDA is available

# Text generation function
def generate(prompt: str, generation_params: dict = {"max_length": 200}) -> str:
    try:
        torch.cuda.empty_cache()  # Clear CUDA cache
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        tokens = model.generate(**inputs, **generation_params)
        completion = tokenizer.decode(tokens[0], skip_special_tokens=True)
        return completion
    except Exception as e:
        return f"Error generating text: {str(e)}"

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate code from a prompt using a pretrained model.")
    parser.add_argument('prompt', type=str, help='A prompt for the model')
    args = parser.parse_args()

    # Generate and print the result
    result = generate(args.prompt)
    print(result)

if __name__ == "__main__":
    main()
