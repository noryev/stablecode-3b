import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

# Define the directory where the model is stored
model_dir = os.getenv("MODEL_DIR", "/path/to/model/directory")

# Load the tokenizer and model from the local directory
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.cuda()  # Move model to GPU if CUDA is available

# Text generation function
def generate(prompt: str, generation_params: dict = {"max_length": 200}) -> str:
    try:
        torch.cuda.empty_cache()  # Clear CUDA cache
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Filter out unsupported generation parameters
        supported_params = {key: val for key, val in generation_params.items() if key in model.config.to_diff_dict()}

        tokens = model.generate(**inputs, **supported_params)

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
