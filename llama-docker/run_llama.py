from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from huggingface_hub import login
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_and_download_model(model_path: str):
    """Download the model if it doesn't exist"""
    try:
        model_name = "meta-llama/Llama-2-7b-hf"
        
        # Verify HF token
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("No HF_TOKEN found in environment variables")
        
        # Check if model already exists
        if Path(model_path).exists() and any(Path(model_path).iterdir()):
            logging.info("Model already exists in local directory")
            return
        
        # Login to Hugging Face
        login(token=hf_token, write_permission=False)
        
        logging.info(f"Downloading model {model_name} to {model_path}")
        
        # Download model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            token=hf_token
        )
        model.save_pretrained(model_path)
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token
        )
        tokenizer.save_pretrained(model_path)
        
        logging.info("Model and tokenizer downloaded successfully!")
        
    except Exception as e:
        logging.error(f"Error downloading model: {str(e)}")
        raise

def format_prompt(prompt, system_prompt="You are a helpful AI assistant. Provide clear, accurate, and concise answers."):
    """Format the prompt with system and user messages"""
    return f"""### System:
{system_prompt}

### User:
{prompt}

### Assistant:
"""

def generate_text(prompt, model, tokenizer, max_length=512):
    """Generate text from the model"""
    # Create generation config
    generation_config = GenerationConfig(
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        repetition_penalty=1.2
    )
    
    # Format prompt
    formatted_prompt = format_prompt(prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    outputs = model.generate(
        **inputs,
        generation_config=generation_config
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Assistant:")[-1].strip()

def main():
    try:
        model_path = "/app/model"
        os.makedirs(model_path, exist_ok=True)
        
        # Download model if needed
        verify_and_download_model(model_path)
        
        # Get the prompt from command line arguments
        prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Tell me about LLaMA."
        logging.info(f"Using prompt: {prompt}")
        
        logging.info("Loading model from local directory")
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Generate response
        response = generate_text(prompt, model, tokenizer)
        print(f"\nResponse: {response}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
