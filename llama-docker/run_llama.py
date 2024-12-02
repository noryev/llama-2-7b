from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv
import os
from huggingface_hub import login, HfApi
import sys
from pathlib import Path
import argparse

# Load environment variables
load_dotenv()

def verify_token():
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("No HF_TOKEN found in environment variables")
    
    print("Verifying Hugging Face token...")
    try:
        api = HfApi()
        user = api.whoami(token=hf_token)
        print(f"Successfully authenticated as: {user['name']}")
        return hf_token
    except Exception as e:
        print(f"Token verification failed: {str(e)}")
        return None

def check_cache(model_name, cache_dir):
    """Check if model is already in cache"""
    cache_path = Path(cache_dir) / "models--meta-llama--Llama-2-7b-hf"
    if not cache_path.exists():
        print("Cache directory not found, will download model.")
        return False
        
    snapshots = list((cache_path / "snapshots").glob("*"))
    if not snapshots:
        print("No model snapshots found in cache.")
        return False
        
    latest_snapshot = sorted(snapshots)[-1]
    
    # Check for essential files
    if not (latest_snapshot / "config.json").exists():
        print("Cache incomplete.")
        return False
        
    print("Found model in cache!")
    return True

def load_model():
    model_name = "meta-llama/Llama-2-7b-hf"
    cache_dir = "/root/.cache/huggingface"
    hf_token = verify_token()
    
    if not hf_token:
        raise ValueError("Token verification failed")
    
    login(token=hf_token, write_permission=False)
    
    model_cached = check_cache(model_name, cache_dir)
    
    try:
        if model_cached:
            print("Loading model from cache...")
        else:
            print(f"Downloading model: {model_name} (this will take a while)...")
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_auth_token=hf_token,
            cache_dir=cache_dir
        )
        print("Model loaded successfully!")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_auth_token=hf_token,
            cache_dir=cache_dir
        )
        print("Tokenizer loaded!")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Model loading failed: {str(e)}", file=sys.stderr)
        raise

def generate_text(prompt, model, tokenizer, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text using Llama 2')
    parser.add_argument('prompt', type=str, help='The prompt to generate text from')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of generated text')
    
    args = parser.parse_args()
    
    try:
        model, tokenizer = load_model()
        response = generate_text(args.prompt, model, tokenizer, args.max_length)
        print("\nResponse:", response)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Go to https://huggingface.co/meta-llama/Llama-2-7b-hf")
        print("2. Click 'Access repository' and accept the terms")
        print("3. Make sure you're logged in with the same account that generated your token")
        print("4. Check you have enough disk space (need at least 15GB free)")
        sys.exit(1)