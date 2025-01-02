from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from dotenv import load_dotenv
import os
from huggingface_hub import login, HfApi
import sys
from pathlib import Path
import argparse

# Load environment variables and set up caching
load_dotenv()

# Set up transformers caching
os.environ['TRANSFORMERS_CACHE'] = os.getenv('TRANSFORMERS_CACHE', 
                                           os.path.expanduser('~/.cache/huggingface'))
os.environ['HF_HOME'] = os.getenv('HF_HOME', 
                                os.path.expanduser('~/.cache/huggingface'))

# Ensure cache directories exist
os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

print(f"Using cache directory: {os.environ['TRANSFORMERS_CACHE']}")

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
    """Check if model is already in cache and complete"""
    # Convert model name to cache directory format
    model_name_formatted = model_name.replace("/", "--")
    cache_path = Path(cache_dir) / f"models--{model_name_formatted}"
    
    if not cache_path.exists():
        print("Cache directory not found, will download model.")
        return False
        
    # Check for specific required files
    required_files = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "model.safetensors.index.json"
    ]
    
    snapshots = list((cache_path / "snapshots").glob("*"))
    if not snapshots:
        print("No model snapshots found in cache.")
        return False
        
    latest_snapshot = sorted(snapshots)[-1]
    
    # Check all required files
    missing_files = []
    for file in required_files:
        if not (latest_snapshot / file).exists():
            missing_files.append(file)
    
    # Check model shards
    model_shards = list(latest_snapshot.glob("model-*.safetensors"))
    if not model_shards:
        missing_files.append("model shards")
    
    if missing_files:
        print(f"Cache incomplete. Missing: {', '.join(missing_files)}")
        return False
        
    print(f"Found complete model in cache at: {latest_snapshot}")
    return True

def load_model():
    model_name = "meta-llama/Llama-2-7b-hf"
    # Use a more persistent cache location
    cache_dir = os.getenv('TRANSFORMERS_CACHE', 
                         os.path.expanduser('~/.cache/huggingface'))
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
            
        # Improved model loading configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            token=hf_token,  # Updated from use_auth_token
            cache_dir=cache_dir
        )
        print("Model loaded successfully!")
        
        # Improved tokenizer configuration
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=hf_token,  # Updated from use_auth_token
            cache_dir=cache_dir,
            padding_side="left"  # Better for chat completion
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("Tokenizer loaded!")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Model loading failed: {str(e)}", file=sys.stderr)
        raise

def format_prompt(prompt, system_prompt="You are a helpful AI assistant. Provide clear, accurate, and concise answers."):
    """Format the prompt with clear structure"""
    return f"""### System:
{system_prompt}

### User:
{prompt}

### Assistant:
"""

def generate_text(prompt, model, tokenizer, max_length=512):
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
    
    # Generate with improved parameters
    outputs = model.generate(
        **inputs,
        generation_config=generation_config
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Clean up response to only include the Assistant's reply
    response = response.split("### Assistant:")[-1].strip()
    
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text using Llama 2')
    parser.add_argument('prompt', type=str, help='The prompt to generate text from')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum length of generated text')
    parser.add_argument('--system_prompt', type=str, 
                      default="You are a helpful AI assistant. Provide clear, accurate, and concise answers.",
                      help='System prompt to guide the model behavior')
    
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
        print("3. Make sure your huggingface token is functioning")
        print("4. Check you have enough disk space")
        sys.exit(1)

Version 4 of 4



Publish
