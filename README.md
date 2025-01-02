# LLaMA-2-7B Docker Implementation

Run Meta's LLaMA-2-7B language model locally using Docker with GPU support.

## Quick Start

1. Navigate to the project directory:
```bash
cd llama-docker
```

2. Set up credentials:
   - Create a `.env` file
   - Add your HuggingFace token:
```bash
HF_TOKEN=your-token-here
```

3. Build and run:
```bash
# Build Docker image
docker build -t llama2-model .

# Run with your prompt
docker run --rm -it --gpus all llama2-model "your prompt here" --max_length 150
```

## Additional Options

- Adjust response length: `--max_length 300`
- Custom system prompt: `--system_prompt "You are an expert in..."`

## Requirements

- Docker with GPU support
- NVIDIA GPU with CUDA
- HuggingFace account with access to LLaMA-2
