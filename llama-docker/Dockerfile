FROM pytorch/pytorch:latest

WORKDIR /app

# Install required packages
RUN pip install transformers accelerate bitsandbytes sentencepiece python-dotenv huggingface_hub

# Create cache directory
RUN mkdir -p /root/.cache/huggingface

# Copy the Python script and .env file
COPY run_llama.py /app/
COPY .env /app/

# Set environment variables properly
ENV $(cat .env | xargs)

ENTRYPOINT ["python", "run_llama.py"]