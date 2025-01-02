FROM pytorch/pytorch:latest

WORKDIR /app

# Install required packages
RUN pip install transformers accelerate bitsandbytes sentencepiece python-dotenv huggingface_hub

# Create model directory
RUN mkdir -p /app/model

# Copy the Python scripts and .env file
COPY download_model.py /app/
COPY run_llama.py /app/
COPY .env /app/

# Set environment variables from .env
ENV $(cat .env | xargs)

# Download model during build
RUN python download_model.py

ENTRYPOINT ["python", "run_llama.py"]
