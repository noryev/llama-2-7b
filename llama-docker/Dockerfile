FROM pytorch/pytorch:latest

WORKDIR /app

# Install required packages
RUN pip install transformers accelerate bitsandbytes sentencepiece python-dotenv huggingface_hub

# Create model directory
RUN mkdir -p /app/model

# Copy the Python script
COPY run_llama.py /app/

# Copy .env file and set permissions
COPY .env /app/.env
RUN chmod 600 /app/.env

# More secure way to load environment variables at runtime
ENTRYPOINT ["sh", "-c", "export $(cat /app/.env | xargs) && python run_llama.py \"$@\"", "--"]
