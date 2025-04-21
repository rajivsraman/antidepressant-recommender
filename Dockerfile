FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install wget for model download
RUN apt-get update && apt-get install -y wget && apt-get clean

# Download DistilBERT model files into the container
RUN mkdir -p models/deep_model && \
    wget https://storage.googleapis.com/adrs-distilbert/deep_model/model.safetensors -O models/deep_model/model.safetensors && \
    wget https://storage.googleapis.com/adrs-distilbert/deep_model/config.json -O models/deep_model/config.json && \
    wget https://storage.googleapis.com/adrs-distilbert/deep_model/tokenizer_config.json -O models/deep_model/tokenizer_config.json && \
    wget https://storage.googleapis.com/adrs-distilbert/deep_model/tokenizer.json -O models/deep_model/tokenizer.json && \
    wget https://storage.googleapis.com/adrs-distilbert/deep_model/vocab.txt -O models/deep_model/vocab.txt && \
    wget https://storage.googleapis.com/adrs-distilbert/deep_model/special_tokens_map.json -O models/deep_model/special_tokens_map.json

# Copy the rest of your code
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
