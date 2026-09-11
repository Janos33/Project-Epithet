# syntax=docker/dockerfile:1

FROM python:3.11-slim

WORKDIR /code

# Added python3-dev to ensure hdbscan builds successfully
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Pre-install CPU-only PyTorch (Keeps image light and speeds up builds)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt 

RUN python -m spacy download en_core_web_sm

COPY . .