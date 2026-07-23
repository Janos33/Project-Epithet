# syntax=docker/dockerfile:1

FROM python:3.11-slim

WORKDIR /code

ENV FLASK_APP=init.py
ENV FLASK_RUN_HOST=0.0.0.0

# Installs gcc and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# STEP 1: Pre-install CPU-only PyTorch
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# STEP 2: Copy requirements
COPY requirements.txt .

# STEP 3: Install requirements using CPU index & no-cache
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt 

COPY . . 

EXPOSE 5000
CMD ["flask", "run", "--debug"]