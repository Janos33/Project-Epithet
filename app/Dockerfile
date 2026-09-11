# syntax=docker/dockerfile:1

FROM python:3.11-slim

WORKDIR /code

ENV FLASK_APP=init.py
ENV FLASK_RUN_HOST=0.0.0.0

COPY requirements.txt .

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["flask", "run", "--debug"]