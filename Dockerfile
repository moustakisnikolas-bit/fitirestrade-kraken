FROM python:3.12-slim

WORKDIR /app

# libgomp1 = OpenMP runtime required by LightGBM (absent on python:*-slim)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/data

# Unbuffered so `docker logs` is live
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
CMD ["python", "bot.py"]
