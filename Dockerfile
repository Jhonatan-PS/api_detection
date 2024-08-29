FROM python:3.12.2-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libstdc++6 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "app:app"]