FROM python:3.9-slim-buster

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    libgpiod2 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    python3-dev \
    make \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://www.piwheels.org/simple -r requirements.txt

# Copy the rest of the application
COPY src/ ./src/
COPY models/ ./models/

CMD ["python", "-u", "src/yolo_bird_monitor.py"]