FROM python:3.11-slim

WORKDIR /app

# System dependencies including ffmpeg for mobile video support
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU (needs special index URL)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies with pinned versions
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "opencv-python-headless==4.8.1.78" \
    "fastapi==0.104.1" \
    "uvicorn==0.24.0" \
    "python-multipart==0.0.6" \
    "timm>=0.9.0" \
    "Pillow>=9.0.0"

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
