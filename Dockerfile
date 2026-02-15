FROM python:3.10-slim-bullseye

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# --- NEW STEP: Upgrade pip ---
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements
COPY requirements.txt ./

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY app.py ./

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]