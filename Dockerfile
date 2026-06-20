# Dockerfile
FROM python:3.10-slim

# System dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy project files
COPY engine/ ./engine/
COPY static/ ./static/
COPY api.py .

# Expose port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]