# Use the official Python 3.9 image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8501 \
    HOST=0.0.0.0

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p .streamlit data

# Copy Streamlit config
COPY .streamlit/config.toml .streamlit/config.toml

# Expose the port the app runs on
EXPOSE ${PORT}

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=${PORT}", "--server.address=${HOST}"]
