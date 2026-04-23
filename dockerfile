# Use slim Python base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

# Install system dependencies (lightweight)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Add src to PYTHONPATH
ENV PYTHONPATH=/app/src

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]