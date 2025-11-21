# Dockerfile for EC2 model API
FROM python:3.10-slim

WORKDIR /app

# Install system deps if needed
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements-ec2.txt .
RUN pip install --no-cache-dir -r requirements-ec2.txt

# Copy app and model folder
COPY app.py .
COPY model/ model/

EXPOSE 80

# Use uvicorn for production; could also use gunicorn+uvicorn workers
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]
