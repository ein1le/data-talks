# Use an official slim Python image
FROM python:3.13-slim

# Install system-level build deps only if needed; minimal is fine for now
# RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code into the container
COPY scripts ./scripts
COPY web_service ./web_service
COPY model_registry ./model_registry

# Expose the port Waitress will listen on
EXPOSE 8000

# - Use Waitress to serve the Flask app object "app" from module "web_service.app"
CMD ["python", "-m", "waitress", "--host=0.0.0.0", "--port=8000", "web_service.app:app"]
