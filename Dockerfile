# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy setup.py + requirements first
COPY requirements.txt .
COPY setup.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]