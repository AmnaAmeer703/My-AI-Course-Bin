# Use Python 3.10.12 base image
FROM python:3.10.12-slim

# Set Working Directory
WORKDIR /app

# Copy requirements and install dependiences
COPY requirements.txt .
RUN pip install --no--cache-dir -r requirements.txt

# Copy rest of application code
COPY . .

# Expose the application port
EXPOSE 8000

# Comand to start FAST API application
CMD ['uvicorn','app:app','--host','0.0.0.0','--port','8000']