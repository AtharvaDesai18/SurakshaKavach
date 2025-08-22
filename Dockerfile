# Dockerfile

# Start with an official Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# The command to run the Flask application
# We use Gunicorn as a production-ready web server
# It will listen on all network interfaces (0.0.0.0) on port 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
