# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire workspace into the container
COPY . .

# Expose the web interface port
EXPOSE 8501

# Serve the HTML, CSS and JavaScript interface
CMD ["python", "app/app.py", "--host", "0.0.0.0", "--port", "8501"]
