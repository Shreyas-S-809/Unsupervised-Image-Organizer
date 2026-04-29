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

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the Streamlit app when the container starts
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
