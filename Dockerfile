# Use the official Python base image
FROM python:3.9

# Set working directory in the container
WORKDIR /app

# Copy all files in the directory into the container
COPY . .

# Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Expose any necessary ports
# EXPOSE 8000

# Command to run the Python file
# CMD ["python", "mnist_digit_recognition_api.py"]
CMD ["uvicorn", "mnist_digit_recognition_api:app", "--host", "0.0.0.0", "--port", "8000"]
