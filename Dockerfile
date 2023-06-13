# Base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the required packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . .

# Expose the port your Flask app runs on
EXPOSE 5000

# Set the environment variable for Flask
#ENV FLASK_APP=your_app_file.py

# Run the Flask application
CMD ["python", "main.py"]
