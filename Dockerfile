# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy all the files from the current directory to the container
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV PORT 5000

# Run the application when the container launches
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
