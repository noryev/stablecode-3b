# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /usr/src/app

# Install git, necessary for some huggingface models
RUN apt-get update && apt-get install -y git

# Install any needed packages specified in requirements.txt
# Ensure that requirements.txt is present in the same directory as the Dockerfile
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

# Copy your local model directory into the container
# Make sure the stablecode directory is in the same directory as your Dockerfile
COPY stablecode /usr/src/app/model

# Modify your Python script to load the model from the copied directory
# Ensure loadmodel.py uses the path '/usr/src/app/model' for loading the model

# Use ENTRYPOINT to specify the script to be run, allowing for additional command-line arguments
ENTRYPOINT ["python", "/usr/src/app/loadmodel.py"]
