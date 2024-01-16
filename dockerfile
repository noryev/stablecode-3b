# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /usr/src/app

# Install git, necessary for some huggingface models
RUN apt-get update && apt-get install -y git

# Install any needed packages specified in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

# Copy your local model directory into the container
# Replace `path_to_your_model_directory` with the actual path to your local model directory
COPY path_to_your_model_directory /usr/src/app/stablecode

# Modify your Python script to load the model from the copied directory
# Ensure loadmodel.py uses the path '/usr/src/app/model' for loading the model

# Use ENTRYPOINT to specify the script to be run, allowing for additional command-line arguments
ENTRYPOINT ["python", "/usr/src/app/loadmodel.py"]
