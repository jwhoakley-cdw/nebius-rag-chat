# Use the official Python image
FROM python:3.14-slim

# Set the working directory
WORKDIR /app

# Add any required OS pkgs
# Install some additional tools (TEMPORARY)
RUN apt-get update && \
    apt-get -y upgrade && \
    rm -rf /var/lib/apt/lists/*

# Copy over pip requirements
COPY . .

# pip install all python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Remove unnecessary files
RUN rm Dockerfile

# Expose the default Streamlit port
EXPOSE 8501

# Run a quick healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
