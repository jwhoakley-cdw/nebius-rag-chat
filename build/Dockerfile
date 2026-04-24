# Use the official Python image
FROM python:3.14-slim

# Set the working directory
WORKDIR /app

# Add any required OS pkgs
# Install some additional tools (TEMPORARY)
RUN apt-get update && \
    apt-get -y upgrade && \
#    apt-get -y install vim-tiny curl && \
    rm -rf /var/lib/apt/lists/*

# Copy over pip requirements
COPY . .

# pip install all python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default Streamlit port
EXPOSE 8501

# Run a quick healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
