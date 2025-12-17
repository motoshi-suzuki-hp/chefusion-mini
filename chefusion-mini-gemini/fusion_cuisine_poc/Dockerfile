FROM python:3.11-slim

# Install system dependencies including make and unzip
RUN apt-get update && \
    apt-get install -y build-essential unzip && \
    rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Set working directory
WORKDIR /app

# Copy only the pyproject.toml first to leverage Docker cache
COPY pyproject.toml /app/

# Generate the lock file and install dependencies
RUN poetry config virtualenvs.create false && \
    poetry lock --no-interaction && \
    poetry install --no-root --without dev --no-interaction --no-ansi

# Copy the rest of the application
COPY . /app

# Set environment variables
ENV PYTHONPATH=/app

CMD ["bash"]