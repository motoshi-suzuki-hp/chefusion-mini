FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OFFLINE_MODE=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies directly with pip for better reliability
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    torchaudio==2.3.0 \
    torch_geometric \
    transformers \
    diffusers \
    datasets \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    pillow \
    requests \
    tqdm \
    pyyaml \
    openai \
    accelerate \
    scipy \
    sentence-transformers \
    jupyterlab \
    jupyter \
    ipywidgets \
    plotly \
    nltk \
    spacy \
    networkx \
    pytorch-lightning

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY config.yaml ./
COPY Makefile ./

# Create necessary directories
RUN mkdir -p /workspace/outputs/{images,recipes}
RUN mkdir -p /workspace/app/{data,models,notebooks}

# Install JupyterLab extensions and configure
RUN jupyter lab --generate-config
RUN echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.port = 8888" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py

# Set default command
CMD ["bash"]

# Expose JupyterLab port
EXPOSE 8888