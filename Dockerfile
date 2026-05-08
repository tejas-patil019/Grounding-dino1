FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda \
    SETUPTOOLS_USE_DISTUTILS=stdlib \
    WEB_CONCURRENCY=1 \
    UVICORN_WORKERS=1 \
    HF_HOME=/opt/hf \
    TRANSFORMERS_CACHE=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_TIMEOUT=60
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    git \
    python3-opencv \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git

RUN mkdir weights && cd weights && \
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

RUN cd GroundingDINO && pip install .
RUN pip uninstall -y numpy transformers && \
    pip install \
      "numpy<2.0" \
      "transformers==4.36.2" \
      fastapi \
      uvicorn \
      python-multipart
RUN python - <<EOF
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained("bert-base-uncased")
AutoModel.from_pretrained("bert-base-uncased")
EOF
COPY server.py server.py
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

