FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

RUN python3.11 -m pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install "transformers>=4.44.2" "accelerate>=0.34.2" "datasets>=2.21.0" \
                "bitsandbytes>=0.43.3" "peft>=0.12.0" "trl>=0.9.6" sentencepiece protobuf

CMD ["python3.11", "train_maid.py"]
