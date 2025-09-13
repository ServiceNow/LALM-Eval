FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel

RUN mkdir -p /app /cache && chmod 777 /app /cache

WORKDIR /app

RUN groupadd -r appuser && useradd -r -g appuser -m appuser
RUN chown -R appuser:appuser /app /cache

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    MODEL_PATH="nvidia/audio-flamingo-3" \
    USE_THINK_MODE="true" \
    PORT=8000 \
    PYTHONPATH="/app:/app/audio-flamingo-3:/app/audio-flamingo-3/llava/model:/app/audio-flamingo/llava/model/quantization" \
    HF_HOME="tmp/cache/huggingface"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy project files
COPY requirements.txt /app/
COPY app.py models.py utils.py infer.py /app/
COPY audio-flamingo-3 /app/audio-flamingo-3/

RUN chown -R appuser:appuser /app

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir deepspeed==0.15.4

RUN pip install --no-cache-dir flash-attn==2.7.3

WORKDIR /app/audio-flamingo-3

WORKDIR /app

RUN site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])') && \
    if [ -d "/app/audio-flamingo-3/llava/train/deepspeed_replace" ]; then \
        mkdir -p $site_pkg_path/deepspeed && \
        cp -rv /app/audio-flamingo-3/llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/ && \
        chmod -R 755 $site_pkg_path/deepspeed/ || echo "Warning: Could not copy deepspeed files"; \
    fi

EXPOSE 8000

COPY --chown=appuser:appuser start.sh /app/start.sh
RUN chmod +x /app/start.sh

USER appuser

CMD ["/app/start.sh"]
