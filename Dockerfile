# Guardian of Truth - inference image
#
# The base image already contains vLLM (CUDA kernels) and a matching PyTorch.
# Model weights are downloaded at build time into /app/model; if you prefer to
# ship them inside the archive, put them into ./model before building and skip
# the download step with --build-arg SKIP_MODEL_DOWNLOAD=1.
FROM vllm/vllm-openai:v0.6.6

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# 1. Python package (layer cached independently of the weights).
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir /app

# 2. Entry points and configs.
COPY scripts /app/scripts
COPY configs /app/configs

# 3. Judge weights: use ./model if it already contains a checkpoint, otherwise
#    download them at build time.  SKIP_MODEL_DOWNLOAD=1 keeps the image small
#    when the weights are mounted at run time instead.
ARG MODEL_ID=Qwen/Qwen2.5-14B-Instruct-AWQ
ARG MODEL_DIR=/app/model
ARG SKIP_MODEL_DOWNLOAD=0
ENV HF_HOME=/app/.hf_cache
COPY model /app/model
RUN if [ "$SKIP_MODEL_DOWNLOAD" = "0" ] && [ ! -f "$MODEL_DIR/config.json" ]; then \
        python /app/scripts/download_model.py --model-id "$MODEL_ID" --out "$MODEL_DIR"; \
    else echo "model download skipped (bundled weights or SKIP_MODEL_DOWNLOAD=1)"; fi

ENV GUARDIAN_MODEL_ID=${MODEL_ID} \
    GUARDIAN_MODEL_PATH=${MODEL_DIR}

# The platform runs `python scripts/predict.py --input ... --output ...`;
# the default CMD is only a convenience for local docker runs.
ENTRYPOINT []
CMD ["python", "scripts/predict.py", "--input", "/data/test.parquet", "--output", "/data/predictions.parquet"]
