# Model weights

This directory is the expected location of the judge weights inside the image
(`GUARDIAN_MODEL_PATH`). Weights are **not** committed to git.

Two options:

1. **Download at Docker build time** (default):

   ```bash
   docker build -t guardian-of-truth .
   ```

   The Dockerfile calls `scripts/download_model.py` and stores
   `Qwen/Qwen2.5-14B-Instruct-AWQ` (~9 GB) here.

2. **Ship the weights inside the archive** (the whole solution must stay below
   40 GB):

   ```bash
   python scripts/download_model.py --model-id Qwen/Qwen2.5-14B-Instruct-AWQ --out model
   docker build --build-arg SKIP_MODEL_DOWNLOAD=1 -t guardian-of-truth .
   ```

For a local run without Docker:

```bash
python scripts/download_model.py --out model
python scripts/predict.py --input test.parquet --output predictions.parquet
```

Any Hugging Face causal-LM with a chat template works (`--model` / `--model-path`,
`configs/default.json → model.id`). Reasonable alternatives:

| model | size (AWQ) | notes |
|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct-AWQ` | ~4.5 GB | fastest, safest for the 30-minute limit |
| `Qwen/Qwen2.5-14B-Instruct-AWQ` | ~9 GB | default, better reasoning |
| `Qwen/Qwen2.5-32B-Instruct-AWQ` | ~19 GB | best quality, may exceed the time budget |
