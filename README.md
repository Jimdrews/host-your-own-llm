# host-your-own-llm

Serve an LLM through vLLM's OpenAI-compatible API in Docker, with model weights kept in an S3 bucket instead of in the image or the serving container.

The repo has two independent parts:

- **`image/`**: a self-contained vLLM image, configured only through environment variables. Build and deploy it however you like.
- **The `llm` CLI + `docker-compose.yml`**: an optional local stack that runs the image next to a local S3 store ([RustFS](https://github.com/rustfs/rustfs)).

## Local stack

Requires Docker with NVIDIA GPU support (NVIDIA Container Toolkit / Docker Desktop with WSL2) and [uv](https://docs.astral.sh/uv/).

```sh
uv run llm serve               # first run asks which model to use, then starts everything
uv run llm status              # what's running, the active model and the endpoint
uv run llm models list         # stored models (* = active)
uv run llm models add <id>     # copy a Hugging Face model into the store
uv run llm models use <id>     # switch the served model (adds it first if needed)
uv run llm models rm <id>      # delete a stored model
uv run llm down                # stop everything; stored models are kept
uv run llm down --purge        # stop and delete all stored models
```

`serve --reconfigure` asks the setup questions again, including a guided model picker. Any command run without its required arguments shows its help.

- API: http://localhost:8000/v1
- S3 API: http://localhost:9000, web console: http://localhost:9001 (log in with `S3_ACCESS_KEY` / `S3_SECRET_KEY` from `.env`)
- Stored models live in the `model-store` volume.

## Image

```sh
docker build -t host-your-own-llm/vllm image/
```

The container runs `vllm serve` from environment variables:

| Variable | Purpose |
|---|---|
| `MODEL` | **Required.** `s3://bucket/path`, a Hugging Face id, or a local path. `s3://` models stream straight to the GPU (`--load-format runai_streamer`) and must use `.safetensors` weights. |
| `MAX_MODEL_LEN` | Context length cap. |
| `GPU_MEMORY_UTILIZATION` | Fraction of GPU memory vLLM may use (default `0.90`). |
| `SERVED_MODEL_NAME` | Model name for API requests (default: `MODEL`). |
| `VLLM_API_KEY` | Require `Authorization: Bearer <key>`. |
| `EXTRA_ARGS` | Any other `vllm serve` flags. |
| `VLLM_PORT` | Listen port (default `8000`). |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` | S3 credentials; on AWS an IAM role works instead. |
| `AWS_ENDPOINT_URL`, `RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0` | Only for S3-compatible stores other than AWS. |

Arguments after the image name are passed straight to `vllm serve`. Run with `--gpus all --ipc=host`. Pin a different vLLM release with `--build-arg VLLM_VERSION=<tag>`.

The image also carries the model store tool, which can run as a one-off job in any deployment (it reads `S3_BUCKET`, the `AWS_*` variables and `HF_TOKEN`):

```sh
docker run --rm --entrypoint python3 <image> /opt/model_store.py seed <hf-id>   # or: list, rm <id>
```

`seed` with no id seeds whatever `MODEL` points to, and skips models already in the bucket.

## Testing

```sh
uv run pytest            # unit + functional tests (~3-4 min; needs Docker and network, no GPU)
uv run pytest -m gpu     # serving on the GPU (~5-10 min; downloads ~1.3 GB)
```

The functional and GPU tests run the real `llm` commands against a real model store and real Hugging Face downloads, in an isolated copy of the project (compose projects `hyol-test` / `hyol-gpu-test`, ports 18000/19000/19001, their own volume and `.env`). They never touch your own stack, `.env` or stored models, and clean up after themselves. The GPU tests need the GPU free, so run `uv run llm down` first.
