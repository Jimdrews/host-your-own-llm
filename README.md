# host-your-own-llm

Run any Hugging Face model on your own GPU behind an OpenAI-compatible API.
Models are downloaded once into a local store and reused, so switching between them is quick.

## Requirements

- Docker with NVIDIA GPU support (NVIDIA Container Toolkit, or Docker Desktop with WSL2)
- [uv](https://docs.astral.sh/uv/)

## Quick start

```sh
uv run llm serve
```

The first run asks which model to use (pick a preset, enter any Hugging Face id, or browse popular ones), the context length and whether to require an API key, then starts everything and prints a ready-to-use `curl` command.

The API is at `http://localhost:8000/v1` and works with any OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<VLLM_API_KEY or anything>")
reply = client.chat.completions.create(model="qwen2.5-0.5b-instruct", messages=[{"role": "user", "content": "Hello!"}])
```

## Commands

| Command | What it does |
|---|---|
| `llm serve` | Start the model store and the model server |
| `llm serve --reconfigure` | Ask the setup questions again |
| `llm status` | Show what's running, the active model and the endpoint |
| `llm models search [query]` | Find popular Hugging Face models that fit your GPU (`--trending` for what's new, `--all` to include ones too big) |
| `llm models list` | List stored models (`*` marks the active one) |
| `llm models add <id>` | Download a Hugging Face model into the store |
| `llm models use <id> [--name <name>]` | Switch the served model, downloading it first if needed |
| `llm models rm <id> [-y]` | Delete a stored model |
| `llm down` | Stop everything; stored models are kept |
| `llm down --purge` | Stop everything and delete all stored models |

Prefix each with `uv run`. Add `--help` to any command for details.

## Configuration

Settings live in `.env`, written by the setup questions. You can also edit it by hand; the next `llm serve` picks up the changes.

| Setting | Purpose |
|---|---|
| `MODEL` | The model to serve |
| `SERVED_MODEL_NAME` | The model name used in API requests |
| `MAX_MODEL_LEN` | Context length; lower it if the model runs out of GPU memory |
| `VLLM_API_KEY` | Require `Authorization: Bearer <key>` on requests |
| `HF_TOKEN` | Hugging Face token, needed for gated models such as Llama |
| `EXTRA_ARGS` | Any other `vllm serve` flags, e.g. `--enable-prefix-caching` |
| `API_PORT` | API port (default `8000`) |

See [`.env.example`](.env.example) for every setting.

## Using the image without the CLI

The vLLM image in `image/` runs on its own and takes the same settings as environment variables:

```sh
docker build -t host-your-own-llm/vllm image/
docker run --gpus all --ipc=host -p 8000:8000 -e MODEL=Qwen/Qwen2.5-0.5B-Instruct host-your-own-llm/vllm
```

`MODEL` can be a Hugging Face id or an `s3://bucket/path`. For S3, also pass `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_DEFAULT_REGION`, plus `AWS_ENDPOINT_URL` and `RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0` for stores other than AWS.

## Tests

```sh
uv run pytest tests/unit   # fast, no Docker needed
uv run pytest              # adds end-to-end tests; needs Docker
uv run pytest -m gpu       # serves real models; needs a free GPU
```
