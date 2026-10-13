#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${MODEL:-}" ]]; then
    echo "error: MODEL is not set. Pass a Hugging Face model id, e.g. -e MODEL=Qwen/Qwen2.5-0.5B-Instruct" >&2
    exit 1
fi

args=(
    "$MODEL"
    --host 0.0.0.0
    --port "${VLLM_PORT:-8000}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
)

[[ "$MODEL" == s3://* ]] && args+=(--load-format runai_streamer)
[[ -n "${MAX_MODEL_LEN:-}" ]] && args+=(--max-model-len "$MAX_MODEL_LEN")
[[ -n "${SERVED_MODEL_NAME:-}" ]] && args+=(--served-model-name "$SERVED_MODEL_NAME")
[[ -n "${VLLM_API_KEY:-}" ]] && args+=(--api-key "$VLLM_API_KEY")

# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && args+=(${EXTRA_ARGS})

exec vllm serve "${args[@]}" "$@"
