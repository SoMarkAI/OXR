#!/usr/bin/env bash
set -euo pipefail

# Use a complete local oxr-model checkout, including Git LFS files.
model_dir="${MODEL_PATH:-${MODEL_NAME:-./oxr-model}}"
if [[ ! -f "$model_dir/plugins/vllm/pyproject.toml" || ! -f "$model_dir/chat_template.jinja" ]]; then
  echo "Expected an oxr-model repository root with plugins/vllm and chat_template.jinja: $model_dir" >&2
  exit 1
fi
python3 -m pip install "$model_dir/plugins/vllm"
export VLLM_PLUGINS=oxr
exec vllm serve "$model_dir" \
  --trust-remote-code --served-model-name oxr \
  --chat-template "$model_dir/chat_template.jinja" \
  --dtype bfloat16 --max-model-len 16384 \
  --attention-config '{"backend":"TRITON_ATTN"}' \
  --host 0.0.0.0 --port 8080 "$@"
