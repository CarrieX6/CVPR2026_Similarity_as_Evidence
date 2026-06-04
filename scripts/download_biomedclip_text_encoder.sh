#!/usr/bin/env bash
# BiomedCLIP 文本塔：microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract（与运行时 Transformers 请求一致）
# 用于 HF_HUB_OFFLINE=1 时设置 OPEN_CLIP_TEXT_ENCODER_LOCAL=/本目录

set -euo pipefail
OUT="${1:-./BiomedCLIP_text_encoder}"
MODEL_ID="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
REV="main"
mkdir -p "$OUT"

download() {
  local base="$1"
  local name="$2"
  local dest="$OUT/$name"
  local url="${base}/${MODEL_ID}/resolve/${REV}/${name}"
  echo "==> $name"
  if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$dest" "$url" || return 1
  elif command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$dest" "$url" || return 1
  else
    echo "需要 wget 或 curl" >&2
    return 1
  fi
  [ -s "$dest" ] || return 1
  return 0
}

for BASE in "https://hf-mirror.com" "https://huggingface.co"; do
  echo "--- 尝试 ${BASE} ---"
  rm -f "$OUT/pytorch_model.bin" "$OUT/model.safetensors" 2>/dev/null || true
  ok=1
  for f in config.json tokenizer_config.json vocab.txt; do
    if ! download "$BASE" "$f"; then
      echo "  缺少 $f" >&2
      ok=0
      break
    fi
  done
  [ "$ok" -eq 1 ] && { download "$BASE" "special_tokens_map.json" || true; }
  [ "$ok" -eq 1 ] && { download "$BASE" "tokenizer.json" || true; }
  if [ "$ok" -ne 1 ]; then
    continue
  fi
  if download "$BASE" "pytorch_model.bin"; then
    :
  elif download "$BASE" "model.safetensors"; then
    :
  else
    echo "  无 pytorch_model.bin 且无 model.safetensors" >&2
    ok=0
  fi
  if [ "$ok" -eq 1 ]; then
    echo ""
    echo "完成，目录: $OUT"
    echo "export OPEN_CLIP_TEXT_ENCODER_LOCAL=$(readlink -f "$OUT" 2>/dev/null || echo "$(cd "$OUT" && pwd)")"
    exit 0
  fi
done

echo "全部镜像失败；请在可联网环境手动下载该 HF 模型到 $OUT" >&2
exit 1
