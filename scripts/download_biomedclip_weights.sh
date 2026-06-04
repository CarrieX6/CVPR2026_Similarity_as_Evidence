#!/usr/bin/env bash
# 在「无法使用 huggingface_hub」时，用 wget/curl 直链下载 BiomedCLIP 所需两个文件。
# 用法：
#   bash scripts/download_biomedclip_weights.sh /path/to/out_dir
# 然后：
#   export OPEN_CLIP_HF_LOCAL="/path/to/out_dir"
# 再运行 train.py

set -euo pipefail
OUT="${1:-./BiomedCLIP_hf_files}"
mkdir -p "$OUT"
MODEL_ID="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
REV="main"

download_one() {
  local base="$1"
  local name="$2"
  local url="${base}/${MODEL_ID}/resolve/${REV}/${name}"
  echo "==> $url"
  if command -v wget >/dev/null 2>&1; then
    wget -O "$OUT/$name" "$url" || return 1
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "$OUT/$name" "$url" || return 1
  else
    echo "需要 wget 或 curl" >&2
    return 1
  fi
}

for BASE in "https://hf-mirror.com" "https://huggingface.co"; do
  if download_one "$BASE" "open_clip_config.json" && download_one "$BASE" "open_clip_pytorch_model.bin"; then
    echo ""
    echo "已保存到: $OUT"
    echo "请执行: export OPEN_CLIP_HF_LOCAL=$(readlink -f "$OUT" 2>/dev/null || echo "$(cd "$OUT" && pwd)")"
    exit 0
  fi
  echo "镜像 ${BASE} 失败，尝试下一个..." >&2
done
echo "全部失败：请用手机热点浏览器打开 hf-mirror 该模型页，手动下载上述两文件到 $OUT" >&2
exit 1
