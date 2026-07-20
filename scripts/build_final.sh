#!/usr/bin/env bash
#
# build_final.sh — Build the 191-document explorer from the tri-model embeddings.
#
# Run after scripts/embed_final.py has produced the .npy files. Paths resolve
# relative to the repository, so this works from a fresh clone with no editing.
# Set PYTHON=... to use a specific interpreter (a virtualenv, say):
#
#     bash scripts/build_final.sh
#     PYTHON=.venv/bin/python bash scripts/build_final.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
CORPUS="$REPO/corpus_build"
OUT="$REPO/output"
PYTHON="${PYTHON:-python3}"

SEGMENTS="$CORPUS/final_segments.json"

# The corpus ships gzipped to keep the repository small; expand on first use.
if [[ ! -f "$SEGMENTS" ]]; then
  if [[ -f "$SEGMENTS.gz" ]]; then
    echo "Decompressing $(basename "$SEGMENTS").gz ..."
    gunzip -k "$SEGMENTS.gz"
  else
    echo "ERROR: neither $SEGMENTS nor $SEGMENTS.gz found." >&2
    exit 1
  fi
fi

missing=()
for m in bge_m3 qwen3 minilm; do
  [[ -f "$CORPUS/emb_$m.npy" ]] || missing+=("emb_$m.npy")
done
if (( ${#missing[@]} )); then
  echo "ERROR: missing embeddings in corpus_build/: ${missing[*]}" >&2
  echo "Generate them first:  $PYTHON scripts/embed_final.py" >&2
  exit 1
fi

mkdir -p "$OUT"

"$PYTHON" "$HERE/build_explorer.py" \
  --segments "$SEGMENTS" \
  --emb bge_m3="$CORPUS/emb_bge_m3.npy" \
  --emb qwen3="$CORPUS/emb_qwen3.npy" \
  --emb minilm="$CORPUS/emb_minilm.npy" \
  --doc-threshold 0.55 \
  --top-k-segments 600 \
  --datum-options "IE_LAND_USE_REVIEW_PHASE_2,EU_NRL,IE_CAP25,IE_CLIMATE_ACT,IE_NBAP,IE_FOOD_VISION,IE_FOREST_STRATEGY,IE_WAP24,EU_EU_WFD,IE_NPF,IE_EPA_SOE_2024" \
  --output "$OUT/policy_graph_191docs.html"

# docs/ is what GitHub Pages serves; keep it in step with the build.
cp "$OUT/policy_graph_191docs.html" "$REPO/docs/index.html"

echo "Built:     $OUT/policy_graph_191docs.html"
echo "Published: $REPO/docs/index.html"
