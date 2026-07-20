#!/bin/bash
# Build the final 191-doc explorer from the tri-model embeddings.
# Run after embed_final.py completes.
set -e
KG="$HOME/Desktop/KG_explorer - updated colours"
HERE="$(cd "$(dirname "$0")" && pwd)"

"$KG/.venv/bin/python" "$KG/policy_kg_explorer_2_3d_new_3.py" \
  --segments "$HERE/merged_segments.json" \
  --emb bge_m3="$HERE/merged_bge_m3.npy" \
  --emb qwen3="$HERE/merged_qwen3.npy" \
  --emb minilm="$HERE/merged_minilm.npy" \
  --doc-threshold 0.55 \
  --top-k-segments 600 \
  --datum-options "IE_LAND_USE_REVIEW_PHASE_2,EU_NRL,IE_CAP25,IE_CLIMATE_ACT,IE_NBAP,IE_FOOD_VISION,IE_FOREST_STRATEGY,IE_WAP24,EU_EU_WFD,IE_NPF,IE_EPA_SOE_2024" \
  --output "$HERE/policy_graph_191docs.html"

echo "Built: $HERE/policy_graph_191docs.html"
