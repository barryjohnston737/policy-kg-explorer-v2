#!/usr/bin/env python3
"""
embed_final.py — Tri-model embed of the final 191-doc corpus (13,871 segments).

Reads merged_segments.json here, writes merged_bge_m3.npy / merged_qwen3.npy /
merged_minilm.npy alongside it. Sequences capped at 1024 tokens (segments are
~350 words) so no pathological segment blows up attention memory.

Run on the Mac (needs the models; ~1-2h first time, downloads ~3.5GB weights):
    cd "corpus_build/embed"
    "$HOME/Desktop/KG_explorer - updated colours/.venv/bin/python" embed_final.py
"""
import json, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
SEGMENTS = HERE / "merged_segments.json"
MODELS = {
    "bge_m3": "BAAI/bge-m3",
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}
BATCH = 8
MAX_SEQ = 1024


def main():
    texts = [s["content"] for s in json.load(open(SEGMENTS, encoding="utf-8"))]
    print(f"{len(texts)} segments to embed")
    from sentence_transformers import SentenceTransformer

    for name, mid in MODELS.items():
        out = HERE / f"merged_{name}.npy"
        if out.exists() and np.load(out, mmap_mode="r").shape[0] == len(texts):
            print(f"[{name}] already complete — skipping")
            continue
        print(f"[{name}] loading {mid}...")
        m = SentenceTransformer(mid)
        if m.max_seq_length > MAX_SEQ:
            m.max_seq_length = MAX_SEQ
        t0 = time.time()
        emb = m.encode(texts, batch_size=BATCH, show_progress_bar=True,
                       normalize_embeddings=True)
        np.save(str(out), np.asarray(emb, dtype=np.float32))
        print(f"[{name}] {np.asarray(emb).shape} in {time.time()-t0:.0f}s -> {out.name}")
        del m
        try:
            import torch
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            torch.cuda.empty_cache()
        except Exception:
            pass
    print("\nDone. Build the explorer with build_final.sh")


if __name__ == "__main__":
    main()
