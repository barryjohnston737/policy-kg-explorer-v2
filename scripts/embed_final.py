#!/usr/bin/env python3
"""
embed_final.py — Tri-model embed of the 191-document corpus (13,871 segments).

Reads corpus_build/final_segments.json (decompressing final_segments.json.gz on
first run) and writes corpus_build/emb_bge_m3.npy, emb_qwen3.npy and
emb_minilm.npy. Sequences are capped at 1024 tokens — segments are ~350 words,
so nothing is truncated in practice, but the cap stops a pathological segment
exhausting attention memory.

Completed models are detected and skipped, so an interrupted run can simply be
restarted.

    python3 scripts/embed_final.py

Requires sentence-transformers. The first run downloads ~3.5 GB of model
weights and takes roughly 1-2 hours on Apple silicon; subsequent runs reuse the
cached weights. Then build the explorer with:

    bash scripts/build_final.sh
"""
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "corpus_build"
SEGMENTS = CORPUS / "final_segments.json"

MODELS = {
    "bge_m3": "BAAI/bge-m3",
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}
BATCH = 8
MAX_SEQ = 1024


def load_segments() -> list:
    """Return the segment list, expanding the shipped .gz on first use."""
    if not SEGMENTS.exists():
        gz = SEGMENTS.with_suffix(".json.gz")
        if not gz.exists():
            sys.exit(f"ERROR: neither {SEGMENTS} nor {gz} found.")
        print(f"Decompressing {gz.name} ...")
        with gzip.open(gz, "rb") as fin, open(SEGMENTS, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    with open(SEGMENTS, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    texts = [s["content"] for s in load_segments()]
    print(f"{len(texts)} segments to embed")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("ERROR: sentence-transformers not installed.\n"
                 "       pip install sentence-transformers")

    for name, model_id in MODELS.items():
        out = CORPUS / f"emb_{name}.npy"
        if out.exists() and np.load(out, mmap_mode="r").shape[0] == len(texts):
            print(f"[{name}] already complete — skipping")
            continue

        print(f"[{name}] loading {model_id} ...")
        model = SentenceTransformer(model_id)
        if model.max_seq_length > MAX_SEQ:
            model.max_seq_length = MAX_SEQ

        t0 = time.time()
        emb = model.encode(texts, batch_size=BATCH, show_progress_bar=True,
                           normalize_embeddings=True)
        np.save(str(out), np.asarray(emb, dtype=np.float32))
        print(f"[{name}] {np.asarray(emb).shape} in {time.time() - t0:.0f}s -> {out.name}")

        del model
        try:                                    # free accelerator memory between models
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print("\nDone. Build the explorer with:  bash scripts/build_final.sh")


if __name__ == "__main__":
    main()
