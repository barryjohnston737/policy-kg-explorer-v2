#!/usr/bin/env python3
"""
ingest_corpus.py — Ingest new policy documents into the knowledge graph.

Reads your reviewed corpus_mapping CSV/XLSX, extracts text from PDFs/TXT,
segments them, generates dual-model embeddings (BGE + MiniLM), and outputs
files ready to merge with your existing 22-doc graph.

Usage:
    python ingest_corpus.py \
        --mapping corpus_mapping_xlsx.csv \
        --docs-dir "C:/path/to/full_doc_database" \
        --output-dir ./ingested \
        --existing-segments all_segments.json \
        --existing-bge embeddings_bge.npy \
        --existing-minilm embeddings_minilm.npy

Outputs (in --output-dir):
    new_segments.json          Segments from the 73 new docs only
    new_embeddings_bge.npy     BGE embeddings for new segments
    new_embeddings_minilm.npy  MiniLM embeddings for new segments
    merged_segments.json       All segments (existing 22 + new 73)
    merged_bge.npy             All BGE embeddings merged
    merged_minilm.npy          All MiniLM embeddings merged
    ingestion_report.json      Stats and any errors

Requirements:
    pip install pdfplumber sentence-transformers torch numpy

If you don't have GPU, sentence-transformers will use CPU (slower but works).
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_text_pdf(filepath):
    """Extract text from PDF using pdfplumber."""
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(str(filepath)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text.strip())
        return "\n\n".join(pages)
    except Exception as e:
        logger.warning(f"pdfplumber failed on {filepath}: {e}")
        # Fallback to PyPDF2
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(filepath))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text.strip())
            return "\n\n".join(pages)
        except Exception as e2:
            logger.error(f"All PDF extraction failed on {filepath}: {e2}")
            return ""


def extract_text_txt(filepath):
    """Extract text from a plain text file."""
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            return Path(filepath).read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    logger.error(f"Could not decode {filepath} with any encoding")
    return ""


def extract_text(filepath):
    """Extract text from any supported file format."""
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return extract_text_pdf(filepath)
    elif ext in (".txt", ".text"):
        return extract_text_txt(filepath)
    elif ext in (".html", ".htm"):
        return extract_text_txt(filepath)  # basic fallback
    else:
        logger.warning(f"Unsupported format: {ext} for {filepath}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

def segment_document(text, doc_id, doc_title, doc_type,
                     target_words=350, min_words=40, max_words=500):
    """
    Segment a document into chunks suitable for embedding.

    Matches the segmentation approach used for your existing 22 docs:
    - Splits on headings and double newlines
    - Target ~350 words per segment (matching your existing median)
    - Falls back to paragraph-level splitting when no headings found
    - Filters out very short segments
    """
    if not text or not text.strip():
        return []

    # Split on likely heading boundaries
    blocks = re.split(r'\n\s*\n', text)

    segments = []
    current_text = ""
    current_heading = f"{doc_title} - Preamble"
    seg_counter = 0

    def flush_segment():
        """Save current_text as a segment if long enough."""
        nonlocal current_text, seg_counter
        if current_text and len(current_text.split()) >= min_words:
            segments.append(_make_segment(
                doc_id, doc_title, doc_type, current_heading,
                current_text, seg_counter
            ))
            seg_counter += 1
        current_text = ""

    def force_split_long_text(long_text, heading):
        """Split oversized text on sentence boundaries near the target size."""
        nonlocal seg_counter
        # Split on sentence boundaries (period + space + capital letter)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z(0-9])', long_text)
        chunk = ""
        sub = 0
        for sent in sentences:
            test = (chunk + " " + sent).strip() if chunk else sent
            if len(test.split()) > max_words and chunk:
                # Save current chunk
                seg_heading = heading if sub == 0 else f"{heading} (Sub {sub})"
                segments.append(_make_segment(
                    doc_id, doc_title, doc_type, seg_heading,
                    chunk.strip(), seg_counter
                ))
                seg_counter += 1
                sub += 1
                chunk = sent
            else:
                chunk = test
        # Leftover
        if chunk and len(chunk.split()) >= min_words:
            seg_heading = heading if sub == 0 else f"{heading} (Sub {sub})"
            segments.append(_make_segment(
                doc_id, doc_title, doc_type, seg_heading,
                chunk.strip(), seg_counter
            ))
            seg_counter += 1

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.split("\n")
        first_line = lines[0].strip()

        # Detect headings: short lines, possibly numbered, possibly uppercase
        is_heading = (
            len(first_line) < 120
            and len(first_line.split()) < 15
            and (
                first_line.isupper()
                or re.match(r"^\d+[\.\)]\s", first_line)
                or re.match(r"^\d+\.\d+", first_line)  # e.g. "3.2 Section Name"
                or re.match(r"^(?:Article|Section|Chapter|Part|Schedule|Annex|Regulation)\s", first_line, re.I)
                or re.match(r"^[IVXLCDM]+[\.\)]\s", first_line)
                or (first_line.isupper() and len(first_line.split()) <= 8)
            )
        )

        if is_heading and current_text:
            # Flush what we have
            word_count = len(current_text.split())
            if word_count > max_words:
                force_split_long_text(current_text, current_heading)
                current_text = ""
            else:
                flush_segment()
            current_heading = first_line
            current_text = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        else:
            # Accumulate text
            if current_text:
                current_text += "\n" + block
            else:
                current_text = block

            # Check if we've hit target — try to split at paragraph boundary
            word_count = len(current_text.split())
            if word_count >= target_words:
                # If well past target, force a split
                if word_count >= max_words:
                    force_split_long_text(current_text, current_heading)
                    current_text = ""
                    current_heading = re.sub(r'\s*\(Sub \d+\)$', '', current_heading)

    # Don't forget the last segment
    if current_text:
        word_count = len(current_text.split())
        if word_count > max_words:
            force_split_long_text(current_text, current_heading)
        elif word_count >= min_words:
            segments.append(_make_segment(
                doc_id, doc_title, doc_type, current_heading,
                current_text, seg_counter
            ))

    return segments


def _make_segment(doc_id, doc_title, doc_type, heading, text, counter):
    """Create a segment dict matching your existing format."""
    return {
        "segment_id": f"{doc_id}_seg_{counter:04d}",
        "doc_id": doc_id,
        "doc_title": doc_title,
        "doc_type": doc_type,
        "heading": heading[:200],
        "content": text.strip(),
        "char_count": len(text.strip()),
        "word_count": len(text.split()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════

# Tri-model stack matching the DAERA <-> Climate+ alignment tool.
# Similarities are computed per model and averaged downstream (matrix
# averaging), so models of different dimensionality combine correctly.
EMBED_MODELS = {
    "bge_m3": "BAAI/bge-m3",                                # 1024d, primary
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",                   # 1024d, second opinion
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",     # 384d, lightweight
}


def generate_embeddings(segments, batch_size=16):
    """Generate embeddings for every model in EMBED_MODELS.

    Returns {model_name: (n_segments, dim) float32 array}. Document-document
    comparison is symmetric, so all text is encoded plainly (no instructions
    or query/passage asymmetry) for every model.
    """
    from sentence_transformers import SentenceTransformer

    texts = [s["content"] for s in segments]
    logger.info(f"Generating embeddings for {len(texts)} segments "
                f"with {len(EMBED_MODELS)} models...")

    out = {}
    for name, model_id in EMBED_MODELS.items():
        logger.info(f"[{name}] loading {model_id}...")
        model = SentenceTransformer(model_id)
        t0 = time.time()
        emb = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        out[name] = np.asarray(emb, dtype=np.float32)
        logger.info(f"[{name}] done in {time.time()-t0:.1f}s — shape: {out[name].shape}")
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    return out


# ═══════════════════════════════════════════════════════════════════════════
# MAPPING READER
# ═══════════════════════════════════════════════════════════════════════════

def read_mapping(mapping_path):
    """Read the reviewed corpus mapping CSV."""
    docs = []
    with open(mapping_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Status", "").strip() == "NEW":
                docs.append({
                    "doc_id": row.get("Doc ID", "").strip(),
                    "clean_title": row.get("Clean Title", "").strip(),
                    "domain": row.get("Domain", "cross_cutting").strip(),
                    "filename": row.get("Filename", "").strip(),
                    "subfolder": row.get("Subfolder", "").strip(),
                    "format": row.get("Format", "").strip(),
                })
    return docs


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Ingest new policy documents")
    parser.add_argument("--mapping", required=True,
                        help="Reviewed corpus mapping CSV")
    parser.add_argument("--docs-dir", required=True,
                        help="Root folder containing policy documents")
    parser.add_argument("--output-dir", default="./ingested",
                        help="Output directory for processed data")
    parser.add_argument("--existing-segments", default="",
                        help="Path to existing all_segments.json (for merging)")
    parser.add_argument("--existing-emb", action="append", default=[],
                        help="Existing embeddings to merge with, as name=path "
                             "(repeatable), e.g. --existing-emb bge_m3=ingested_v2/embeddings_bge_m3.npy. "
                             "Names must match EMBED_MODELS.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Embedding batch size (reduce if out of memory)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding generation (just extract and segment)")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Read mapping ──────────────────────────────────────────
    logger.info("Step 1: Reading reviewed mapping...")
    new_docs = read_mapping(args.mapping)
    logger.info(f"  {len(new_docs)} new documents to process")

    # ── Step 2: Extract text ──────────────────────────────────────────
    logger.info("Step 2: Extracting text from documents...")
    all_new_segments = []
    errors = []
    doc_stats = []

    for i, doc in enumerate(new_docs):
        # Find the file
        if doc["subfolder"]:
            filepath = docs_dir / doc["subfolder"] / doc["filename"]
        else:
            filepath = docs_dir / doc["filename"]

        if not filepath.exists():
            # Try case-insensitive search
            found = False
            search_dir = docs_dir / doc["subfolder"] if doc["subfolder"] else docs_dir
            if search_dir.exists():
                for f in search_dir.iterdir():
                    if f.name.lower() == doc["filename"].lower():
                        filepath = f
                        found = True
                        break
            if not found:
                errors.append({"doc_id": doc["doc_id"], "error": f"File not found: {filepath}"})
                logger.warning(f"  [{i+1}/{len(new_docs)}] NOT FOUND: {filepath}")
                continue

        logger.info(f"  [{i+1}/{len(new_docs)}] {doc['doc_id']}: {doc['filename'][:50]}")

        # Extract text
        text = extract_text(filepath)
        if not text or len(text.strip()) < 100:
            errors.append({"doc_id": doc["doc_id"], "error": f"No text extracted ({len(text)} chars)"})
            logger.warning(f"    No text extracted!")
            continue

        # Segment
        segments = segment_document(
            text,
            doc_id=doc["doc_id"],
            doc_title=doc["clean_title"],
            doc_type=doc["domain"],
        )

        if not segments:
            errors.append({"doc_id": doc["doc_id"], "error": "No segments produced"})
            logger.warning(f"    No segments produced!")
            continue

        all_new_segments.extend(segments)
        doc_stats.append({
            "doc_id": doc["doc_id"],
            "title": doc["clean_title"],
            "domain": doc["domain"],
            "chars_extracted": len(text),
            "segments": len(segments),
            "avg_words": sum(s["word_count"] for s in segments) // max(len(segments), 1),
        })
        logger.info(f"    {len(text):,} chars → {len(segments)} segments")

    logger.info(f"\nExtraction complete: {len(all_new_segments)} segments from {len(doc_stats)} docs")
    if errors:
        logger.warning(f"  {len(errors)} documents had errors")

    # Save new segments
    with open(output_dir / "new_segments.json", "w", encoding="utf-8") as f:
        json.dump(all_new_segments, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved new_segments.json ({len(all_new_segments)} segments)")

    # ── Step 3: Generate embeddings ───────────────────────────────────
    if not args.skip_embeddings and all_new_segments:
        logger.info("Step 3: Generating tri-model embeddings...")
        new_embs = generate_embeddings(all_new_segments, batch_size=args.batch_size)
        for name, arr in new_embs.items():
            np.save(str(output_dir / f"new_embeddings_{name}.npy"), arr)
        logger.info("Saved new embeddings: " +
                    ", ".join(f"{n} {a.shape}" for n, a in new_embs.items()))
    else:
        new_embs = None
        if args.skip_embeddings:
            logger.info("Step 3: Skipping embeddings (--skip-embeddings flag)")

    # ── Step 4: Merge with existing ───────────────────────────────────
    if args.existing_segments and Path(args.existing_segments).exists():
        logger.info("Step 4: Merging with existing data...")

        with open(args.existing_segments, encoding="utf-8") as f:
            existing_segments = json.load(f)

        merged_segments = existing_segments + all_new_segments
        with open(output_dir / "merged_segments.json", "w", encoding="utf-8") as f:
            json.dump(merged_segments, f, indent=2, ensure_ascii=False)
        logger.info(f"Merged segments: {len(existing_segments)} + {len(all_new_segments)} = {len(merged_segments)}")

        if new_embs is not None:
            for spec in args.existing_emb:
                if "=" not in spec:
                    logger.warning(f"--existing-emb expects name=path, got: {spec}")
                    continue
                name, path = (x.strip() for x in spec.split("=", 1))
                if name not in new_embs:
                    logger.warning(f"No new embeddings for model '{name}' — skipping merge")
                    continue
                if not Path(path).exists():
                    logger.warning(f"Existing embeddings not found: {path}")
                    continue
                existing = np.load(path)
                merged = np.vstack([existing, new_embs[name]])
                np.save(str(output_dir / f"merged_{name}.npy"), merged)
                logger.info(f"Merged {name}: {existing.shape} + {new_embs[name].shape} → {merged.shape}")
    else:
        logger.info("Step 4: No existing data provided — skipping merge")

    # ── Step 5: Report ────────────────────────────────────────────────
    report = {
        "total_new_docs_in_mapping": len(new_docs),
        "docs_processed": len(doc_stats),
        "docs_with_errors": len(errors),
        "total_new_segments": len(all_new_segments),
        "embeddings_generated": new_embs is not None,
        "domain_breakdown": dict(defaultdict(int)),
        "doc_stats": doc_stats,
        "errors": errors,
    }

    domain_counts = defaultdict(int)
    domain_segments = defaultdict(int)
    for ds in doc_stats:
        domain_counts[ds["domain"]] += 1
        domain_segments[ds["domain"]] += ds["segments"]
    report["domain_breakdown"] = {
        d: {"docs": domain_counts[d], "segments": domain_segments[d]}
        for d in sorted(domain_counts.keys())
    }

    with open(output_dir / "ingestion_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("INGESTION COMPLETE")
    print("=" * 65)
    print(f"Documents processed:   {len(doc_stats)}/{len(new_docs)}")
    print(f"Errors:                {len(errors)}")
    print(f"New segments:          {len(all_new_segments)}")
    if new_embs is not None:
        for n, a in new_embs.items():
            print(f"{n} embeddings:      {a.shape}")
    print(f"\nDomain breakdown:")
    for d in sorted(domain_counts.keys()):
        print(f"  {d:15s}: {domain_counts[d]:3d} docs, {domain_segments[d]:5d} segments")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  {e['doc_id']:35s}: {e['error']}")
    print(f"\nOutputs in: {output_dir}/")

    if args.existing_segments:
        print(f"\nTo rebuild the graph with merged data, run:")
        print(f"  python scripts/build_explorer.py \\")
        print(f"    --segments {output_dir}/merged_segments.json \\")
        emb_flags = " ".join(f"--emb {n}={output_dir}/merged_{n}.npy" for n in EMBED_MODELS)
        print(f"    {emb_flags} \\")
        print(f"    --output policy_graph.html")


if __name__ == "__main__":
    main()