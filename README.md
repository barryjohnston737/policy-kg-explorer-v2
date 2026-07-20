# Policy Knowledge Graph Explorer — v2 (191-document corpus)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21459285.svg)](https://doi.org/10.5281/zenodo.21459285)

An interactive knowledge-graph tool that maps the semantic relationships across a corpus of
**191 Irish, EU, and global environmental-governance documents** — national acts and plans,
EU directives and regulations, international conventions, and a condensed local-authority
tier — spanning climate, water, biodiversity, agriculture, forestry, and cross-cutting
policy domains.

This is the **expanded second version**. The original 95-document demonstrator remains
available in its own repository; v2 doubles the corpus and adds the unified document library
that underpins it. Both are kept for comparison.

Documents are segmented into ~350-word passages, embedded with a **tri-model ensemble**
(BGE-M3 + Qwen3-Embedding-0.6B + MiniLM, per-model similarity matrices averaged), and
assembled into a document-level similarity graph. The output is a single standalone HTML
explorer with three linked views — **2D network**, **3D orbit**, and **similarity heatmap** —
plus a **reference-datum mode** that positions and ranks every document relative to any
chosen framework document.

Connection strength is classified into **weak / moderate / strong bands** derived from the
corpus's own background similarity distribution (μ+1σ / +1.5σ / +2σ), recalibrated
automatically on every build.

> **The policy web.** Every line is measured thematic overlap between two documents. The
> tangle is not noise — it is the finding: environmental governance is deeply interconnected,
> and no document stands alone. The explorer's filters, strength bands, and datum mode exist
> to untangle it one question at a time.

> **Status:** research / decision-support tool. Similarity measures thematic overlap, not
> agreement or legal consistency. See the
> [technical specification](docs/Technical_Specification.md) for method detail, interpretation
> guidance, and limitations, and the [policy brief](docs/Policy_Brief.md) for a plain-language
> summary for policy audiences. The formatted, citable PDF of the brief is deposited
> separately on Zenodo (DOI below).

**Developed by B. Johnston & J. Moran — Atlantic Technological University.**
Licensed under the [MIT License](LICENSE). 

## Citation

**Software:**
Johnston, B. & Moran, J. (2026). *Policy Knowledge Graph Explorer: a multi-model semantic
knowledge graph for exploring environmental governance corpora* (v2.0.2). Zenodo.
https://doi.org/10.5281/zenodo.21459285

**Policy brief:**
Johnston, B. & Moran, J. (2026). *Mapping the Policy Web: a semantic knowledge-graph tool for
navigating Ireland's environmental governance*. Zenodo.
https://doi.org/10.5281/zenodo.YYYYYYY

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

---

## Corpus at a glance (v2)

| Domain | Documents |
|---|---|
| Water | 53 |
| Cross-cutting | 52 |
| Climate | 40 |
| Biodiversity | 30 |
| Agriculture | 14 |
| Forestry | 2 |
| **Total** | **191** |

13,871 segments · ~5.07M words · tri-model embeddings. Strength cutoffs for this corpus:
weak ≥ 0.68, moderate ≥ 0.74, strong ≥ 0.80 (from 18,145 document pairs).

The 28 local-authority climate action plans, 5 local biodiversity plans, and 4 county
development plans are each condensed into a single **sampled composite node** so a
near-duplicate family does not dominate the view or skew the calibration; Dublin City's plan
is kept individually as a named exemplar.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/build_explorer.py` | Graph builder + HTML explorer generator |
| `scripts/build_final.sh` | One-command rebuild with this corpus's settings |
| `scripts/embed_final.py` | Tri-model embedding of the assembled segments |
| `scripts/ingest_corpus.py` | Text extraction and ~350-word segmentation |
| `scripts/scan_corpus.py` | Corpus inventory and text-quality checks |
| `scripts/gen_corpus_scope.py` | Derives `corpus_scope.csv` from the master library |
| `scripts/merge_libraries.py`, `sync_from_csv.py` | Library assembly and round-tripping |
| `scripts/gen_download_batch.py` | Builds fetch lists for documents still needing text |
| `requirements.txt` | Python dependencies, grouped by which task needs them |
| `master_library.csv` | Unified document library (300 documents, provenance, quality flags) |
| `corpus_scope.csv` | Which library documents are in this build, and why |
| `corpus_build/final_segments.json.gz` | The 13,871 assembled segments (gzipped, ~10 MB) |
| `output/` / `docs/index.html` | Built explorer HTML (also served via GitHub Pages / Vercel) |
| `docs/` | Technical specification, policy brief, and screenshots (Markdown + web; formatted PDFs live on Zenodo) |

Embeddings (`.npy`) are large and regenerable; they are not committed. The segment
corpus is committed gzipped — the scripts decompress it automatically on first use.

## Rebuilding

From a fresh clone, two commands:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. embed the segments — tri-model, ~1–2 h and ~3.5 GB of model weights on
#    the first run; completed models are skipped if you need to restart
.venv/bin/python scripts/embed_final.py

# 2. build the explorer into output/ and refresh docs/index.html
PYTHON=.venv/bin/python bash scripts/build_final.sh
```

`PYTHON` defaults to `python3` if unset. Both scripts resolve paths relative to
the repository, so neither needs editing, and the gzipped corpus is decompressed
automatically on first use.

If you already have the `.npy` embeddings and only want to rebuild the graph,
`numpy` and `networkx` alone are enough — step 2 takes about 15 seconds.

Rebuilding the corpus from source documents (rather than the committed segments)
additionally needs the source PDFs and the external text directories referenced
by `gen_corpus_scope.py`; see `corpus_scope.csv` for where each document's text
came from.

## Method in one paragraph

Every document is split into ~350-word segments along its natural structure. Three
independent embedding models fingerprint every segment; a document's fingerprint per model
is the mean of its segment embeddings. For each document pair, cosine similarity is computed
per model and averaged (matrix averaging — models of different dimensionality are never
vector-averaged). Pairs above the edge threshold become graph edges; the strongest
cross-document segment pairs are recorded as paragraph-level links; explicit EU
regulation/directive citations become cross-reference edges. Strength bands are z-score
cut-offs on the corpus's own pair-similarity distribution.

## Known limitations (v2)

- **Thematic similarity only** — measures shared subject matter, not agreement, contradiction,
  or legal consistency.
- **Four thin nodes** (Arterial Drainage Act, National Monuments Act, Coast Protection Act,
  Monkfish Regulations) are scanned image-PDFs whose full text was not recovered before this
  build; they appear weakly connected. Replacements are queued for the next corpus refresh.
- **~49 documents deferred** — publisher sites (gov.ie, NPWS) block automated download; these
  are recorded in `corpus_scope.csv` for a future enrichment pass.
- **Calibration is corpus-specific** — bands describe what is unusual *for this corpus* and
  are not comparable across corpora or absolute probabilities of relatedness.
