# Irish/EU Policy Knowledge Graph Explorer — Technical Specification

**Status:** Research / decision-support tool  |  **Last updated:** July 2026
**B. Johnston & J. Moran — Atlantic Technological University**

> **Document purpose:** This specification describes the knowledge-graph tool that maps the
> semantic relationships across a corpus of 191 Irish, EU and international policy, legislative, and strategy
> documents spanning climate, water, biodiversity, agriculture, forestry, and cross-cutting
> domains. The opening sections and interpretation guidance are written for users with no
> computing or data-science background. Later sections give analysts the detail required to
> run, modify, or audit the pipeline.
>
> **Status:** Research and decision-support tool. Outputs are a structured prompt for human
> review, not an automated assessment of policy coherence or consistency, and not a
> substitute for expert judgement.

This specification describes **version 2** of the tool (191-document corpus).

---

## Summary of appropriate use

The explorer is a discovery and triage instrument. Used well, it rapidly surfaces which
documents in a large policy library share substantial thematic ground, which documents act
as bridges between policy areas, and where a new document (such as the Land Use Review) sits
relative to the existing landscape — work that would be slow and inconsistent to do by hand.
Used poorly, as a scoreboard, a measure of policy alignment or legal consistency, or a
source of precise figures, it manufactures false confidence.

The single guiding principle: **the score starts the conversation; it does not end it.** A
strong connection means two documents talk about the same things in similar language — it
does not mean they agree, are mutually consistent, or are legally compatible. Every
consequential reading of these outputs should be confirmed against the underlying document
text by someone with domain expertise.

## 1. What this tool does — a plain-language overview

### The problem it solves

Irish environmental governance is spread across a large and growing library of documents:
national acts and plans, EU directives and regulations, sectoral strategies, and
international conventions. Understanding how these documents relate — which cover the same
ground, which depend on each other, where the connective tissue between policy areas lies —
is essential for coherent policy-making, but reading and cross-referencing 191 documents by
hand is impractical.

The tool reads every document, splits it into passages, converts each passage into a
numerical "fingerprint" of its meaning using pre-trained language models, and then measures
how closely every pair of documents matches in that meaning-space. The result is a
similarity score for every document pair, sorted into strength bands and presented through a
set of linked interactive views.

This does not replace expert reading; it tells you which relationships are most worth
examining. Think of it as an intelligent first pass that produces a prioritised map of the
policy landscape, not a set of conclusions.

### The corpus

The current corpus is 191 documents assigned to six policy domains:

| Domain | Documents | Examples |
|---|---|---|
| Water | 53 | Water Framework Directive, Water Action Plan 2024, Maritime Area Planning Act |
| Cross-cutting | 52 | A Living Land (Land Use Review), EPA State of the Environment, National Planning Framework |
| Climate | 40 | Climate Action Plans 2024/2025, Climate Act 2021, European Climate Law |
| Biodiversity | 30 | Nature Restoration Law, National Biodiversity Action Plan, Pollinator Plan |
| Agriculture | 14 | Food Vision 2030, CAP Strategic Plan, Good Agricultural Practice Regulations |
| Forestry | 2 | Forest Strategy 2023–2030, Forestry Act 2014 |

Each document is split into passages of roughly 350 words along its natural structure
(headings, sections, paragraphs), giving **13,871 segments** in total. The 28 local-authority climate action plans, 5 local biodiversity plans and 4 county development plans are each condensed into a single sampled composite node so that a near-duplicate family does not dominate the view or skew the calibration; Dublin City's plan is retained individually as a named exemplar. The segment, not the
whole document, is the unit that the language models read; document-level relationships are
then assembled from the segment fingerprints. The corpus inventory is maintained in a
mapping spreadsheet (`corpus_mapping.xlsx.csv`); source URLs for every document are listed
in the [document library](DOCUMENT_LIBRARY.md).

### What the outputs look like

The pipeline produces a single interactive web page (a standalone HTML file) with three
linked views and a reference-datum mode:

- **2D network.** Documents as circles, connections as lines. Line colour shows connection
  strength (green = strong, amber = moderate, blue-grey = weak); each circle's outer ring
  shows the strength of that document's strongest connection, while its fill colour shows
  the policy domain.
- **3D orbit.** The same graph in three dimensions, useful for seeing cluster structure that
  a flat layout has to squash. Only the most important nodes are labelled to keep the view
  readable; every node reveals its name on hover.
- **Heatmap.** The full 191 × 191 matrix of document similarities, grouped by domain, for
  systematic scanning.
- **Datum mode.** A dropdown sets any document as the fixed reference point ("datum"). The
  views become radial: the datum sits at the centre and every other document is placed by
  its similarity to it, with a ranked list replacing the heatmap. A curated shortlist of ten
  framework documents (one per policy lever) is offered first, but any of the 191 documents
  can be the datum.

All views share one set of controls: a search box, similarity range sliders, strength-band
shortcuts, domain filters, a light/dark theme toggle, and a detail panel showing the
selected document's profile and closest neighbours.

**A note on first impressions.** Viewed from a distance, the network is dense to the point
of tangle. That is not a rendering flaw — it is the central empirical observation: Irish
environmental governance is deeply interconnected, and no document stands alone. The 3D
view labels this deliberately ("The policy web"), and the interface's filters, bands, and
datum mode exist precisely to untangle it one question at a time.

## 2. Data security and local processing

**All analysis is local.** The pipeline runs entirely on the local machine. No document
text, no embeddings, and no similarity results are sent to any external server, cloud
service, or third party at any point. The only network activity during analysis is a
one-time download of the embedding model files from Hugging Face the first time each model
is used; once downloaded, the models are cached locally and subsequent runs need no internet
connection.

**One caveat for the visual output.** The HTML explorer embeds all of its data inside the
file itself — nothing is uploaded when it is opened. However, it loads its charting
libraries (D3.js, three.js, 3d-force-graph) and fonts from public content-delivery networks,
so an internet connection is required to display the visualisations. No corpus data travels
in either direction over these connections.

### What is stored on disk

| File type | What it contains | Sensitive? |
|---|---|---|
| `.json`, `.csv`, `.graphml` | Document titles, segment text (in the segments file), similarity scores, graph structure | As sensitive as the source documents |
| `.html` | Interactive explorer; document titles, similarity data, and segment headings embedded in the file | As sensitive as the source documents |
| `.npy` (NumPy binary) | The numerical embedding vectors derived from the text | Derived data; does not contain readable text, but could in principle be matched to source text by a party who also holds the source documents |

The `.npy` files are intermediate working files that let the pipeline skip the
computationally expensive embedding step when the text has not changed. If no longer needed
they can be deleted safely; the pipeline will regenerate them on the next run. All source
documents in this corpus are public policy documents, so sensitivity is low, but the same
handling rules apply if the corpus is ever extended with non-public material.

## 3. Output files

| File | Description |
|---|---|
| `policy_graph_191docs.html` | The complete interactive explorer: 2D / 3D / heatmap views, datum mode, all controls. Standalone; all data embedded |
| `policy_graph_191docs_data.json` | Full machine-readable graph: every node (documents, sections, clusters) and edge with per-model similarity scores |
| `policy_graph_191docs.graphml` | The same graph in GraphML for import into Gephi, Cytoscape, or Neo4j |
| `policy_graph_191docs_similarity.csv` | The full 191 × 191 document similarity matrix |
| `ingested_v2/merged_segments.json` | Every segment with its text, heading, word count, and parent document |
| `ingested_v2/embeddings_*.npy` | Cached embedding matrices, one per model |
| `ingested_v2/ingestion_report.json` | Per-document extraction statistics and errors from the most recent ingestion |

## 4. How it works — method detail

### Step 1: Corpus assembly and text extraction

Documents (PDF, TXT, HTML) live in `raw_documents/` and are inventoried in
`corpus_mapping.xlsx.csv`. Rows marked `NEW` are picked up by the ingestion script
(`ingest_corpus.py`), which extracts text with pdfplumber (PyPDF2 as fallback). Scanned
image-only PDFs yield no text and are reported as errors rather than silently skipped. Rows
are marked `INGESTED` once processed, so re-runs never duplicate work.

### Step 2: Segmentation

Each document is split on headings and paragraph boundaries into segments targeting ~350
words (minimum 40, maximum 500, with sentence-boundary splitting for oversized blocks).
Headings are detected heuristically (numbered sections, "Article"/"Section"/"Schedule"
patterns, all-caps lines). Each segment carries its parent document, domain, heading, and
word count. Only the segment text itself is embedded; titles and domains are metadata used
for display and grouping.

### Step 3: Embedding with three models

Every segment is read by three independent embedding models (Section 5). Each model returns
a fixed-length list of numbers — an "embedding" — that encodes where the text sits in that
model's understanding of meaning.

**Symmetric encoding.** Unlike a search task (query vs. passage), document-to-document
comparison is symmetric: both sides are the same kind of text. All text is therefore encoded
plainly, with no instructions or query/passage asymmetry, for every model. Sequences are
capped at 1,024 tokens (segments average ~500), which bounds memory without losing content.

**Combining models of different sizes.** Because the models return vectors of different
lengths (1024 vs. 384 dimensions), the vectors are never averaged directly. Instead, each
model produces its own complete similarity matrix and those matrices are averaged. This
matrix-averaging approach is the correct way to combine heterogeneous embedding models and
matches the method used in the companion DAERA ↔ Climate+ alignment tool.

### Step 4: Document-level similarity

A document's fingerprint under each model is the average of its segments' embeddings
(mean-pooling). For every pair of documents, cosine similarity is computed per model and the
per-model scores are averaged, giving one score per pair — 18,145 pairs for 191 documents.

### Step 5: Graph construction

- **Document edges.** Pairs with an averaged similarity of at least 0.60 become graph edges,
  each recording its per-model scores and whether all models independently exceed the
  threshold.
- **Segment links.** The 300 most similar cross-document segment pairs (score ≥ 0.78) are
  recorded, revealing paragraph-level textual overlap — shared provisions or language rather
  than broad thematic likeness.
- **Cross-references.** Explicit citations of EU regulations and directives in the text
  (e.g. "Regulation (EU) 2024/1991") are detected by pattern-matching and linked to the
  corresponding document where it is in the corpus.
- **Centrality.** Betweenness centrality is precomputed at nine threshold levels so the
  "bridging role" node-sizing option responds instantly to the sliders.

## 5. The embedding models

| Model | Dimensions | Context window | Role |
|---|---|---|---|
| BGE-M3 (`BAAI/bge-m3`) | 1024 | 8,192 tokens | Primary. Multi-granularity retrieval model; handles long policy text well |
| Qwen3-Embedding-0.6B (`Qwen/Qwen3-Embedding-0.6B`) | 1024 | 32k tokens | Second opinion from an independent model family |
| MiniLM (`all-MiniLM-L6-v2`) | 384 | 256 tokens | Lightweight, fast, general-purpose third signal |

This is the same three-model stack used by the DAERA ↔ Climate+ alignment tool, which keeps
the two instruments methodologically consistent and directly comparable. Using several
models protects against any single model's quirks: a connection that all three models rate
highly is much less likely to be an artefact of one model's training data. Segments longer
than a model's context window are truncated for that model — in practice this mainly affects
MiniLM, whose 256-token window sees roughly the first two-thirds of a 350-word segment.

Embeddings are computed once per model and cached as `.npy` files; re-runs skip any model
whose cache matches the current segment count.

## 6. Connection-strength bands

Raw similarity scores are hard to interpret in isolation — whether 0.78 is impressive
depends entirely on the corpus. The explorer therefore classifies every score against the
background distribution of all 18,145 document-pair similarities in this corpus:

| Band | Cut-off | What it means in practice |
|---|---|---|
| **Strong** | mean + 2σ (≈ top 2% of pairs) | Exceptional thematic overlap for this corpus. Worth prioritising for expert review |
| **Moderate** | mean + 1.5σ (≈ top 6%) | Clearly above-background connection. Merits investigation |
| **Weak** | mean + 1σ (≈ top 16%) | Above-average connection. Context and hypothesis generation only |

The exact numeric cut-offs are recomputed automatically on every build from the current
corpus, printed in the build log, and displayed live in the explorer's sidebar together with
the distribution statistics (μ, σ, number of pairs). They will shift when documents are
added or the model stack changes — this is intended behaviour, not drift.

**How this differs from the DAERA tool's calibration.** The DAERA tool calibrates its tiers
against a null distribution of pairs expected to be unrelated (cross-theme need pairs),
which yields approximate false-positive rates. That approach is unavailable here: this
corpus is a single, densely interrelated policy domain with no defensible pool of
"known-unrelated" document pairs. The bands are therefore relative statements — "unusually
similar for this corpus" — rather than absolute probabilities of relatedness. They rank and
triage; they do not certify.

## 7. Setup and operation

### 7.1 Requirements

- Python 3.10+
- numpy, networkx, pdfplumber
- sentence-transformers, torch, transformers (embedding runs only)

```bash
pip install -r requirements.txt
```

The first embedding run downloads the models from Hugging Face (BGE-M3 ≈ 2.2 GB, Qwen3
≈ 1.2 GB, MiniLM ≈ 80 MB). A GPU is helpful but not required; on Apple Silicon the models
use the built-in GPU automatically.

### 7.2 Adding a new document

1. Place the file in `raw_documents/`.
2. Add a row to `corpus_mapping.xlsx.csv` with Status `NEW`, a unique Doc ID, a clean title,
   and a domain. Add its source URL to the [document library](DOCUMENT_LIBRARY.md).
3. Run the ingestion script, pointing at the existing merged data:

```bash
python ingest_corpus.py --mapping corpus_mapping.xlsx.csv --docs-dir raw_documents \
    --output-dir ingested_v3 --existing-segments ingested_v2/merged_segments.json \
    --existing-emb bge_m3=ingested_v2/embeddings_bge_m3.npy \
    --existing-emb qwen3=ingested_v2/embeddings_qwen3.npy \
    --existing-emb minilm=ingested_v2/merged_minilm.npy
```

4. Rebuild the explorer (7.4). Strength bands recalibrate automatically.

### 7.3 Re-embedding the whole corpus

If the model stack changes, `embed_corpus_3models.py` re-embeds every segment with the
configured models, skipping any model whose cache is already complete.

### 7.4 Building the explorer

```bash
python policy_kg_explorer_2_3d_new_3.py \
    --segments ingested_v2/merged_segments.json \
    --emb bge_m3=ingested_v2/embeddings_bge_m3.npy \
    --emb qwen3=ingested_v2/embeddings_qwen3.npy \
    --emb minilm=ingested_v2/merged_minilm.npy \
    --output policy_graph_191docs.html
```

| Flag | Default | Description |
|---|---|---|
| `--emb name=path` | — | Embedding model and file; repeat once per model. Similarities are averaged across all models given |
| `--doc-threshold` | 0.60 | Minimum averaged similarity for a document edge |
| `--seg-threshold` | 0.78 | Minimum similarity for a segment-level link |
| `--top-k-segments` | 300 | Maximum number of segment links recorded |
| `--datum-options` | built-in list | Comma-separated Doc IDs for the curated datum shortlist |
| `--skip-exports` | off | Write the HTML only (skip JSON/GraphML/CSV) |

## 8. How to interpret the results

**This is the most important section of the document.**

### 8.1 The shared controls

- Use the **search box** to find a document by name; non-matching nodes dim, and picking a
  suggestion selects the document across every view.
- Click a **strength band** (each shows its live count) to snap the similarity sliders to
  exactly that band; click again to reset. The Min/Max sliders can also be set manually.
- Click a **domain chip** to isolate one policy domain; **Clear filters** resets everything.
- **Visual encoding:** a node's fill colour is its policy domain; its outer ring is the
  strength band of its best connection in the current range (grey ring = nothing in range).
  Line colours use the same strength scheme.
- **Node sizing** (2D/3D) can reflect document size, connection count, bridging role
  (betweenness), paragraph-level overlap, or total connection strength — each option is
  explained in the sidebar.

### 8.2 The three views

**2D network** is the primary reading view: clusters, bridges, and outliers are all visible
at once. Selecting a node highlights its neighbourhood and lists its closest documents with
scores. **3D orbit** shows the same structure with an extra dimension; distant nodes fade
with depth, and only the most important nodes carry labels (hover for any name). A
persistent caption in this view — "The policy web" — states the intended reading directly:
every line is measured thematic overlap, and the tangle itself is the finding. **Heatmap**
shows every pair, including those below the edge threshold; it is the best view for
systematic scanning — a bright block where two domains meet indicates a substantial
inter-domain interface, while a dark row flags a document with little in common with
anything.

### 8.3 Datum mode

Datum mode answers the question "where does everything stand relative to this document?".
The datum is pinned at the centre; every other document's distance from it reflects its
similarity, with dashed guide rings marking the band cut-offs. The Ranking view lists all 94
other documents in order. This is the mode to use when assessing a new publication (e.g. the
Land Use Review) against the existing landscape, or checking which national instruments sit
closest to an EU obligation (e.g. the Nature Restoration Law).

### 8.4 Reading rules

1. **Start at the band, not the decimal.** Treat Strong / Moderate / Weak as the unit of
   interpretation. Three-decimal scores imply a precision the method does not have.
2. **Similarity is not agreement.** A strong connection means two documents discuss the same
   subject matter in similar language. They may reinforce each other, duplicate each other,
   or contradict each other — the score cannot tell the difference. Only reading can.
3. **Use it to generate hypotheses, then verify.** Every consequential connection should be
   confirmed by reading the underlying passages; the segment-level links in the detail panel
   point to where to look.
4. **Look at patterns, not single scores.** A document that connects moderately to a whole
   cluster is a more robust signal than one isolated high score, which may reflect a wording
   coincidence.
5. **Isolation is informative.** A document with few connections above Weak occupies ground
   the rest of the corpus does not cover — which may be its purpose or may flag an
   integration gap worth examining.
6. **Mind the document-size effect.** Long, wide-ranging documents naturally connect to many
   things because they mention many things. The bridging-role and paragraph-overlap sizing
   options help separate genuine connective documents from merely long ones.

## 9. Limitations

**Method-level**

- **Thematic similarity only.** The tool measures overlap of subject matter and language. It
  does not detect agreement, contradiction, legal dependency, or policy coherence.
- **Band calibration is corpus-specific and relative.** The cut-offs describe what is
  unusual for this corpus; they are not false-positive rates and are not comparable across
  corpora (Section 6).
- **Mean-pooling dilutes long documents.** A long document with one highly relevant chapter
  may score lower against a target than the chapter itself would. The segment-level links
  partially compensate.

**Model-level**

- **Training-data blind spots.** Models inherit the coverage gaps of their training data;
  specialised Irish policy terminology or very recent instruments may be poorly represented.
- **Context-window truncation.** MiniLM (256 tokens) sees only the opening portion of longer
  segments.
- **Model agreement is not correctness.** Averaging reduces single-model quirks but not
  shared biases.

**Process-level**

- **Extraction quality varies.** Scanned or heavily formatted PDFs can yield noisy text (one
  document, the MARPOL convention, could not be extracted at all and is excluded).
  Segmentation heuristics occasionally split or merge sections imperfectly.
- **The tool sees only what is in the corpus.** A missing connection may mean a missing
  document, not a missing relationship.
- **Snapshot in time.** Outputs reflect the corpus at the moment of the run and should be
  regenerated when documents are added or updated.
- **No human relevance feedback.** Scores are unsupervised; the tool has never been told
  which connections experts consider genuinely important.

## 10. Glossary

| Term | Plain-language meaning |
|---|---|
| Embedding | A fixed-length list of numbers produced by a language model to represent the meaning of a piece of text |
| Cosine similarity | A measure (here effectively 0–1) of how closely two embeddings point in the same direction — how similar the texts are in meaning |
| Segment | A passage of roughly 350 words, the unit of text given to the models |
| Mean-pooling | Averaging all of a document's segment embeddings to make one document-level fingerprint |
| Matrix averaging | Combining several models by averaging their similarity scores rather than their raw vectors |
| Strength band | The Weak / Moderate / Strong label given to a score based on how unusual it is against this corpus's background distribution |
| Background distribution | The similarity scores of all 18,145 document pairs, used to define what "unusually similar" means here |
| Betweenness centrality | How often a document sits on the shortest path between other documents — high values mark bridges between policy areas |
| Datum | A document chosen as the fixed reference point; all others are positioned and ranked by similarity to it |
| Segment link | A pair of passages from two documents with very high similarity — evidence of shared language, not just shared topic |
| `.npy` file | A binary file storing a numerical array, created by the NumPy library |
| GraphML | A standard graph file format, readable by Gephi, Cytoscape, etc. |
