#!/usr/bin/env python3
"""
policy_kg_explorer.py — Build a seed knowledge graph from policy segments
and generate an interactive HTML explorer.

Usage:
    python policy_kg_explorer.py \
        --segments all_segments.json \
        --bge embeddings_bge.npy \
        --minilm embeddings_minilm.npy \
        --output policy_graph.html \
        --doc-threshold 0.60 \
        --seg-threshold 0.78 \
        --top-k-segments 300

Outputs:
    - policy_graph.html        Interactive D3 force-directed graph (open in browser)
    - seed_graph.json          Raw graph data (nodes + edges)
    - doc_similarity_matrix.csv  Full 22x22 document similarity matrix
    - seed_graph.graphml       GraphML for Gephi/Neo4j import
"""

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

try:
    import networkx as nx
except ImportError:
    print("Installing networkx...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "networkx", "-q"])
    import networkx as nx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════

class PolicyGraphBuilder:
    """Builds a seed knowledge graph from segment embeddings."""

    def __init__(
        self,
        segments: list[dict],
        embeddings: dict,
        doc_threshold: float = 0.60,
        seg_threshold: float = 0.78,
        top_k_segments: int = 300,
        min_words: int = 40,
    ):
        self.segments = segments
        self.embeddings = dict(embeddings)  # {model_name: (n_segments, dim) array}
        for name, arr in self.embeddings.items():
            assert arr.shape[0] == len(segments), (
                f"embedding '{name}' has {arr.shape[0]} rows but there are {len(segments)} segments")
        self.doc_threshold = doc_threshold
        self.seg_threshold = seg_threshold
        self.top_k = top_k_segments
        self.min_words = min_words
        self.G = nx.DiGraph()

    def build(self) -> nx.DiGraph:
        """Run the full pipeline."""
        logger.info(f"Building graph from {len(self.segments)} segments")
        self._create_document_nodes()
        self._create_section_nodes()
        self._create_doc_type_clusters()
        self._compute_doc_similarity()
        self._compute_segment_similarity()
        self._extract_cross_references()
        self._print_summary()
        return self.G

    # ── Documents ───────────────────────────────────────────────────────

    def _create_document_nodes(self):
        self._docs = {}
        for seg in self.segments:
            did = seg["doc_id"]
            if did not in self._docs:
                self._docs[did] = {
                    "title": seg["doc_title"],
                    "doc_type": seg["doc_type"],
                    "segments": 0, "words": 0, "chars": 0,
                }
            self._docs[did]["segments"] += 1
            self._docs[did]["words"] += seg.get("word_count", 0)
            self._docs[did]["chars"] += seg.get("char_count", 0)

        for did, info in self._docs.items():
            self.G.add_node(f"doc:{did}", node_type="document", **info)
        logger.info(f"  {len(self._docs)} document nodes")

    # ── Sections ────────────────────────────────────────────────────────

    def _create_section_nodes(self):
        count = 0
        for i, seg in enumerate(self.segments):
            if seg["word_count"] < self.min_words:
                continue
            sid = f"seg:{seg['segment_id']}"
            self.G.add_node(
                sid, node_type="section", doc_id=seg["doc_id"],
                heading=seg["heading"], word_count=seg["word_count"],
                content_preview=seg["content"][:300], segment_index=i,
            )
            self.G.add_edge(f"doc:{seg['doc_id']}", sid, edge_type="contains")
            count += 1
        self._n_sections = count
        logger.info(f"  {count} section nodes ({len(self.segments)-count} filtered)")

    # ── Clusters ────────────────────────────────────────────────────────

    def _create_doc_type_clusters(self):
        types = defaultdict(list)
        for did, info in self._docs.items():
            types[info["doc_type"]].append(did)
        for dtype, dids in types.items():
            cid = f"cluster:{dtype}"
            self.G.add_node(cid, node_type="cluster", label=dtype, doc_count=len(dids))
            for did in dids:
                self.G.add_edge(f"doc:{did}", cid, edge_type="belongs_to_cluster")
        logger.info(f"  {len(types)} cluster nodes")

    # ── Document similarity ─────────────────────────────────────────────

    def _compute_doc_similarity(self):
        # Mean-pool segment embeddings per document, per model.
        # Models of different dimensionality are combined by averaging their
        # per-pair similarity scores (matrix averaging), never their vectors.
        doc_indices = defaultdict(list)
        for i, seg in enumerate(self.segments):
            if seg["word_count"] >= self.min_words:
                doc_indices[seg["doc_id"]].append(i)

        names = list(self.embeddings.keys())
        self._doc_vecs = {n: {} for n in names}
        for n in names:
            emb = self.embeddings[n]
            for did, idx in doc_indices.items():
                self._doc_vecs[n][did] = emb[np.array(idx)].mean(axis=0)
        # alias kept for export_similarity_csv ordering
        self._doc_bge = self._doc_vecs[names[0]]

        dids = sorted(doc_indices.keys())
        count = 0
        self._doc_sim_matrix = {}

        for a, b in combinations(dids, 2):
            per_model = {n: _cos(self._doc_vecs[n][a], self._doc_vecs[n][b]) for n in names}
            avg = sum(per_model.values()) / len(per_model)
            rec = dict(per_model)
            rec["avg"] = avg
            self._doc_sim_matrix[(a, b)] = rec

            if avg >= self.doc_threshold:
                agree = all(v >= self.doc_threshold for v in per_model.values())
                attrs = {f"similarity_{n}": round(float(v), 4) for n, v in per_model.items()}
                self.G.add_edge(
                    f"doc:{a}", f"doc:{b}",
                    edge_type="thematically_similar",
                    similarity_avg=round(float(avg), 4),
                    all_models_agree=agree,
                    **attrs,
                )
                count += 1

        logger.info(f"  {count} document similarity edges "
                    f"(threshold={self.doc_threshold}, models={names})")

    # ── Segment similarity ──────────────────────────────────────────────

    def _compute_segment_similarity(self):
        valid = [i for i, s in enumerate(self.segments) if s["word_count"] >= self.min_words]
        idx = np.array(valid)

        norm_mats = {}
        for name, emb in self.embeddings.items():
            m = emb[idx]
            norm_mats[name] = (m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-10)).astype(np.float32)

        n = len(valid)
        logger.info(f"  Computing segment similarity ({n} segments, "
                    f"{len(norm_mats)} models, chunked)...")

        # Chunked top-k: process rows in blocks so memory stays bounded at any
        # corpus size (a full n*n matrix is ~2 GB at 16k segments). A min-heap
        # keeps the global top_k cross-document pairs above the threshold.
        import heapq
        doc_ids_list = [self.segments[ig]["doc_id"] for ig in valid]
        _, doc_codes = np.unique(doc_ids_list, return_inverse=True)
        mats = list(norm_mats.values())
        heap = []  # (score, il, jl), min-heap of size <= top_k
        BLOCK = 1024
        for i0 in range(0, n, BLOCK):
            i1 = min(i0 + BLOCK, n)
            sim_block = mats[0][i0:i1] @ mats[0].T
            for m in mats[1:]:
                sim_block += m[i0:i1] @ m.T
            sim_block /= len(mats)
            # mask same-document pairs and enforce j > i (upper triangle)
            sim_block[doc_codes[i0:i1, None] == doc_codes[None, :]] = 0.0
            cols = np.arange(n)[None, :]
            rows_g = np.arange(i0, i1)[:, None]
            sim_block[cols <= rows_g] = 0.0
            cand = np.argwhere(sim_block >= self.seg_threshold)
            for bi, jl in cand:
                score = float(sim_block[bi, jl])
                item = (score, int(i0 + bi), int(jl))
                if len(heap) < self.top_k:
                    heapq.heappush(heap, item)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, item)

        ranked = sorted(heap, key=lambda x: -x[0])
        count = 0
        self._seg_pairs = defaultdict(lambda: {"count": 0, "max_sim": 0, "pairs": []})

        for score, il, jl in ranked:
            if count >= self.top_k:
                break

            ig, jg = valid[il], valid[jl]
            sa, sb = self.segments[ig], self.segments[jg]

            attrs = {f"similarity_{name}": round(float(m[il] @ m[jl]), 4)
                     for name, m in norm_mats.items()}
            self.G.add_edge(
                f"seg:{sa['segment_id']}", f"seg:{sb['segment_id']}",
                edge_type="semantically_similar",
                similarity_avg=round(score, 4),
                cross_doc=f"{sa['doc_id']} <> {sb['doc_id']}",
                **attrs,
            )

            key = tuple(sorted([sa["doc_id"], sb["doc_id"]]))
            self._seg_pairs[key]["count"] += 1
            self._seg_pairs[key]["max_sim"] = max(self._seg_pairs[key]["max_sim"], score)
            if len(self._seg_pairs[key]["pairs"]) < 3:
                self._seg_pairs[key]["pairs"].append({
                    "a_heading": sa["heading"][:60],
                    "b_heading": sb["heading"][:60],
                    "sim": round(score, 4),
                })
            count += 1

        logger.info(f"  {count} segment similarity edges (threshold={self.seg_threshold})")

    # ── Cross references ────────────────────────────────────────────────

    def _extract_cross_references(self):
        lookup = {
            "2021/1119": "doc:EU_CLIMATE_LAW",
            "2024/1991": "doc:EU_NRL",
            "2000/60": "doc:EU_WFD", "60/2000": "doc:EU_WFD",
        }
        seen, count = set(), 0
        for seg in self.segments:
            if seg["word_count"] < 20:
                continue
            sn = f"seg:{seg['segment_id']}"
            if sn not in self.G:
                continue
            for m in re.finditer(
                r"(?:Regulation|Directive)\s*\((?:EU|EC)\)\s*(?:No\.?\s*)?(\d{4}/\d+|\d+/\d{4})",
                seg["content"],
            ):
                target = lookup.get(m.group(1))
                if target and target in self.G:
                    key = (sn, target)
                    if key not in seen:
                        seen.add(key)
                        self.G.add_edge(sn, target, edge_type="cross_references", ref_text=m.group(0))
                        count += 1
        logger.info(f"  {count} cross-reference edges")

    # ── Summary ─────────────────────────────────────────────────────────

    def _print_summary(self):
        G = self.G
        nt = defaultdict(int)
        for n in G.nodes:
            nt[G.nodes[n].get("node_type", "?")] += 1
        et = defaultdict(int)
        for u, v in G.edges:
            et[G.edges[u, v].get("edge_type", "?")] += 1

        print("\n" + "=" * 60)
        print("SEED GRAPH SUMMARY")
        print("=" * 60)
        print(f"Nodes: {G.number_of_nodes()}")
        for t, c in sorted(nt.items()):
            print(f"  {t:25s}: {c:,}")
        print(f"Edges: {G.number_of_edges()}")
        for t, c in sorted(et.items()):
            print(f"  {t:25s}: {c:,}")
        print(f"Connected components: {nx.number_weakly_connected_components(G)}")
        print("=" * 60)

    # ── Export ──────────────────────────────────────────────────────────

    def export_json(self, path: str):
        data = {"nodes": [], "edges": []}
        for n, attrs in self.G.nodes(data=True):
            data["nodes"].append({"id": n, **{k: str(v) if not isinstance(v, (int, float, str, bool)) else v for k, v in attrs.items()}})
        for u, v, attrs in self.G.edges(data=True):
            data["edges"].append({"source": u, "target": v, **{k: str(v2) if not isinstance(v2, (int, float, str, bool)) else v2 for k, v2 in attrs.items()}})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Exported JSON → {path}")

    def export_graphml(self, path: str):
        G2 = self.G.copy()
        for n in G2.nodes:
            for k, v in list(G2.nodes[n].items()):
                if not isinstance(v, (int, float, str, bool)):
                    G2.nodes[n][k] = str(v)
        for u, v in G2.edges:
            for k, val in list(G2.edges[u, v].items()):
                if not isinstance(val, (int, float, str, bool)):
                    G2.edges[u, v][k] = str(val)
        nx.write_graphml(G2, path)
        logger.info(f"Exported GraphML → {path}")

    def export_similarity_csv(self, path: str):
        dids = sorted(self._doc_bge.keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([""] + dids)
            for a in dids:
                row = [a]
                for b in dids:
                    if a == b:
                        row.append("1.0000")
                    else:
                        key = tuple(sorted([a, b]))
                        val = self._doc_sim_matrix.get(key, {}).get("avg", 0)
                        row.append(f"{val:.4f}")
                w.writerow(row)
        logger.info(f"Exported similarity matrix → {path}")

    def get_viz_data(self) -> dict:
        """Prepare data for the HTML visualization, including node sizing metrics."""

        # ── Build doc-only subgraph for centrality computations ──
        doc_ids = sorted(self._docs.keys())
        doc_sim_edges = []
        for u, v, a in self.G.edges(data=True):
            if a.get("edge_type") == "thematically_similar":
                doc_sim_edges.append((
                    u.replace("doc:", ""), v.replace("doc:", ""),
                    a.get("similarity_avg", 0),
                ))

        # Precompute betweenness centrality at several thresholds
        betweenness_by_thresh = {}
        for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
            dg = nx.Graph()
            dg.add_nodes_from(doc_ids)
            for s, t, sim in doc_sim_edges:
                if sim >= thresh:
                    dg.add_edge(s, t, weight=sim)
            bc = nx.betweenness_centrality(dg, weight="weight")
            betweenness_by_thresh[str(int(thresh * 100))] = {
                k: round(v, 5) for k, v in bc.items()
            }

        # Cross-doc segment link count per document
        seg_link_counts = defaultdict(int)
        for key, info in self._seg_pairs.items():
            seg_link_counts[key[0]] += info["count"]
            seg_link_counts[key[1]] += info["count"]

        # ── Build doc entries ──
        docs = []
        for did, info in sorted(self._docs.items()):
            conns = {}
            for thresh in [0.60, 0.70, 0.80, 0.90]:
                c = 0
                for u, v, a in self.G.edges(data=True):
                    if a.get("edge_type") != "thematically_similar":
                        continue
                    if (u == f"doc:{did}" or v == f"doc:{did}") and a.get("similarity_avg", 0) >= thresh:
                        c += 1
                conns[str(int(thresh * 100))] = c
            docs.append({
                "id": did,
                "title": info["title"],
                "type": info["doc_type"],
                "segs": info["segments"],
                "words": info["words"],
                "conns": conns,
                "seg_links": seg_link_counts.get(did, 0),
            })

        edges = []
        for u, v, a in self.G.edges(data=True):
            if a.get("edge_type") == "thematically_similar":
                edges.append({
                    "s": u.replace("doc:", ""),
                    "t": v.replace("doc:", ""),
                    "avg": a.get("similarity_avg", 0),
                    "agree": a.get("all_models_agree", a.get("dual_model_agree", False)),
                })

        seg_pairs = []
        for key, info in sorted(self._seg_pairs.items(), key=lambda x: -x[1]["count"]):
            seg_pairs.append({
                "a": key[0], "b": key[1],
                "count": info["count"],
                "max_sim": round(info["max_sim"], 4),
                "examples": info["pairs"],
            })

        # ── Connection-strength cutoffs from the corpus similarity ──
        # background distribution (all doc pairs, connected or not).
        # weak / moderate / strong = mean + 1σ / 1.5σ / 2σ (strong clamped to
        # the 99th percentile as a safety net so the class is never empty).
        sims = np.array([v["avg"] for v in self._doc_sim_matrix.values()], dtype=float)
        mu, sd = float(sims.mean()), float(sims.std())
        strength = {
            "mean": round(mu, 4),
            "std": round(sd, 4),
            "n_pairs": int(sims.size),
            "weak": round(mu + 1.0 * sd, 4),
            "moderate": round(mu + 1.5 * sd, 4),
            "strong": round(min(mu + 2.0 * sd, float(np.percentile(sims, 99))), 4),
        }
        logger.info(
            f"  Strength cutoffs (n={sims.size} pairs, mu={mu:.3f}, sd={sd:.3f}): "
            f"weak>={strength['weak']}, moderate>={strength['moderate']}, strong>={strength['strong']}"
        )

        return {
            "docs": docs,
            "edges": edges,
            "seg_pairs": seg_pairs[:25],
            "betweenness": betweenness_by_thresh,
            "strength": strength,
        }


# ═══════════════════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════════════════

def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ═══════════════════════════════════════════════════════════════════════════
# SHARED TEMPLATE PARTS (theme system matches the DAERA alignment tools)
# ═══════════════════════════════════════════════════════════════════════════

# Embedded sidebar logo (Climate Co Centre)
LOGO_B64 = "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"

# DAERA-style theme variables (dark default + light override)
THEME_CSS = """
:root {
  --bg-deep: #0C1821; --bg-panel: #132330; --bg-card: #1A2D3D; --bg-hover: #22384A;
  --border: #243B4F; --border-hover: #3F5A72;
  --text-primary: #E4EDF3; --text-secondary: #8BA3B5; --text-muted: #567088;
  --accent: #3ABFBF; --accent-dim: #3ABFBF20;
  --tier-strong: #00E676; --tier-moderate: #FFB300; --tier-weak: #7A99B0;
  --tier-low: #1B3040; --tier-empty: #0E1920;
  --shadow: rgba(0,0,0,0.5);
}
[data-theme="light"] {
  --bg-deep: #EFF1F5; --bg-panel: #FFFFFF; --bg-card: #F5F7FA; --bg-hover: #E8EEF4;
  --border: #D8DCE4; --border-hover: #B8C4D0;
  --text-primary: #1A2A38; --text-secondary: #3A4A5A; --text-muted: #6A7A8A;
  --accent: #0E8A8A; --accent-dim: #0E8A8A20;
  --tier-strong: #12B0A0; --tier-moderate: #D48F00; --tier-weak: #7A9EC0;
  --tier-low: #E2ECF4; --tier-empty: #F8FAFC;
  --shadow: rgba(30,50,70,0.25);
}
"""

FONTS_LINK = '<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">'

# Theme palettes shared between the SVG/WebGL renderers (JS)
THEME_JS = """
const PALETTES = {
  dark: {
    type: { climate:'#3ABFBF', biodiversity:'#4CAF6E', water:'#5AA3D9',
            agriculture:'#D4903A', forestry:'#5B9E3A', cross_cutting:'#8899AA' },
    strength: { strong:'#00E676', moderate:'#FFB300', weak:'#7A99B0', sub:'#1E3344' },
    accent: '#3ABFBF', bg3d: '#0C1821', label: '#C8D8E4', labelSel: '#ffffff',
    labelDim: '#345068', nodeDim: '#22384A', linkSub: 'rgba(58,191,191,0.04)',
    guide3d: 0x243B4F,
    labelBg: 'rgba(12,24,33,0.82)', labelBorder: 'rgba(74,105,133,0.95)',
  },
  light: {
    type: { climate:'#0E8A8A', biodiversity:'#2E7D4F', water:'#2F6FAE',
            agriculture:'#B26F1D', forestry:'#3E7D2C', cross_cutting:'#5A6B7A' },
    strength: { strong:'#12B0A0', moderate:'#D48F00', weak:'#7A9EC0', sub:'#D8DCE4' },
    accent: '#0E8A8A', bg3d: '#EFF1F5', label: '#3A4A5A', labelSel: '#1A2A38',
    labelDim: '#B8C4D0', nodeDim: '#C8D4E0', linkSub: 'rgba(14,138,138,0.05)',
    guide3d: 0xb8c4d0,
    labelBg: 'rgba(255,255,255,0.88)', labelBorder: 'rgba(150,166,182,0.95)',
  },
};
let theme = (localStorage.getItem('kg-theme') === 'light') ? 'light' : 'dark';
document.documentElement.setAttribute('data-theme', theme);
function pal() { return PALETTES[theme]; }
function typeColor(t) { return pal().type[t] || pal().type.cross_cutting; }
function strengthColor(band) { return pal().strength[band]; }
function hexToRgb(hex) {
  const h = hex.replace('#', '');
  return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)];
}
function colorWithAlpha(hex, a) {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${a})`;
}
"""

THEME_TOGGLE_JS = """
function toggleTheme() {
  theme = theme === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('kg-theme', theme);
  updateThemeBtn();
  onThemeChange();
}
function updateThemeBtn() {
  document.getElementById('theme-label').textContent = theme === 'light' ? 'Dark mode' : 'Light mode';
  document.getElementById('theme-icon').textContent = theme === 'light' ? '\\u263D' : '\\u263C';
}
"""

# Curated datum shortlist: one framework document per policy lever.
# Any other document can still be chosen from the "All documents" group.
DATUM_OPTIONS = [
    "LAND_USE_REVIEW_PHASE_2",   # A Living Land — Land Use Review Phase 2
    "EU_NRL",                    # EU Nature Restoration Law
    "CAP25",                     # Climate Action Plan 2025
    "CLIMATE_ACT_2021",          # Climate Act (statutory framework)
    "NBAP",                      # National Biodiversity Action Plan 2023-2030
    "FOOD_VISION_2030",          # Agri-food strategy
    "FOREST_STRATEGY",           # Ireland's Forest Strategy 2023-2030
    "WAP24",                     # Water Action Plan 2024
    "EU_WFD",                    # EU Water Framework Directive
    "NPF",                       # National Planning Framework
]


# ═══════════════════════════════════════════════════════════════════════════
# COMBINED HTML GENERATOR — 2D / 3D / Heatmap+Ranking with runtime datum
# ═══════════════════════════════════════════════════════════════════════════

def generate_combined_html(viz_data: dict, heat_data: dict, output_path: str,
                           datum_options=None, model_label="dual-model embeddings"):
    """Single standalone HTML explorer.

    Views: 2D network, 3D orbit, Heatmap. A reference-datum selector switches
    the whole interface into datum mode at runtime (no rebuild needed — the
    full similarity matrix is embedded): 2D/3D become radial layouts around
    the datum with guide rings/spheres at the strength cutoffs, and the third
    view becomes a similarity ranking. Curated datum shortlist + any document."""

    datum_options = [d for d in (datum_options or DATUM_OPTIONS) if d in heat_data["ids"]]
    data = dict(viz_data)
    data["heat"] = heat_data
    data["datumOptions"] = datum_options
    data_json = json.dumps(data, separators=(",", ":"))

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Policy Knowledge Graph Explorer</title>
__FONTS_LINK__
<style>
__THEME_CSS__
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'DM Sans', system-ui, sans-serif; background: var(--bg-deep); color: var(--text-primary); overflow: hidden; height: 100vh; }
.app { display: grid; grid-template-columns: 1fr 340px; height: 100vh; }
.graph-area { position: relative; overflow: hidden; background: var(--bg-deep); }
#graph-2d, #graph-3d { position: absolute; inset: 0; }
#graph-3d, #view-3rd { display: none; }
#view-3rd { position: absolute; inset: 0; padding: 12px 14px; }
.sidebar { background: var(--bg-panel); border-left: 1px solid var(--border); overflow-y: auto; overflow-x: hidden; scrollbar-gutter: stable; padding: 20px; display: flex; flex-direction: column; gap: 2px; }
.sidebar::-webkit-scrollbar { width: 4px; }
.sidebar::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

svg { width: 100%; height: 100%; }
.link, .spoke { stroke-linecap: round; }
.guide { fill: none; stroke: var(--border); stroke-dasharray: 4 6; }
.guide-label { font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; letter-spacing: 0.04em; }
.node circle { cursor: pointer; }
.node text { fill: var(--text-secondary); font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 500; pointer-events: none; }
.node.datum circle { stroke-width: 3; }
.node.datum text { fill: var(--text-primary); font-weight: 700; font-size: 11px; }
.node.selected circle { stroke-width: 3 !important; }
.node.selected text { fill: var(--text-primary); font-weight: 700; }
.node.connected circle { stroke-width: 2 !important; opacity: 1 !important; }
.node.connected text { opacity: 1 !important; }
.node.dimmed circle { opacity: 0.07; }
.node.dimmed text { opacity: 0.05; }
.link.dimmed, .spoke.dimmed { opacity: 0.015 !important; }

h1 { font-size: 16px; font-weight: 700; letter-spacing: -0.02em; }
.subtitle { font-size: 12px; color: var(--text-secondary); margin-bottom: 8px; line-height: 1.5; }
.datum-chip { display: none; align-items: center; gap: 6px; background: var(--accent-dim); border: 1px solid var(--accent); border-radius: 8px; padding: 5px 10px; font-size: 11px; font-weight: 600; color: var(--text-primary); margin: 2px 0 8px; line-height: 1.35; }
.datum-chip.show { display: inline-flex; }
.header-row { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }
.header-row img { height: 40px; border-radius: 4px; }

.view-toggle { display: flex; gap: 5px; margin: 4px 0 6px; }
.vt { flex: 1; padding: 7px 4px; background: var(--bg-card); color: var(--text-secondary); border: 1px solid var(--border); border-radius: 6px; font-family: inherit; font-size: 11px; font-weight: 600; cursor: pointer; transition: all 0.15s; }
.vt:hover { border-color: var(--border-hover); }
.vt.active { border-color: var(--accent); color: var(--text-primary); background: var(--accent-dim); }

.btn-row { display: flex; gap: 6px; margin-bottom: 8px; }
.reset-btn, .clear-btn { flex: 1; padding: 8px; background: transparent; color: var(--text-secondary); border: 1px solid var(--border); border-radius: 8px; font-size: 11px; font-family: inherit; cursor: pointer; transition: all 0.15s; }
.reset-btn:hover, .clear-btn:hover { background: var(--bg-hover); border-color: var(--accent); color: var(--text-primary); }
.clear-btn.attn { border-color: var(--tier-moderate); color: var(--text-primary); }

.datum-select-wrap { margin: 2px 0 4px; }
.datum-select { width: 100%; background: var(--bg-card); color: var(--text-primary); border: 1px solid var(--border); border-radius: 8px; padding: 7px 8px; font-size: 11px; font-family: inherit; cursor: pointer; }
.datum-select:hover { border-color: var(--border-hover); }
.datum-select:focus { outline: none; border-color: var(--accent); }

.search-wrap { position: relative; }
.search-input { width: 100%; padding: 7px 10px; margin: 4px 0 2px; border-radius: 8px; border: 1px solid var(--border); background: var(--bg-card); color: var(--text-primary); font-family: inherit; font-size: 11px; outline: none; transition: border-color 0.15s; }
.search-input:focus { border-color: var(--accent); }
.search-hint { font-size: 9px; color: var(--text-muted); min-height: 12px; margin-bottom: 4px; }
.search-drop { position: absolute; top: 100%; left: 0; right: 0; z-index: 50; background: var(--bg-card); border: 1px solid var(--border-hover); border-radius: 8px; margin-top: 2px; max-height: 260px; overflow-y: auto; display: none; box-shadow: 0 10px 30px var(--shadow); }
.search-drop.open { display: block; }
.sd-item { display: flex; align-items: center; gap: 7px; padding: 7px 10px; font-size: 11px; color: var(--text-secondary); cursor: pointer; border-bottom: 1px solid var(--border); }
.sd-item:last-child { border-bottom: none; }
.sd-item:hover, .sd-item.hl { background: var(--bg-hover); color: var(--text-primary); }
.sd-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.sd-text { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sd-more { padding: 6px 10px; font-size: 9px; font-style: italic; color: var(--text-muted); text-align: center; }

.panel { display: flex; flex-direction: column; flex: 0 0 auto; }
.panel-head { font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-muted); margin: 14px 0 6px; padding-bottom: 4px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; gap: 8px; cursor: pointer; user-select: none; transition: color 0.15s; }
.panel-head:hover { color: var(--text-secondary); }
.chev { font-size: 10px; flex-shrink: 0; transition: transform 0.15s; }
.panel.closed .chev { transform: rotate(-90deg); }
.panel.closed .panel-body { display: none; }
.panel.grow { flex: 1 1 auto; min-height: 100px; }
.panel.grow.closed { flex: 0 0 auto; min-height: 0; }
.panel.grow .panel-body { flex: 1; min-height: 0; overflow-y: auto; overflow-x: hidden; }
#detail-panel::-webkit-scrollbar { width: 3px; }
#detail-panel::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.ctrl { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; padding: 2px 0; }
.ctrl label { font-size: 11px; color: var(--text-secondary); min-width: 70px; }
.ctrl input[type=range] { flex: 1; accent-color: var(--accent); }
.ctrl-val { font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600; min-width: 36px; text-align: right; color: var(--text-primary); }
.ctrl select { flex: 1; background: var(--bg-card); color: var(--text-primary); border: 1px solid var(--border); border-radius: 6px; padding: 5px 8px; font-size: 12px; font-family: inherit; cursor: pointer; }
.ctrl select:hover { border-color: var(--border-hover); }
.ctrl select:focus { outline: none; border-color: var(--accent); }
.sizing-hint { font-size: 10px; color: var(--text-muted); margin: -4px 0 8px; font-style: italic; line-height: 1.5; }

.tier-bands { display: flex; flex-direction: column; gap: 4px; margin-top: 6px; }
.tband { display: flex; justify-content: space-between; align-items: center; font-size: 10px; color: var(--text-secondary); padding: 4px 9px; border: 1px solid var(--border); border-radius: 8px; cursor: pointer; transition: all 0.15s; user-select: none; }
.tband:hover { border-color: var(--border-hover); background: var(--bg-hover); }
.tband.active { border-color: var(--accent); background: var(--accent-dim); color: var(--text-primary); }
.tband-label { display: flex; align-items: center; gap: 6px; }
.tband-count { font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 600; color: var(--accent); }
.leg-line { width: 14px; height: 3px; border-radius: 2px; }

.metric-legend { margin-bottom: 10px; }
.metric-toggle { display: flex; justify-content: space-between; align-items: center; cursor: pointer; padding: 6px 10px; background: var(--bg-card); border-radius: 6px; font-size: 11px; color: var(--text-muted); transition: color 0.15s; }
.metric-toggle:hover { color: var(--text-secondary); }
#metric-toggle-arrow { font-size: 10px; transition: transform 0.2s; }
#metric-toggle-arrow.open { transform: rotate(180deg); }
.metric-items { padding: 8px 0 0; }
.metric-item { padding: 8px 10px; margin-bottom: 4px; border-radius: 6px; border-left: 2px solid var(--border); transition: border-color 0.2s, background 0.2s; }
.metric-item.active { border-left-color: var(--accent); background: var(--accent-dim); }
.metric-name { font-size: 11px; font-weight: 600; color: var(--text-secondary); margin-bottom: 3px; }
.metric-item.active .metric-name { color: var(--text-primary); }
.metric-desc { font-size: 10px; color: var(--text-muted); line-height: 1.5; }
.metric-item.active .metric-desc { color: var(--text-secondary); }

.legend { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.leg { display: flex; align-items: center; gap: 5px; font-size: 11px; color: var(--text-secondary); padding: 4px 10px; border-radius: 16px; border: 1px solid var(--border); cursor: pointer; transition: all 0.15s; user-select: none; }
.leg:hover { border-color: var(--border-hover); background: var(--bg-hover); }
.leg.active { border-color: var(--accent); background: var(--accent-dim); color: var(--text-primary); }
.leg-dot { width: 8px; height: 8px; border-radius: 50%; }

.stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }
.stat { background: var(--bg-card); border-radius: 8px; padding: 10px 12px; border: 1px solid var(--border); }
.stat-val { font-family: 'JetBrains Mono', monospace; font-size: 17px; font-weight: 700; color: var(--text-primary); }
.stat-label { font-size: 9px; color: var(--text-muted); margin-top: 2px; text-transform: uppercase; letter-spacing: 0.05em; }

.doc-card { background: var(--bg-card); border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; border: 1px solid var(--border); }
.doc-title { font-size: 13px; font-weight: 600; margin-bottom: 6px; line-height: 1.35; }
.doc-meta { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
.tag { font-size: 10px; font-weight: 500; padding: 2px 8px; border-radius: 10px; }
.doc-row { display: flex; justify-content: space-between; font-size: 12px; padding: 3px 0; color: var(--text-secondary); }
.doc-row-val { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--text-primary); }

.conn-item { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid var(--border); font-size: 12px; }
.conn-item:last-child { border: none; }
.conn-item:hover .conn-name { color: var(--text-primary); }
.conn-name { color: var(--text-secondary); max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: pointer; }
.conn-bar { display: flex; align-items: center; gap: 6px; }
.conn-score { font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 600; color: var(--accent); }
.bar { width: 50px; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 2px; }

.seg-pair { background: var(--bg-card); border-radius: 8px; padding: 10px 12px; margin-bottom: 6px; font-size: 11px; border: 1px solid var(--border); }
.seg-pair-header { display: flex; justify-content: space-between; margin-bottom: 4px; gap: 8px; }
.seg-pair-docs { font-weight: 500; color: var(--text-secondary); }
.seg-pair-count { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--accent); white-space: nowrap; }
.seg-pair-example { color: var(--text-muted); font-size: 10px; line-height: 1.4; margin-top: 4px; padding-top: 4px; border-top: 1px solid var(--border); }

.tooltip {
  position: absolute; pointer-events: none; background: var(--bg-card); border: 1px solid var(--border-hover);
  border-radius: 8px; padding: 10px 14px; font-size: 12px; max-width: 260px;
  box-shadow: 0 8px 32px var(--shadow); opacity: 0; transition: opacity 0.15s;
  z-index: 100;
}
.tooltip.show { opacity: 1; }
.tooltip-title { font-weight: 600; margin-bottom: 4px; }
.tooltip-meta { color: var(--text-secondary); font-size: 11px; }

.theme-toggle { padding: 6px 10px; border-radius: 6px; border: 1px solid var(--border); background: transparent; color: var(--text-muted); font-family: inherit; font-size: 11px; cursor: pointer; transition: all 0.15s; width: 100%; display: flex; justify-content: space-between; align-items: center; }
.theme-toggle:hover { border-color: var(--border-hover); color: var(--text-primary); }
.footer-block { margin-top: auto; padding-top: 16px; }

/* ---- heatmap ---- */
.grid-wrap { overflow: auto; border: 1px solid var(--border); border-radius: 8px; height: 100%; background: var(--bg-panel); }
#grid table { border-collapse: separate; border-spacing: 0; background: var(--bg-panel); }
#grid th, #grid td { padding: 0; }
#grid .corner { position: sticky; top: 0; left: 0; z-index: 7; background: var(--bg-panel); }
#grid thead th { background: var(--bg-panel); }
#grid .col-head { position: sticky; top: 0; z-index: 3; background: var(--bg-panel); height: 132px; vertical-align: bottom; padding-bottom: 4px; cursor: default; }
#grid .col-head span { display: inline-block; writing-mode: vertical-rl; transform: rotate(180deg); font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--text-secondary); white-space: nowrap; max-height: 118px; overflow: hidden; }
#grid .col-dot { width: 100%; height: 3px; border-radius: 2px; margin-top: 2px; }
#grid tbody th { position: sticky; left: 0; z-index: 2; background: var(--bg-panel); text-align: right; padding: 0 8px 0 6px; font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--text-secondary); white-space: nowrap; border-left: 3px solid transparent; cursor: pointer; }
#grid tbody th:hover { color: var(--text-primary); }
#grid tbody th.sel { color: var(--accent); font-weight: 700; }
#grid .group-row th { position: sticky; left: 0; z-index: 2; text-align: left; padding: 6px 8px 3px; font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; background: var(--bg-panel); cursor: default; }
#grid td.cell { width: 13px; height: 13px; min-width: 13px; border-radius: 2px; cursor: crosshair; }
#grid td.cell:hover { outline: 2px solid var(--text-primary); outline-offset: -2px; }
#grid tr.dim-row th, #grid tr.dim-row td { opacity: 0.14; }
#grid td.dim-col, #grid th.dim-col { opacity: 0.14; }
#grid tr.hidden-row { display: none; }
.hm-tooltip {
  position: fixed; pointer-events: none; background: var(--bg-card); border: 1px solid var(--border-hover);
  border-radius: 8px; padding: 10px 14px; font-size: 12px; max-width: 320px;
  box-shadow: 0 8px 32px var(--shadow); opacity: 0; transition: opacity 0.1s; z-index: 100;
}
.hm-tooltip.show { opacity: 1; }
.overview-caption {
  position: absolute; left: 18px; bottom: 18px; z-index: 20; max-width: 440px;
  background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 16px; font-size: 12px; line-height: 1.55; color: var(--text-secondary);
  box-shadow: 0 8px 32px var(--shadow); opacity: 0; transition: opacity 0.6s;
  pointer-events: none;
}
.overview-caption.show { opacity: 1; }
.overview-caption .oc-title { font-weight: 700; color: var(--text-primary); font-size: 13px; margin-bottom: 3px; }
.overview-caption .oc-stats { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--accent); margin-top: 5px; }
.tt-band { font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.tt-docs { font-weight: 600; line-height: 1.4; margin-bottom: 4px; }
.tt-meta { color: var(--text-secondary); font-size: 11px; }

/* ---- ranking ---- */
.rank-wrap { overflow: auto; border: 1px solid var(--border); border-radius: 8px; height: 100%; background: var(--bg-panel); }
.rank-table { width: 100%; border-collapse: separate; border-spacing: 0; }
.rank-table th { position: sticky; top: 0; z-index: 2; background: var(--bg-panel); text-align: left; font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); padding: 10px 12px 8px; border-bottom: 1px solid var(--border); }
.rank-table td { padding: 7px 12px; border-bottom: 1px solid var(--border); font-size: 12px; }
.rank-table tr.r-row { cursor: pointer; transition: background 0.1s; }
.rank-table tr.r-row:hover td { background: var(--bg-hover); }
.rank-table tr.r-row.sel td { background: var(--accent-dim); }
.rank-table tr.hidden-row { display: none; }
.rank-table tr.dim-row td { opacity: 0.25; }
.r-rank { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--text-muted); width: 40px; }
.r-doc { display: flex; align-items: center; gap: 8px; }
.r-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.r-title { color: var(--text-primary); line-height: 1.35; }
.r-id { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--text-muted); }
.r-dom { white-space: nowrap; width: 110px; }
.r-sim { width: 200px; }
.r-sim-inner { display: flex; align-items: center; gap: 8px; }
.r-val { font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; min-width: 44px; }
.r-bar { flex: 1; height: 5px; background: var(--border); border-radius: 3px; overflow: hidden; }
.r-bar-fill { height: 100%; border-radius: 3px; }
</style>
</head>
<body>
<div class="app">
  <div class="graph-area" id="graph-area">
    <div id="graph-2d"><svg id="svg"></svg><div class="tooltip" id="tooltip"></div></div>
    <div id="graph-3d"></div>
    <div class="overview-caption" id="overview-caption"></div>
    <div id="view-3rd">
      <div class="grid-wrap" id="grid-wrap" style="display:none"><div id="grid"></div></div>
      <div class="rank-wrap" id="rank-wrap" style="display:none"><div id="rank"></div></div>
    </div>
  </div>
  <div class="sidebar" id="sidebar">
    <div class="header-row"><img src="data:image/jpeg;base64,__LOGO_B64__" alt="Climate Co Centre"><h1>Policy KG explorer</h1></div>
    <div class="subtitle">Irish/EU policy corpus &middot; <span id="doc-count"></span> documents &middot; __MODEL_LABEL__</div>
    <div class="datum-chip" id="datum-chip"><span>&#9673;</span><span id="datum-chip-title"></span></div>

    <div class="datum-select-wrap">
      <select class="datum-select" id="datum-select" onchange="setDatum(this.value)"></select>
    </div>

    <div class="view-toggle">
      <button class="vt active" id="btn-2d" onclick="setView('2d')">2D network</button>
      <button class="vt" id="btn-3d" onclick="setView('3d')">3D orbit</button>
      <button class="vt" id="btn-3rd" onclick="setView('3rd')">Heatmap</button>
    </div>
    <div class="btn-row">
      <button class="reset-btn" onclick="resetGraph()">Reset view</button>
      <button class="clear-btn" id="clear-btn" onclick="clearFilters()">Clear filters</button>
    </div>
    <div class="search-wrap">
      <input type="search" id="graph-search" class="search-input" placeholder="Search documents..." autocomplete="off">
      <div class="search-drop" id="search-drop"></div>
    </div>
    <div class="search-hint" id="search-count"></div>

    <div class="panel" data-panel="range">
      <div class="panel-head" onclick="togglePanel(this)"><span id="range-heading">Similarity range &amp; strength</span><span class="chev">&#9662;</span></div>
      <div class="panel-body">
        <div class="ctrl">
          <label>Min similarity</label>
          <input type="range" id="thresh-min" min="30" max="100" value="75" step="1">
          <span class="ctrl-val" id="thresh-min-val">0.75</span>
        </div>
        <div class="ctrl">
          <label>Max similarity</label>
          <input type="range" id="thresh-max" min="30" max="100" value="100" step="1">
          <span class="ctrl-val" id="thresh-max-val">1.00</span>
        </div>
        <div class="tier-bands" id="strength-legend"></div>
        <div class="sizing-hint" style="margin-top:6px" id="strength-hint"></div>
      </div>
    </div>

    <div class="panel" data-panel="sizing">
      <div class="panel-head" onclick="togglePanel(this)">Node sizing<span class="chev">&#9662;</span></div>
      <div class="panel-body">
        <div class="ctrl">
          <label>Sized by</label>
          <select id="sizing-mode" onchange="updateNodeSizing(this.value)">
            <option value="doc_size" selected>Document size</option>
            <option value="degree">Graph degree</option>
            <option value="betweenness">Betweenness centrality</option>
            <option value="seg_links">Segment links</option>
            <option value="weighted_deg">Weighted degree</option>
          </select>
        </div>
        <div class="sizing-hint" id="sizing-hint">Node radius reflects segment count</div>
        <div class="metric-legend" id="metric-legend">
          <div class="metric-toggle" onclick="toggleMetricLegend()">
            <span id="metric-toggle-label">What do these mean?</span>
            <span id="metric-toggle-arrow">&#9662;</span>
          </div>
          <div class="metric-items" id="metric-items" style="display:none">
            <div class="metric-item" data-mode="doc_size">
              <div class="metric-name">Document size</div>
              <div class="metric-desc">Segment count. Shows corpus composition &mdash; which documents are physically the largest. Useful as a baseline but doesn&rsquo;t reflect policy importance.</div>
            </div>
            <div class="metric-item" data-mode="degree">
              <div class="metric-name">Graph degree</div>
              <div class="metric-desc">Connection count within the current similarity range. This is reactive &mdash; drag the sliders and watch nodes resize in real time.</div>
            </div>
            <div class="metric-item" data-mode="betweenness">
              <div class="metric-name">Betweenness centrality</div>
              <div class="metric-desc">Surfaces documents that bridge between otherwise disconnected policy clusters. High-centrality documents are the connective tissue &mdash; e.g. if you want to trace how climate policy connects to water regulation, the largest nodes here are the path.</div>
            </div>
            <div class="metric-item" data-mode="seg_links">
              <div class="metric-name">Segment links</div>
              <div class="metric-desc">Cross-document text overlap at the paragraph level. Rewards substantive textual alignment rather than just thematic proximity &mdash; documents that share specific policy language or provisions, not just broad topics.</div>
            </div>
            <div class="metric-item" data-mode="weighted_deg">
              <div class="metric-name">Weighted degree</div>
              <div class="metric-desc">Sum of similarity scores for all connections. Rewards documents that are both broadly connected and strongly connected. Highlights comprehensive documents that touch every domain at high similarity.</div>
            </div>
          </div>
        </div>
        <div class="sizing-hint" id="label-hint">3D labels: the most important nodes by this metric are labelled; hover any node for its name.</div>
      </div>
    </div>

    <div class="panel" data-panel="domains">
      <div class="panel-head" onclick="togglePanel(this)">Policy domains (click to filter)<span class="chev">&#9662;</span></div>
      <div class="panel-body"><div class="legend" id="legend"></div></div>
    </div>

    <div class="panel" data-panel="stats">
      <div class="panel-head" onclick="togglePanel(this)">Graph stats<span class="chev">&#9662;</span></div>
      <div class="panel-body"><div class="stats" id="stats"></div></div>
    </div>

    <div class="panel grow" data-panel="selection">
      <div class="panel-head" onclick="togglePanel(this)">Selection<span class="chev">&#9662;</span></div>
      <div class="panel-body" id="detail-panel"></div>
    </div>

    <div class="footer-block">
      <button class="theme-toggle" id="theme-btn" onclick="toggleTheme()">
        <span id="theme-label">Light mode</span><span id="theme-icon">&#9788;</span>
      </button>
    </div>
  </div>
</div>
<div class="hm-tooltip" id="hm-tooltip"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.137.5/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/3d-force-graph@1.70.20/dist/3d-force-graph.min.js"></script>
<script>
const DATA = __DATA_JSON__;

__THEME_JS__

const TYPE_LABELS = {
  climate: 'Climate', biodiversity: 'Biodiversity', water: 'Water',
  agriculture: 'Agriculture', forestry: 'Forestry', cross_cutting: 'Cross-cutting'
};

const S = DATA.strength;
function strengthOf(v) {
  if (v >= S.strong) return 'strong';
  if (v >= S.moderate) return 'moderate';
  if (v >= S.weak) return 'weak';
  return 'sub';
}

// ── Shared state ───────────────────────────────────────────
let viewMode = '2d';
const DEFAULT_MIN_CORPUS = Math.max(0.30, Math.round(S.weak * 100) / 100);
let tMin = DEFAULT_MIN_CORPUS;
let tMax = 1.00;
let selectedDoc = null;
let filterType = null;
let sizingMode = 'doc_size';
let activeBand = null;
let searchTerm = '';
let cachedConn = new Set();

// datum state (null = corpus mode)
let datumId = null;
let SIMS = {};            // other doc id -> similarity to datum
let rankedIds = [];       // ids sorted by similarity to datum, desc
let simMin = 0, simMax = 1;

const HM = DATA.heat;     // { ids, m } — full matrix, domain-sorted order
const hmIndex = Object.fromEntries(HM.ids.map((id, i) => [id, i]));

function defaultMin() { return datumId ? 0.30 : DEFAULT_MIN_CORPUS; }

const SIZING_HINTS = {
  doc_size: 'Node radius reflects segment count',
  degree: 'Node radius reflects number of connections in the similarity range',
  betweenness: 'Node radius reflects bridging role between policy clusters',
  seg_links: 'Node radius reflects fine-grained cross-document text overlap',
  weighted_deg: 'Node radius reflects total connection strength',
};

function getEdges() {
  return DATA.edges.filter(e => e.avg >= tMin && e.avg <= tMax);
}

function passesFilter(d) { return !filterType || d.type === filterType || d.id === datumId; }
function nodeMatches(d) {
  if (!searchTerm) return true;
  const hay = (d.id + ' ' + (d.title || '') + ' ' + (TYPE_LABELS[d.type] || '')).toLowerCase();
  return searchTerm.split(/\\s+/).filter(Boolean).every(t => hay.includes(t));
}
function inRange(d) {
  if (!datumId || d.id === datumId) return true;
  return SIMS[d.id] >= tMin && SIMS[d.id] <= tMax;
}
function nodeVisible(d) {
  if (datumId && d.id === datumId) return true;
  return passesFilter(d) && nodeMatches(d) && inRange(d);
}
function hasFilter() { return !!(filterType || searchTerm); }
function hasAnyActiveFilter() {
  return !!(filterType || searchTerm || activeBand ||
    Math.abs(tMin - defaultMin()) > 0.005 || tMax < 0.995);
}
function updateClearBtn() {
  const b = document.getElementById('clear-btn');
  if (b) b.classList.toggle('attn', hasAnyActiveFilter());
}

function recomputeConn() {
  cachedConn = new Set();
  if (!selectedDoc || datumId) return;
  getEdges().forEach(e => {
    if (e.s === selectedDoc || e.t === selectedDoc) { cachedConn.add(e.s); cachedConn.add(e.t); }
  });
}

// ── Node radii ─────────────────────────────────────────────
function computeRadii(mode) {
  const edges = getEdges();
  const degree = {};
  const weightedDeg = {};
  DATA.docs.forEach(d => { degree[d.id] = 0; weightedDeg[d.id] = 0; });
  edges.forEach(e => {
    degree[e.s] += 1; degree[e.t] += 1;
    weightedDeg[e.s] += e.avg; weightedDeg[e.t] += e.avg;
  });
  const bKey = String(Math.min(90, Math.max(50, Math.round(tMin * 20) * 5)));
  const bc = DATA.betweenness[bKey] || DATA.betweenness['75'] || {};
  const raw = {};
  DATA.docs.forEach(d => {
    switch (mode) {
      case 'doc_size':    raw[d.id] = d.segs; break;
      case 'degree':      raw[d.id] = degree[d.id] || 0; break;
      case 'betweenness': raw[d.id] = bc[d.id] || 0; break;
      case 'seg_links':   raw[d.id] = d.seg_links || 0; break;
      case 'weighted_deg':raw[d.id] = weightedDeg[d.id] || 0; break;
    }
  });
  const vals = Object.values(raw);
  const mn = Math.min(...vals), mx = Math.max(...vals);
  const range = mx - mn || 1;
  const radii = {};
  DATA.docs.forEach(d => { radii[d.id] = 8 + ((raw[d.id] - mn) / range) * 28; });
  return { radii, raw };
}

function datumNodeR(d) { return d.id === datumId ? 16 : Math.max(5, Math.min(14, 3 + Math.sqrt(d.segs) * 0.8)); }

function refreshRadiiForMode() {
  if (datumId) {
    nodes2d.forEach(d => { d.r = datumNodeR(d); d.metricVal = d.id === datumId ? 0 : SIMS[d.id]; });
    nodes3d.forEach(d => { d.r = datumNodeR(d); d.metricVal = d.id === datumId ? 0 : SIMS[d.id]; });
  } else {
    const { radii, raw } = computeRadii(sizingMode);
    nodes2d.forEach(d => { d.r = radii[d.id]; d.metricVal = raw[d.id]; });
    nodes3d.forEach(d => { d.r = radii[d.id]; d.metricVal = raw[d.id]; });
  }
}

const initRadii = computeRadii('doc_size');
const nodes2d = DATA.docs.map(d => ({ ...d, r: initRadii.radii[d.id], metricVal: initRadii.raw[d.id] }));
const nodes3d = DATA.docs.map(d => ({ ...d, r: initRadii.radii[d.id], metricVal: initRadii.raw[d.id] }));
const nodeMap = Object.fromEntries(nodes2d.map(n => [n.id, n]));

// ── Node strength bands (ring colours) ─────────────────────
let nodeBands = {};
function recomputeNodeBands() {
  nodeBands = {};
  if (datumId) {
    DATA.docs.forEach(d => { nodeBands[d.id] = d.id === datumId ? 'datum' : strengthOf(SIMS[d.id]); });
    return;
  }
  const best = {};
  getEdges().forEach(e => {
    best[e.s] = Math.max(best[e.s] || 0, e.avg);
    best[e.t] = Math.max(best[e.t] || 0, e.avg);
  });
  DATA.docs.forEach(d => { nodeBands[d.id] = best[d.id] ? strengthOf(best[d.id]) : 'none'; });
}
function nodeRingColor(id) {
  const b = nodeBands[id];
  if (b === 'datum') return pal().accent;
  if (!b || b === 'none' || b === 'sub') return pal().nodeDim;
  return strengthColor(b);
}

// ── Datum switching ────────────────────────────────────────
function setDatum(id) {
  datumId = id || null;
  if (datumId) {
    const di = hmIndex[datumId];
    SIMS = {};
    HM.ids.forEach((oid, j) => { if (oid !== datumId) SIMS[oid] = HM.m[di][j]; });
    rankedIds = HM.ids.filter(x => x !== datumId).sort((a, b) => SIMS[b] - SIMS[a]);
    const vals = Object.values(SIMS);
    simMin = Math.min(...vals); simMax = Math.max(...vals);
    if (selectedDoc === datumId) selectedDoc = null;
    document.getElementById('datum-chip-title').textContent = 'Datum: ' + nodeMap[datumId].title;
    document.getElementById('datum-chip').classList.add('show');
    document.getElementById('btn-3rd').textContent = 'Ranking';
    document.getElementById('range-heading').textContent = 'Similarity to datum';
    // pin datum in 2D/3D
    const dn2 = nodeMap[datumId];
    nodes2d.forEach(n => { n.fx = null; n.fy = null; });
    dn2.fx = CX; dn2.fy = CY;
    nodes3d.forEach(n => { n.fx = undefined; n.fy = undefined; n.fz = undefined; });
    const dn3 = nodes3d.find(n => n.id === datumId);
    dn3.fx = 0; dn3.fy = 0; dn3.fz = 0;
  } else {
    SIMS = {}; rankedIds = [];
    document.getElementById('datum-chip').classList.remove('show');
    document.getElementById('btn-3rd').textContent = 'Heatmap';
    document.getElementById('range-heading').textContent = 'Similarity range & strength';
    nodes2d.forEach(n => { n.fx = null; n.fy = null; });
    nodes3d.forEach(n => { n.fx = undefined; n.fy = undefined; n.fz = undefined; });
  }
  document.getElementById('datum-select').value = datumId || '';
  // reset range + band to mode defaults
  activeBand = null;
  tMin = defaultMin(); tMax = 1.00;
  didFit3D = false;
  updatePanelVisibility();
  buildStrengthBands();
  highlightBand();
  syncSliderUI();
  // rebuild the active third view if needed
  if (viewMode === '3rd') { show3rd(); }
  refreshStructure();
}

function updatePanelVisibility() {
  const sizingPanel = document.querySelector('.panel[data-panel="sizing"]');
  if (sizingPanel) sizingPanel.style.display = (datumId || viewMode === '3rd') ? 'none' : 'flex';
}

function buildDatumSelect() {
  const sel = document.getElementById('datum-select');
  const curated = DATA.datumOptions.filter(id => nodeMap[id]);
  const curatedSet = new Set(curated);
  let h = '<option value="">No datum — full corpus view</option>';
  h += '<optgroup label="Suggested datums">';
  curated.forEach(id => { h += `<option value="${id}">${nodeMap[id].title}</option>`; });
  h += '</optgroup><optgroup label="All documents">';
  [...DATA.docs].filter(d => !curatedSet.has(d.id)).sort((a, b) => a.title.localeCompare(b.title))
    .forEach(d => { h += `<option value="${d.id}">${d.title}</option>`; });
  h += '</optgroup>';
  sel.innerHTML = h;
  sel.value = datumId || '';
}

// ═══════════════════════════════ 2D VIEW ═══════════════════
const svg = d3.select('#svg');
const container2d = svg.append('g');
const guideGroup = container2d.append('g');
const linkGroup = container2d.append('g');
const nodeGroup = container2d.append('g');

const area = document.getElementById('graph-area');
const W = area.clientWidth, H = area.clientHeight;
const CX = W / 2, CY = H / 2;
const MAXR = Math.min(W, H) / 2 - 60;
svg.attr('viewBox', `0 0 ${W} ${H}`);

const zoomBehavior = d3.zoom()
  .scaleExtent([0.3, 4])
  .on('zoom', (e) => container2d.attr('transform', e.transform));
svg.call(zoomBehavior);

const sim = d3.forceSimulation(nodes2d)
  .force('center', d3.forceCenter(CX, CY))
  .force('charge', d3.forceManyBody().strength(-200))
  .force('collide', d3.forceCollide(d => d.r + 12))
  .force('x', d3.forceX(CX).strength(0.04))
  .force('y', d3.forceY(CY).strength(0.04))
  .on('tick', ticked);

const drag = d3.drag()
  .on('start', (e, d) => { if (d.id === datumId) return; if (!e.active) sim.alphaTarget(0.15).restart(); d.fx = d.x; d.fy = d.y; })
  .on('drag', (e, d) => { if (d.id === datumId) return; d.fx = e.x; d.fy = e.y; })
  .on('end', (e, d) => { if (d.id === datumId) return; if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; });

function radiusOf(simv) {
  const t = (simMax - simv) / (simMax - simMin || 1);
  return 70 + t * (MAXR - 70);
}

const tooltip = document.getElementById('tooltip');
const MODE_LABELS = {
  doc_size: 'Segments', degree: 'Connections', betweenness: 'Centrality',
  seg_links: 'Segment links', weighted_deg: 'Weighted degree',
};
function metricDisplay(d) {
  return sizingMode === 'betweenness' ? d.metricVal.toFixed(4) : Math.round(d.metricVal * 100) / 100;
}
function showTooltip(e, d) {
  let extra;
  if (datumId) {
    extra = d.id === datumId
      ? '<div class="tooltip-meta" style="margin-top:4px">Reference datum</div>'
      : `<div class="tooltip-meta" style="margin-top:4px;color:${strengthColor(strengthOf(SIMS[d.id]))}">Similarity to datum: ${SIMS[d.id].toFixed(3)} (${strengthOf(SIMS[d.id])})</div>`;
  } else {
    extra = `<div class="tooltip-meta" style="margin-top:4px;color:${pal().accent}">${MODE_LABELS[sizingMode]}: ${metricDisplay(d)}</div>`;
  }
  tooltip.innerHTML = `<div class="tooltip-title">${d.title}</div>
    <div class="tooltip-meta">${TYPE_LABELS[d.type]} &middot; ${d.segs} segments &middot; ${(d.words/1000).toFixed(0)}k words</div>${extra}`;
  tooltip.classList.add('show');
  tooltip.style.left = (e.offsetX + 16) + 'px';
  tooltip.style.top = (e.offsetY - 10) + 'px';
}
function hideTooltip() { tooltip.classList.remove('show'); }

function label2D(d) {
  if (datumId) {
    // adaptive labels in radial mode, ranked among currently visible docs
    const top = new Set(rankedIds.filter(id => nodeVisible(nodeMap[id])).slice(0, 18));
    const show = d.id === datumId || d.id === selectedDoc ||
      (searchTerm ? (nodeMatches(d) && passesFilter(d) && inRange(d)) : (top.has(d.id) && nodeVisible(d)));
    if (!show) return '';
  }
  const s = d.id.replace(/_/g, ' ');
  return s.length > 20 ? s.slice(0, 18) + '..' : s;
}

function drawGuides() {
  guideGroup.selectAll('*').remove();
  if (!datumId) return;
  [
    { v: S.strong,   band: 'strong',   lbl: `strong \\u2265 ${S.strong.toFixed(2)}` },
    { v: S.moderate, band: 'moderate', lbl: `moderate \\u2265 ${S.moderate.toFixed(2)}` },
    { v: S.weak,     band: 'weak',     lbl: `weak \\u2265 ${S.weak.toFixed(2)}` },
  ].forEach(g => {
    if (g.v > simMax || g.v < simMin) return;
    const r = radiusOf(g.v);
    guideGroup.append('circle').attr('class', 'guide')
      .attr('cx', CX).attr('cy', CY).attr('r', r)
      .attr('stroke', strengthColor(g.band)).attr('stroke-opacity', 0.9)
      .attr('stroke-width', 2).attr('stroke-dasharray', '7 5');
    guideGroup.append('text').attr('class', 'guide-label')
      .attr('fill', strengthColor(g.band)).attr('fill-opacity', 1)
      .attr('x', CX + 8).attr('y', CY - r - 7).text(g.lbl);
  });
}

function render2D() {
  refreshRadiiForMode();
  recomputeNodeBands();
  drawGuides();

  let edgeData;
  if (datumId) {
    edgeData = rankedIds.map(id => ({ source: datumId, target: id, avg: SIMS[id], spoke: true }));
    sim.force('link', null);
    sim.force('radial', d3.forceRadial(d => d.id === datumId ? 0 : radiusOf(SIMS[d.id]), CX, CY).strength(0.85));
    sim.force('charge', d3.forceManyBody().strength(-30));
  } else {
    edgeData = getEdges().map(e => ({ source: e.s, target: e.t, avg: e.avg, spoke: false }));
    sim.force('radial', null);
    sim.force('link', d3.forceLink(edgeData)
      .id(d => d.id)
      .distance(d => Math.max(40, (1 - d.avg) * 250))
      .strength(d => d.avg * 0.35));
    sim.force('charge', d3.forceManyBody().strength(-200));
  }
  sim.force('collide', d3.forceCollide(d => d.r + (datumId ? 8 : 12)));
  sim.alpha(0.5).restart();

  const links = linkGroup.selectAll('line').data(edgeData, d => (d.source.id||d.source) + '_' + (d.target.id||d.target));
  links.exit().remove();
  const linksEnter = links.enter().append('line').attr('class', 'link');
  linksEnter.merge(links)
    .attr('stroke', d => strengthColor(strengthOf(d.avg)))
    .attr('stroke-width', d => Math.max(0.4, (d.avg - 0.5) * (datumId ? 4 : 5)))
    .attr('opacity', d => {
      if (datumId) {
        const band = strengthOf(d.avg);
        return { strong: 0.75, moderate: 0.45, weak: 0.22, sub: 0.06 }[band];
      }
      return Math.max(0.12, (d.avg - 0.5) * 1.5);
    });

  const ng = nodeGroup.selectAll('g.node').data(nodes2d, d => d.id);
  const enter = ng.enter().append('g').attr('class', 'node').call(drag)
    .on('click', (e, d) => { if (d.id !== datumId) { selectedDoc = selectedDoc === d.id ? null : d.id; refreshAll(); } })
    .on('mouseenter', (e, d) => showTooltip(e, d))
    .on('mouseleave', hideTooltip);
  enter.append('circle');
  enter.append('text').attr('text-anchor', 'middle');

  const all = enter.merge(ng);
  all.classed('datum', d => d.id === datumId);
  all.select('circle')
    .attr('r', d => d.r)
    .attr('fill', d => d.id === datumId ? pal().accent + '40' : typeColor(d.type) + '30')
    .attr('stroke', d => d.id === datumId ? pal().accent : nodeRingColor(d.id))
    .attr('stroke-width', d => d.id === datumId ? 3 : 2);
  all.select('text')
    .attr('dy', d => d.r + (datumId ? 12 : 14))
    .text(d => label2D(d));

  applySelection2D();
}

function ticked() {
  linkGroup.selectAll('line')
    .attr('x1', d => (d.source.x !== undefined ? d.source.x : (nodeMap[d.source] || {}).x) || CX)
    .attr('y1', d => (d.source.y !== undefined ? d.source.y : (nodeMap[d.source] || {}).y) || CY)
    .attr('x2', d => (d.target.x !== undefined ? d.target.x : (nodeMap[d.target] || {}).x) || CX)
    .attr('y2', d => (d.target.y !== undefined ? d.target.y : (nodeMap[d.target] || {}).y) || CY);
  nodeGroup.selectAll('g.node')
    .attr('transform', d => `translate(${d.x},${d.y})`);
}

function applySelection2D() {
  const sel = selectedDoc;
  nodeGroup.selectAll('g.node')
    .classed('selected', d => d.id === sel)
    .classed('connected', d => !datumId && sel && d.id !== sel && cachedConn.has(d.id) && nodeVisible(d))
    .classed('dimmed', d =>
      d.id !== datumId && d.id !== sel &&
      ((!datumId && sel && !cachedConn.has(d.id)) || !nodeVisible(d)));
  nodeGroup.selectAll('g.node').select('text').text(d => label2D(d));
  linkGroup.selectAll('line').each(function(d) {
      const s = d.source.id || d.source, t = d.target.id || d.target;
      const isConn = sel && (s === sel || t === sel);
      let filtered;
      if (datumId) {
        filtered = !nodeVisible(nodeMap[t] || {});
      } else {
        filtered = hasFilter() && !(nodeVisible(nodeMap[s] || {}) && nodeVisible(nodeMap[t] || {}));
      }
      d3.select(this)
        .classed('dimmed', (!datumId && sel && !isConn) || filtered)
        .attr('stroke', isConn ? pal().accent : strengthColor(strengthOf(d.avg)));
    });
}

// ═══════════════════════════════ 3D VIEW ═══════════════════
let graph3D = null;
let didFit3D = false;
let topLabelIds = new Set();
const el3d = document.getElementById('graph-3d');
const R3_SCALE = 0.55;

// Camera-distance-adaptive link width: very fine when zoomed out (the whole
// graph in view), thickening as the user zooms in. Quantised so the link
// geometry is only rebuilt when the scale steps, not on every frame.
let linkWidthScale = 0.4;
let _lwLast = 0;
let compact3D = false;
let _nodeScale = 1;
let zoomLabelAll = false;   // zoomed in close: label every visible node
let perfLines = false;      // many edges visible: render as hairline lines (fast)
function updatePerfMode() {
  // width>0 links render as 3D tube meshes — fine for hundreds, laggy for
  // thousands. Above this threshold, fall back to GPU line primitives.
  perfLines = !datumId && getEdges().length > 900;
}

// Zoomed-out overview: pull the layout tighter and enlarge node objects so
// the full graph and its connections read as one structure.
function applyCompact3D(on) {
  if (!graph3D || compact3D === on || datumId) return;
  compact3D = on;
  configure3DForces();
  try { graph3D.d3ReheatSimulation(); } catch (e) {}
  updateOverviewCaption();
}

function applyZoomLimits() {
  // The camera can never pull back past "whole dataset comfortably in frame",
  // so the graph is always visible; zoom-in keeps a floor so the user can't
  // fly inside a node.
  if (!graph3D) return;
  let controls;
  try { controls = graph3D.controls(); } catch (e) { return; }
  controls.minDistance = 30;
  if (datumId) {
    controls.maxDistance = 1400;   // generous — the shells define their own scale
    return;
  }
  let maxR2 = 0;
  nodes3d.forEach(n => {
    const d2 = (n.x || 0) ** 2 + (n.y || 0) ** 2 + (n.z || 0) ** 2;
    if (d2 > maxR2) maxR2 = d2;
  });
  const R = Math.sqrt(maxR2) + 70;
  const fov = ((graph3D.camera().fov || 40) * Math.PI) / 180;
  const fitD = R / Math.tan(fov / 2);
  // keep the ceiling above the compact-mode trigger so the overview
  // behaviour stays reachable
  controls.maxDistance = Math.max(950, fitD * 1.12);
}

function updateOverviewCaption() {
  const el = document.getElementById('overview-caption');
  if (!el) return;
  const show = viewMode === '3d' && !datumId;
  if (show) {
    el.innerHTML = `<div class="oc-title">The policy web</div>
      Every line is measured thematic overlap between two documents. The tangle is not
      noise &mdash; it is the finding: Irish environmental governance is deeply
      interconnected, and no document stands alone. Zoom in to untangle it.
      <div class="oc-stats">${DATA.docs.length} documents &middot; ${getEdges().length} connections in view</div>`;
  }
  el.classList.toggle('show', show);
}

function applyNodeScale(d) {
  const s = Math.min(1.35, Math.max(1, d / 560));
  if (Math.abs(s - _nodeScale) < 0.08) return;
  _nodeScale = s;
  nodes3d.forEach(n => {
    const o = n.__threeObj;
    if (o) o.scale.setScalar(s);
  });
}

function updateLinkWidthScale() {
  if (!graph3D) return;
  const now = Date.now();
  if (now - _lwLast < 150) return;
  _lwLast = now;
  const d = graph3D.camera().position.length();
  updateGuideRingScales();
  // fog tracks the camera: starts just short of the graph centre and fades
  // across the far half, so depth is legible at any zoom and the graph
  // never disappears when zoomed far out
  try {
    const fog = graph3D.scene().fog;
    if (fog) { fog.near = Math.max(100, d - 120); fog.far = d + 900; }
  } catch (e) {}
  // compact overview with hysteresis so it doesn't flip-flop at the boundary
  if (!datumId) {
    if (!compact3D && d > 800) applyCompact3D(true);
    else if (compact3D && d < 620) applyCompact3D(false);
  }
  applyNodeScale(d);
  let f = Math.min(2.5, Math.max(0.18, 340 / Math.max(1, d)));
  if (f > 1) f = 1 + (f - 1) * 0.25;   // thicken at a quarter of the original rate when zooming in
  const q = Math.round(f * 4) / 4;
  if (q !== linkWidthScale) {
    linkWidthScale = q;
    if (!perfLines) graph3D.linkWidth(graph3D.linkWidth());  // widths constant 0 in perf mode
    graph3D.linkColor(l => link3DColor(l));   // re-apply zoom-scaled alphas
  }
  const za = d < 400;   // close enough that full labelling stays readable
  if (za !== zoomLabelAll) {
    zoomLabelAll = za;
    computeTopLabels();
    graph3D.nodeThreeObject(n => makeNodeObject(n));
    _nodeScale = 1;
    setTimeout(() => { try { applyNodeScale(graph3D.camera().position.length()); } catch (e) {} }, 80);
  }
}

const _ringCache = {};
function ringTexture(color) {
  if (_ringCache[color]) return _ringCache[color];
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const ctx = c.getContext('2d');
  ctx.strokeStyle = color; ctx.lineWidth = 20;
  ctx.beginPath(); ctx.arc(128, 128, 112, 0, Math.PI * 2); ctx.stroke();
  const tex = new THREE.CanvasTexture(c);
  _ringCache[color] = tex;
  return tex;
}
function textTexture(text, color, borderColor) {
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d');
  const font = '600 48px "JetBrains Mono", "DM Sans", system-ui, sans-serif';
  ctx.font = font;
  const padX = 20, H = 78, radius = 16, bw = 3;
  const w = Math.ceil(ctx.measureText(text).width) + padX * 2;
  c.width = w; c.height = H;
  const ctx2 = c.getContext('2d');
  // rounded-pill background + border
  const x = bw, y = bw, rw = w - bw * 2, rh = H - bw * 2;
  ctx2.beginPath();
  ctx2.moveTo(x + radius, y);
  ctx2.lineTo(x + rw - radius, y);
  ctx2.arcTo(x + rw, y, x + rw, y + radius, radius);
  ctx2.lineTo(x + rw, y + rh - radius);
  ctx2.arcTo(x + rw, y + rh, x + rw - radius, y + rh, radius);
  ctx2.lineTo(x + radius, y + rh);
  ctx2.arcTo(x, y + rh, x, y + rh - radius, radius);
  ctx2.lineTo(x, y + radius);
  ctx2.arcTo(x, y, x + radius, y, radius);
  ctx2.closePath();
  ctx2.fillStyle = pal().labelBg;
  ctx2.fill();
  ctx2.lineWidth = bw;
  ctx2.strokeStyle = borderColor || pal().labelBorder;
  ctx2.stroke();
  // label text
  ctx2.font = font; ctx2.fillStyle = color; ctx2.textBaseline = 'middle';
  ctx2.fillText(text, padX, H / 2 + 2);
  return { tex: new THREE.CanvasTexture(c), aspect: w / H };
}

function computeTopLabels() {
  if (datumId) {
    // label the top-ranked documents among those currently visible, so
    // isolating a band (e.g. Weak) still labels its members
    topLabelIds = new Set(rankedIds.filter(id => nodeVisible(nodeMap[id])).slice(0, 18));
  } else {
    // if filters have narrowed the view to a small set, label all of it
    const vis = nodes3d.filter(n => nodeVisible(n));
    if (vis.length <= 25) {
      topLabelIds = new Set(vis.map(n => n.id));
    } else {
      const sorted = [...nodes3d].sort((a, b) => b.metricVal - a.metricVal);
      topLabelIds = new Set(sorted.slice(0, 14).map(n => n.id));
    }
  }
}

function nodeDimmed3D(n) {
  if (n.id === datumId || n.id === selectedDoc) return false;
  if (!nodeVisible(n)) return true;
  if (!datumId && selectedDoc && !cachedConn.has(n.id)) return true;
  return false;
}

function labelVisible3D(n) {
  if (n.id === datumId || n.id === selectedDoc) return true;
  if (!datumId && selectedDoc) return cachedConn.has(n.id) && nodeVisible(n);
  if (searchTerm) return nodeMatches(n) && passesFilter(n) && inRange(n);
  if (zoomLabelAll && !nodeDimmed3D(n)) return true;
  return topLabelIds.has(n.id) && !nodeDimmed3D(n);
}

function makeNodeObject(n) {
  const group = new THREE.Group();
  const isDatum = n.id === datumId;
  const dim = nodeDimmed3D(n);
  const color = isDatum ? pal().accent : (dim ? pal().nodeDim : typeColor(n.type));
  const r = isDatum ? 10 : Math.max(3, n.r * 0.55);

  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(r, 18, 12),
    new THREE.MeshLambertMaterial({ color, transparent: true, opacity: dim ? 0.05 : (isDatum ? 0.42 : 0.30), depthWrite: false })
  );
  group.add(sphere);

  const ringColor = isDatum ? pal().accent : (dim ? pal().nodeDim : nodeRingColor(n.id));
  const ring = new THREE.Sprite(new THREE.SpriteMaterial({
    map: ringTexture(ringColor), transparent: true,
    opacity: dim ? 0.2 : 1, depthWrite: false,
  }));
  // Sized as a clear outer orbit (~1.14x the sphere radius): sprites billboard
  // to the camera plane, not the camera point, so a ring hugging the sphere
  // drifts off its silhouette for off-centre nodes. The extra clearance keeps
  // sphere and ring visually concentric in every orientation.
  const d = r * 2.6;
  ring.scale.set(d, d, 1);
  group.add(ring);

  if (!dim && labelVisible3D(n)) {
    const s = n.id.replace(/_/g, ' ');
    const emphasised = isDatum || n.id === selectedDoc;
    const lbl = textTexture(
      s.length > 20 ? s.slice(0, 18) + '..' : s,
      emphasised ? pal().labelSel : pal().label,
      emphasised ? pal().accent : pal().labelBorder
    );
    const labelH = isDatum ? 6.2 : 5.2;
    const label = new THREE.Sprite(new THREE.SpriteMaterial({
      map: lbl.tex, transparent: true, depthWrite: false,
      depthTest: false,   // always legible in front of spheres and links
      fog: false,         // labels stay readable at any zoom distance
    }));
    label.renderOrder = 999;
    label.scale.set(labelH * lbl.aspect, labelH, 1);
    label.position.set(0, -(r + 5), 0);
    group.add(label);
  }
  return group;
}

const BAND_ALPHA_3D = { strong: 1.0, moderate: 0.78, weak: 0.5, sub: 0.14 };

function zoomAlpha(base, band) {
  // links recede as the camera pulls back so they never blanket the overview;
  // strong links fade the least, weak the most
  const k = Math.min(1, 0.2 + 0.8 * linkWidthScale);
  const floor = { strong: 0.45, moderate: 0.3, weak: 0.2, sub: 0.1 }[band] || 0.2;
  return Math.max(base * floor, base * k);
}

function link3DColor(l) {
  const s = l.source.id || l.source, t = l.target.id || l.target;
  if (datumId) {
    if (selectedDoc && t === selectedDoc) return pal().accent;
    if (!nodeVisible(nodeMap[t] || {})) return pal().linkSub;
    const band = strengthOf(l.avg);
    const base = { strong: 1.0, moderate: 0.72, weak: 0.42, sub: 0.1 }[band];
    return colorWithAlpha(strengthColor(band), zoomAlpha(base, band).toFixed(3));
  }
  if (selectedDoc) {
    return (s === selectedDoc || t === selectedDoc) ? pal().accent : pal().linkSub;
  }
  if (hasFilter() && !(nodeVisible(nodeMap[s] || {}) && nodeVisible(nodeMap[t] || {}))) {
    return pal().linkSub;
  }
  const band = strengthOf(l.avg);
  return colorWithAlpha(strengthColor(band), zoomAlpha(BAND_ALPHA_3D[band], band).toFixed(3));
}

let guideSpheres = [];
const _guideRingCache = {};
function guideRingTexture(color) {
  // thinner ring than the node rings — a crisp shell boundary line
  if (_guideRingCache[color]) return _guideRingCache[color];
  const c = document.createElement('canvas');
  c.width = c.height = 512;
  const ctx = c.getContext('2d');
  ctx.strokeStyle = color; ctx.lineWidth = 7;
  ctx.beginPath(); ctx.arc(256, 256, 248, 0, Math.PI * 2); ctx.stroke();
  const tex = new THREE.CanvasTexture(c);
  _guideRingCache[color] = tex;
  return tex;
}
function updateGuideSpheres() {
  if (!graph3D) return;
  const scene = graph3D.scene();
  guideSpheres.forEach(g => scene.remove(g));
  guideSpheres = [];
  if (!datumId) return;
  [['strong', S.strong], ['moderate', S.moderate], ['weak', S.weak]].forEach(([band, v]) => {
    if (v > simMax || v < simMin) return;
    const r = radiusOf(v) * R3_SCALE;
    const color = strengthColor(band);
    // faint wireframe shell in the band colour
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(r, 32, 20),
      new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 0.05 })
    );
    scene.add(mesh);
    guideSpheres.push(mesh);
    // solid camera-facing ring marking the shell boundary, in the band colour
    const ring = new THREE.Sprite(new THREE.SpriteMaterial({
      map: guideRingTexture(color), transparent: true, opacity: 0.85, depthWrite: false,
    }));
    ring.userData.guideR = r;
    ring.scale.set(r * 2.06, r * 2.06, 1);
    scene.add(ring);
    guideSpheres.push(ring);
  });
  updateGuideRingScales();
}

function updateGuideRingScales() {
  // A billboard ring at the shell's centre plane appears smaller than the
  // sphere's silhouette under perspective; scale by d/sqrt(d^2 - r^2) so the
  // ring sits exactly on the visible outer limit of each threshold sphere.
  if (!graph3D || !datumId) return;
  const d = graph3D.camera().position.length();
  guideSpheres.forEach(obj => {
    const r = obj.userData && obj.userData.guideR;
    if (!r) return;
    const k = d > r * 1.05 ? d / Math.sqrt(d * d - r * r) : 3;
    const s = r * 2.06 * Math.min(3, k);
    obj.scale.set(s, s, 1);
  });
}

function links3DData() {
  if (datumId) {
    return rankedIds.map(id => ({ source: datumId, target: id, avg: SIMS[id] }));
  }
  return getEdges().map(e => ({ source: e.s, target: e.t, avg: e.avg }));
}

// Corpus-size-aware layout: repulsion that balances at ~95 documents
// inflates the cloud at larger counts, so both forces scale with N to hold
// the overview at roughly constant visual density. At 95 docs both = 1.
const SIZE_REF = 95;
const sizeDistK = Math.sqrt(SIZE_REF / Math.max(1, DATA.docs.length));
const sizeChargeK = SIZE_REF / Math.max(1, DATA.docs.length);

function configure3DForces() {
  if (!graph3D) return;
  if (datumId) {
    graph3D.d3Force('charge').strength(-25);
    graph3D.d3Force('link').distance(l => radiusOf(l.avg) * R3_SCALE).strength(1);
  } else {
    const k = (compact3D ? 0.8 : 1) * sizeDistK;
    graph3D.d3Force('charge').strength((compact3D ? -95 : -130) * sizeChargeK);
    graph3D.d3Force('link')
      .distance(l => Math.max(50 * sizeDistK, (1 - l.avg) * 300 * k))
      .strength(compact3D ? 0.3 : 0.25);
  }
}

function init3D() {
  refreshRadiiForMode();
  recomputeNodeBands();
  computeTopLabels();
  updatePerfMode();
  graph3D = ForceGraph3D()(el3d)
    .width(el3d.clientWidth)
    .height(el3d.clientHeight)
    .graphData({ nodes: nodes3d, links: links3DData() })
    .backgroundColor(pal().bg3d)
    .nodeLabel(n => {
      let extra;
      if (datumId) {
        extra = n.id === datumId ? 'Reference datum'
          : `Similarity to datum: ${SIMS[n.id].toFixed(3)} (${strengthOf(SIMS[n.id])})`;
      } else {
        extra = `${MODE_LABELS[sizingMode]}: ${metricDisplay(n)}`;
      }
      return `<div style="background:var(--bg-card);padding:10px 14px;border-radius:8px;border:1px solid var(--border-hover);font-size:12px;max-width:260px;color:var(--text-primary);box-shadow:0 8px 32px var(--shadow);font-family:'DM Sans',sans-serif">
        <div style="font-weight:600;margin-bottom:4px">${n.title}</div>
        <div style="color:var(--text-secondary);font-size:11px">${TYPE_LABELS[n.type]} &middot; ${n.segs} segments</div>
        <div style="color:${pal().accent};font-size:11px;margin-top:4px">${extra}</div>
      </div>`;
    })
    .nodeThreeObject(n => makeNodeObject(n))
    .linkResolution(4)
    .linkWidth(l => {
      if (perfLines) return 0;   // GL line primitives — near-free at any count
      const s = l.source.id || l.source, t = l.target.id || l.target;
      if (selectedDoc && (s === selectedDoc || t === selectedDoc)) {
        return Math.max(1.0, (l.avg - 0.5) * 6 * Math.max(0.6, linkWidthScale));
      }
      return Math.max(0.15, (l.avg - 0.5) * (datumId ? 4.5 : 5.5) * linkWidthScale);
    })
    .linkColor(l => link3DColor(l))
    .linkCurvature((datumId || perfLines) ? 0 : 0.08)
    .linkOpacity(1)
    .onNodeClick(n => { if (n.id !== datumId) { selectedDoc = selectedDoc === n.id ? null : n.id; refreshAll(); } })
    .onNodeDrag(n => {
      // the datum is the fixed reference point — keep it pinned at the origin
      if (n.id === datumId) { n.fx = 0; n.fy = 0; n.fz = 0; n.x = 0; n.y = 0; n.z = 0; }
    })
    .onNodeDragEnd(n => {
      if (n.id === datumId) { n.fx = 0; n.fy = 0; n.fz = 0; n.x = 0; n.y = 0; n.z = 0; }
    })
    .onBackgroundClick(() => { if (selectedDoc) { selectedDoc = null; refreshAll(); } })
    .onEngineStop(() => {
      if (!didFit3D) { didFit3D = true; graph3D.zoomToFit(600, 70); }
      setTimeout(applyZoomLimits, 700);   // after the zoom-to-fit animation
    });

  configure3DForces();
  try { graph3D.scene().fog = new THREE.Fog(pal().bg3d, 280, 1300); } catch (e) {}
  updateGuideSpheres();
  try {
    graph3D.controls().addEventListener('change', updateLinkWidthScale);
  } catch (e) {}
  updateLinkWidthScale();
}

function update3D() {
  if (!graph3D) return;
  refreshRadiiForMode();
  recomputeNodeBands();
  computeTopLabels();
  if (datumId) { compact3D = false; updateOverviewCaption(); }   // datum shells need stable geometry
  setTimeout(applyZoomLimits, 400);
  updatePerfMode();
  graph3D.graphData({ nodes: nodes3d, links: links3DData() });
  configure3DForces();
  graph3D
    .backgroundColor(pal().bg3d)
    .nodeThreeObject(n => makeNodeObject(n))
    .linkCurvature((datumId || perfLines) ? 0 : 0.08)
    .linkColor(l => link3DColor(l))
    .linkWidth(graph3D.linkWidth());
  // rebuilt node objects come back at scale 1 — reapply the zoom scale
  _nodeScale = 1;
  setTimeout(() => { try { applyNodeScale(graph3D.camera().position.length()); } catch (e) {} }, 80);
  updateOverviewCaption();
  try { graph3D.scene().fog = new THREE.Fog(pal().bg3d, 280, 1300); } catch (e) {}
  updateGuideSpheres();
}

// ═══════════════════════════ HEATMAP (corpus mode) ═════════
let hmBuilt = false;
let hmCellEls = [], hmRowEls = [], hmRowHeadEls = [], hmColHeadEls = [];
const hmGrid = document.getElementById('grid');
const hmTooltip = document.getElementById('hm-tooltip');
const hmN = HM.ids.length;

function hmShortId(id) {
  const s = id.replace(/_/g, ' ');
  return s.length > 26 ? s.slice(0, 24) + '..' : s;
}
function hmCellColor(v) {
  if (v >= tMin && v <= tMax && v >= S.weak) {
    const band = strengthOf(v);
    const t = Math.min(1, Math.max(0, (v - S.weak) / (1 - S.weak)));
    return colorWithAlpha(strengthColor(band), +(0.30 + 0.70 * t).toFixed(2));
  }
  return v >= tMin && v <= tMax ? 'var(--tier-low)' : 'var(--tier-empty)';
}

function hmBuild() {
  let h = '<table><thead><tr><th class="corner"></th>';
  HM.ids.forEach((id, j) => {
    const d = nodeMap[id];
    h += `<th class="col-head" data-c="${j}" title="${d.title.replace(/"/g,'&quot;')}"><span>${hmShortId(id)}</span><div class="col-dot" style="background:${typeColor(d.type)}"></div></th>`;
  });
  h += '</tr></thead><tbody>';
  let lastType = null;
  HM.ids.forEach((id, i) => {
    const d = nodeMap[id];
    if (d.type !== lastType) {
      lastType = d.type;
      const count = HM.ids.filter(x => nodeMap[x].type === d.type).length;
      h += `<tr class="group-row" data-type="${d.type}"><th style="color:${typeColor(d.type)}" colspan="${hmN + 1}">${TYPE_LABELS[d.type]} (${count})</th></tr>`;
    }
    h += `<tr data-r="${i}" data-type="${d.type}"><th style="border-left-color:${typeColor(d.type)}" data-r="${i}" title="${d.title.replace(/"/g,'&quot;')}">${hmShortId(id)}</th>`;
    for (let j = 0; j < hmN; j++) h += `<td class="cell" data-r="${i}" data-c="${j}"></td>`;
    h += '</tr>';
  });
  h += '</tbody></table>';
  hmGrid.innerHTML = h;

  hmCellEls = []; hmRowEls = []; hmRowHeadEls = [];
  hmColHeadEls = [...hmGrid.querySelectorAll('.col-head')];
  hmGrid.querySelectorAll('tbody tr[data-r]').forEach(tr => {
    const i = +tr.dataset.r;
    hmRowEls[i] = tr;
    hmRowHeadEls[i] = tr.querySelector('th');
    hmCellEls[i] = [...tr.querySelectorAll('td.cell')];
  });
  hmBuilt = true;
  hmRecolor(); hmApplyFilters(); hmHighlightRow();
}

function hmRecolor() {
  if (!hmBuilt) return;
  for (let i = 0; i < hmN; i++) {
    const row = hmCellEls[i];
    const mi = HM.m[i];
    for (let j = 0; j < hmN; j++) {
      row[j].style.background = i === j ? 'var(--tier-empty)' : hmCellColor(mi[j]);
    }
  }
}

function hmApplyFilters() {
  if (!hmBuilt) return;
  let visible = 0;
  HM.ids.forEach((id, i) => {
    const d = nodeMap[id];
    const domOk = !filterType || d.type === filterType;
    const searchOk = nodeMatches(d);
    hmRowEls[i].classList.toggle('hidden-row', !domOk);
    hmRowEls[i].classList.toggle('dim-row', domOk && !searchOk);
    if (domOk && searchOk) visible++;
  });
  hmGrid.querySelectorAll('.group-row').forEach(gr => {
    gr.classList.toggle('hidden-row', !!(filterType && gr.dataset.type !== filterType));
  });
  hmColHeadEls.forEach((th, j) => {
    th.classList.toggle('dim-col', !!(searchTerm && !nodeMatches(nodeMap[HM.ids[j]])));
  });
  if (viewMode === '3rd' && !datumId && (searchTerm || filterType)) {
    document.getElementById('search-count').textContent = `${visible} of ${hmN} rows shown`;
  }
}

function hmHighlightRow() {
  if (!hmBuilt) return;
  hmRowHeadEls.forEach((th, i) => th.classList.toggle('sel', HM.ids[i] === selectedDoc));
}

hmGrid.addEventListener('click', (e) => {
  const td = e.target.closest('td.cell');
  const th = e.target.closest('tbody th[data-r]');
  let id = null;
  if (td) id = HM.ids[+td.dataset.r];
  else if (th) id = HM.ids[+th.dataset.r];
  if (!id) return;
  selectedDoc = selectedDoc === id ? null : id;
  refreshAll();
});
hmGrid.addEventListener('mouseover', (e) => {
  const td = e.target.closest('td.cell');
  if (!td) { hmTooltip.classList.remove('show'); return; }
  const i = +td.dataset.r, j = +td.dataset.c;
  if (i === j) { hmTooltip.classList.remove('show'); return; }
  const v = HM.m[i][j];
  const band = strengthOf(v);
  const bandLabel = { strong: 'Strong', moderate: 'Moderate', weak: 'Weak', sub: 'Below weak cutoff' }[band];
  const bandColor = band === 'sub' ? 'var(--text-muted)' : strengthColor(band);
  const a = nodeMap[HM.ids[i]], b = nodeMap[HM.ids[j]];
  hmTooltip.innerHTML = `
    <div class="tt-band" style="color:${bandColor}">${bandLabel} &middot; ${v.toFixed(3)}</div>
    <div class="tt-docs">${a.title}<br>&harr; ${b.title}</div>
    <div class="tt-meta">${TYPE_LABELS[a.type]} &times; ${TYPE_LABELS[b.type]} &middot; click to inspect</div>`;
  hmTooltip.classList.add('show');
});
hmGrid.addEventListener('mousemove', (e) => {
  const pad = 14;
  let x = e.clientX + pad, y = e.clientY + pad;
  const r = hmTooltip.getBoundingClientRect();
  if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - pad;
  hmTooltip.style.left = x + 'px';
  hmTooltip.style.top = y + 'px';
});
hmGrid.addEventListener('mouseleave', () => hmTooltip.classList.remove('show'));

// ═══════════════════════════ RANKING (datum mode) ══════════
const rankEl = document.getElementById('rank');
let rankBuiltFor = null;
let rankRowEls = {};

function rankBuild() {
  let h = `<table class="rank-table"><thead><tr>
    <th>#</th><th>Document</th><th>Domain</th><th>Similarity to datum</th>
  </tr></thead><tbody>`;
  rankedIds.forEach((id, i) => {
    const d = nodeMap[id];
    const v = SIMS[id];
    const band = strengthOf(v);
    const bc = band === 'sub' ? 'var(--text-muted)' : strengthColor(band);
    const barW = Math.max(2, Math.round(((v - simMin) / (simMax - simMin || 1)) * 100));
    h += `<tr class="r-row" data-id="${id}">
      <td class="r-rank">${i + 1}</td>
      <td><div class="r-doc"><span class="r-dot" style="background:${typeColor(d.type)}"></span>
        <div><div class="r-title">${d.title}</div><div class="r-id">${id.replace(/_/g,' ')}</div></div></div></td>
      <td class="r-dom"><span class="tag" style="background:${typeColor(d.type)}20;color:${typeColor(d.type)}">${TYPE_LABELS[d.type]}</span></td>
      <td class="r-sim"><div class="r-sim-inner">
        <span class="r-val" style="color:${bc}">${v.toFixed(3)}</span>
        <div class="r-bar"><div class="r-bar-fill" style="width:${barW}%;background:${bc}"></div></div>
      </div></td>
    </tr>`;
  });
  h += '</tbody></table>';
  rankEl.innerHTML = h;
  rankRowEls = {};
  rankEl.querySelectorAll('tr.r-row').forEach(tr => {
    rankRowEls[tr.dataset.id] = tr;
    tr.onclick = () => { const id = tr.dataset.id; selectedDoc = selectedDoc === id ? null : id; refreshAll(); };
  });
  rankBuiltFor = datumId;
  rankApply();
}

function rankApply() {
  if (rankBuiltFor !== datumId) return;
  let visible = 0;
  rankedIds.forEach(id => {
    const d = nodeMap[id];
    const show = passesFilter(d) && inRange(d);
    const match = nodeMatches(d);
    const tr = rankRowEls[id];
    if (!tr) return;
    tr.classList.toggle('hidden-row', !show);
    tr.classList.toggle('dim-row', show && !match);
    tr.classList.toggle('sel', id === selectedDoc);
    if (show && match) visible++;
  });
  if (viewMode === '3rd' && datumId && (searchTerm || filterType || tMin > 0.30 || tMax < 1.00)) {
    document.getElementById('search-count').textContent = `${visible} of ${rankedIds.length} documents shown`;
  }
}

function show3rd() {
  document.getElementById('grid-wrap').style.display = datumId ? 'none' : 'block';
  document.getElementById('rank-wrap').style.display = datumId ? 'block' : 'none';
  if (datumId) {
    if (rankBuiltFor !== datumId) rankBuild(); else rankApply();
  } else {
    if (!hmBuilt) hmBuild(); else { hmRecolor(); hmApplyFilters(); hmHighlightRow(); }
  }
}

// ═══════════════════════════ SHARED UI ═════════════════════
function setView(mode) {
  viewMode = mode;
  document.getElementById('graph-2d').style.display = mode === '2d' ? 'block' : 'none';
  document.getElementById('graph-3d').style.display = mode === '3d' ? 'block' : 'none';
  document.getElementById('view-3rd').style.display = mode === '3rd' ? 'block' : 'none';
  document.getElementById('btn-2d').classList.toggle('active', mode === '2d');
  document.getElementById('btn-3d').classList.toggle('active', mode === '3d');
  document.getElementById('btn-3rd').classList.toggle('active', mode === '3rd');
  updatePanelVisibility();

  if (mode === '3d') {
    if (!graph3D) { init3D(); } else { graph3D.resumeAnimation(); }
    graph3D.width(el3d.clientWidth).height(el3d.clientHeight);
    update3D();
    updateOverviewCaption();
  } else if (mode === '3rd') {
    if (graph3D) graph3D.pauseAnimation();
    show3rd();
    updateOverviewCaption();
  } else {
    if (graph3D) graph3D.pauseAnimation();
    render2D();
    updateOverviewCaption();
  }
  updateStats();
  renderDetailPanel();
}

function refreshAll() {
  updateClearBtn();
  recomputeConn();
  if (viewMode === '2d') { applySelection2D(); }
  else if (viewMode === '3d') { update3D(); }
  else { if (datumId) rankApply(); else { hmApplyFilters(); hmHighlightRow(); } }
  updateStats();
  renderDetailPanel();
}

function refreshStructure() {
  updateClearBtn();
  recomputeConn();
  if (viewMode === '2d') { render2D(); }
  else if (viewMode === '3d') { update3D(); }
  else { if (datumId) { rankApply(); } else { hmRecolor(); hmApplyFilters(); hmHighlightRow(); } }
  updateStats();
  renderDetailPanel();
}

function onThemeChange() {
  buildDomainLegend();
  buildStrengthBands();
  if (hmBuilt) hmBuild();
  if (rankBuiltFor === datumId && datumId) rankBuild();
  refreshStructure();
}

function updateNodeSizing(mode) {
  sizingMode = mode;
  document.getElementById('sizing-hint').textContent = SIZING_HINTS[mode] || '';
  highlightActiveMetric(mode);
  if (viewMode === '2d') {
    refreshRadiiForMode();
    nodeGroup.selectAll('g.node').select('circle')
      .transition().duration(500).ease(d3.easeCubicOut)
      .attr('r', d => d.r);
    nodeGroup.selectAll('g.node').select('text')
      .transition().duration(500).ease(d3.easeCubicOut)
      .attr('dy', d => d.r + 14);
    sim.force('collide', d3.forceCollide(d => d.r + 12));
    sim.alpha(0.3).restart();
  } else if (viewMode === '3d' && graph3D) {
    refreshRadiiForMode();
    computeTopLabels();
    graph3D.nodeThreeObject(n => makeNodeObject(n));
  }
  renderDetailPanel();
}

function toggleMetricLegend() {
  const items = document.getElementById('metric-items');
  const arrow = document.getElementById('metric-toggle-arrow');
  const label = document.getElementById('metric-toggle-label');
  const isOpen = items.style.display !== 'none';
  items.style.display = isOpen ? 'none' : 'block';
  arrow.classList.toggle('open', !isOpen);
  label.textContent = isOpen ? 'What do these mean?' : 'Hide descriptions';
  if (!isOpen) highlightActiveMetric(sizingMode);
}

function highlightActiveMetric(mode) {
  document.querySelectorAll('.metric-item').forEach(el => {
    el.classList.toggle('active', el.dataset.mode === mode);
  });
}

// ── Panels ─────────────────────────────────────────────────
function togglePanel(head) {
  head.parentElement.classList.toggle('closed');
  const closed = [...document.querySelectorAll('.panel.closed')].map(p => p.dataset.panel);
  localStorage.setItem('kg-panels-closed', JSON.stringify(closed));
}
(function(){
  try {
    JSON.parse(localStorage.getItem('kg-panels-closed') || '[]').forEach(k => {
      const p = document.querySelector(`.panel[data-panel="${k}"]`);
      if (p) p.classList.add('closed');
    });
  } catch(e) {}
})();

// ── Sliders ────────────────────────────────────────────────
const minSlider = document.getElementById('thresh-min');
const maxSlider = document.getElementById('thresh-max');
function syncSliderUI() {
  minSlider.value = Math.round(tMin * 100);
  maxSlider.value = Math.round(tMax * 100);
  document.getElementById('thresh-min-val').textContent = tMin.toFixed(2);
  document.getElementById('thresh-max-val').textContent = tMax.toFixed(2);
}
minSlider.addEventListener('input', function() {
  tMin = this.value / 100;
  if (tMin > tMax) tMax = tMin;
  activeBand = null; highlightBand();
  syncSliderUI();
  refreshStructure();
});
maxSlider.addEventListener('input', function() {
  tMax = this.value / 100;
  if (tMax < tMin) tMin = tMax;
  activeBand = null; highlightBand();
  syncSliderUI();
  refreshStructure();
});

// ── Strength bands ─────────────────────────────────────────
const BANDS = {
  strong:   { label: `Strong \\u2265 ${S.strong.toFixed(2)}`,                              lo: S.strong,   hi: 1.00 },
  moderate: { label: `Moderate ${S.moderate.toFixed(2)}\\u2013${S.strong.toFixed(2)}`,     lo: S.moderate, hi: S.strong },
  weak:     { label: `Weak ${S.weak.toFixed(2)}\\u2013${S.moderate.toFixed(2)}`,           lo: S.weak,     hi: S.moderate },
};
function setBand(band) {
  if (activeBand === band) {
    activeBand = null; tMin = defaultMin(); tMax = 1.00;
  } else {
    activeBand = band;
    tMin = Math.round(BANDS[band].lo * 100) / 100;
    tMax = band === 'strong' ? 1.00 : Math.round(BANDS[band].hi * 100) / 100;
  }
  highlightBand(); syncSliderUI(); refreshStructure();
}
function highlightBand() {
  document.querySelectorAll('#strength-legend .tband').forEach(el => {
    el.classList.toggle('active', el.dataset.band === activeBand);
  });
}
function buildStrengthBands() {
  const wrap = document.getElementById('strength-legend');
  wrap.innerHTML = '';
  Object.entries(BANDS).forEach(([k, b]) => {
    let n;
    if (datumId) {
      n = rankedIds.filter(id => SIMS[id] >= b.lo && (k === 'strong' ? SIMS[id] <= b.hi : SIMS[id] < b.hi)).length;
    } else {
      n = DATA.edges.filter(e => e.avg >= b.lo && (k === 'strong' ? e.avg <= b.hi : e.avg < b.hi)).length;
    }
    const el = document.createElement('div');
    el.className = 'tband';
    el.dataset.band = k;
    el.innerHTML = `<span class="tband-label"><span class="leg-line" style="background:${strengthColor(k)}"></span>${b.label}</span><span class="tband-count">${n}</span>`;
    el.onclick = () => setBand(k);
    wrap.appendChild(el);
  });
  highlightBand();
  const what = datumId ? 'documents in each band relative to the datum' : 'graph edges in each band';
  document.getElementById('strength-hint').textContent =
    `Counts = ${what}. Bands from the corpus similarity distribution (n=${S.n_pairs} pairs): \\u03BC=${S.mean.toFixed(2)}, \\u03C3=${S.std.toFixed(2)} \\u2192 \\u03BC+1\\u03C3 / +1.5\\u03C3 / +2\\u03C3. Click a band to isolate it. Node rings are coloured by each document's strongest connection in the current range (grey = none); fills keep the domain colour.`;
}

// ── Domain legend ──────────────────────────────────────────
function buildDomainLegend() {
  const legend = document.getElementById('legend');
  legend.innerHTML = '';
  Object.entries(TYPE_LABELS).forEach(([k, v]) => {
    const el = document.createElement('div');
    el.className = 'leg' + (filterType === k ? ' active' : '');
    el.innerHTML = `<div class="leg-dot" style="background:${typeColor(k)}"></div>${v}`;
    el.onclick = () => {
      filterType = filterType === k ? null : k;
      document.querySelectorAll('#legend .leg').forEach(x => x.classList.remove('active'));
      if (filterType) el.classList.add('active');
      updateSearchCount();
      refreshAll();
    };
    legend.appendChild(el);
  });
}

// ── Search ─────────────────────────────────────────────────
const searchInput = document.getElementById('graph-search');
const searchDrop = document.getElementById('search-drop');
let hlIndex = -1, suggestions = [];
function buildSuggestions(q) {
  const tokens = q.toLowerCase().split(/\\s+/).filter(Boolean);
  if (!tokens.length) return [];
  const scored = [];
  DATA.docs.forEach(d => {
    const id = d.id.replace(/_/g, ' ').toLowerCase();
    const title = (d.title || '').toLowerCase();
    let rank = -1;
    if (id.startsWith(tokens[0])) rank = 0;
    else if (tokens.some(t => id.includes(t))) rank = 1;
    else if (tokens.every(t => title.includes(t))) rank = 2;
    if (rank >= 0) scored.push({ d, rank });
  });
  scored.sort((a, b) => a.rank - b.rank || a.d.title.localeCompare(b.d.title));
  return scored.map(s => s.d);
}
function renderDrop() {
  if (!suggestions.length) { closeDrop(); return; }
  searchDrop.innerHTML = suggestions.slice(0, 12).map((d, i) => `
    <div class="sd-item${i === hlIndex ? ' hl' : ''}" data-id="${d.id}">
      <span class="sd-dot" style="background:${typeColor(d.type)}"></span>
      <span class="sd-text">${d.title}</span>
    </div>`).join('') +
    (suggestions.length > 12 ? `<div class="sd-more">${suggestions.length - 12} more\\u2026 keep typing</div>` : '');
  searchDrop.classList.add('open');
  searchDrop.querySelectorAll('.sd-item').forEach(el => {
    el.onclick = () => pickSuggestion(el.dataset.id);
  });
}
function closeDrop() { searchDrop.classList.remove('open'); hlIndex = -1; }
function pickSuggestion(id) {
  if (id !== datumId) selectedDoc = id;
  searchTerm = '';
  searchInput.value = nodeMap[id] ? nodeMap[id].title : id;
  updateSearchCount();
  closeDrop();
  refreshAll();
}
function updateSearchCount() {
  const el = document.getElementById('search-count');
  if (!searchTerm && !filterType) { el.textContent = ''; return; }
  if (viewMode === '3rd') return;  // the 3rd views set their own counts
  const pool = datumId ? DATA.docs.filter(d => d.id !== datumId) : DATA.docs;
  const n = pool.filter(d => passesFilter(d) && nodeMatches(d) && inRange(d)).length;
  el.textContent = `${n} of ${pool.length} documents match`;
}
searchInput.addEventListener('input', function() {
  searchTerm = this.value.trim().toLowerCase();
  suggestions = buildSuggestions(searchTerm);
  hlIndex = -1;
  renderDrop();
  updateSearchCount();
  refreshAll();
});
searchInput.addEventListener('keydown', function(e) {
  if (!searchDrop.classList.contains('open')) return;
  const max = Math.min(suggestions.length, 12) - 1;
  if (e.key === 'ArrowDown') { e.preventDefault(); hlIndex = Math.min(max, hlIndex + 1); renderDrop(); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); hlIndex = Math.max(0, hlIndex - 1); renderDrop(); }
  else if (e.key === 'Enter' && hlIndex >= 0) { e.preventDefault(); pickSuggestion(suggestions[hlIndex].id); }
  else if (e.key === 'Escape') { closeDrop(); }
});
document.addEventListener('click', (e) => {
  if (!e.target.closest('.search-wrap')) closeDrop();
});

// ── Clear filters ──────────────────────────────────────────
function clearFilters() {
  filterType = null; activeBand = null; searchTerm = '';
  searchInput.value = '';
  document.getElementById('search-count').textContent = '';
  closeDrop();
  document.querySelectorAll('#legend .leg').forEach(x => x.classList.remove('active'));
  tMin = defaultMin(); tMax = 1.00;
  highlightBand(); syncSliderUI(); refreshStructure();
}

// ── Stats ──────────────────────────────────────────────────
function updateStats() {
  let html;
  if (datumId) {
    const inR = rankedIds.filter(id => { const d = nodeMap[id]; return inRange(d) && passesFilter(d) && nodeMatches(d); }).length;
    const strong = rankedIds.filter(id => SIMS[id] >= S.strong).length;
    const vals = Object.values(SIMS).sort((a, b) => a - b);
    const median = vals[Math.floor(vals.length / 2)];
    html = `
      <div class="stat"><div class="stat-val">${rankedIds.length}</div><div class="stat-label">Docs vs datum</div></div>
      <div class="stat"><div class="stat-val">${inR}</div><div class="stat-label">In current view</div></div>
      <div class="stat"><div class="stat-val">${strong}</div><div class="stat-label">Strong alignments</div></div>
      <div class="stat"><div class="stat-val">${median.toFixed(2)}</div><div class="stat-label">Median similarity</div></div>`;
  } else {
    const edgeCount = getEdges().length;
    html = `
      <div class="stat"><div class="stat-val">${DATA.docs.length}</div><div class="stat-label">Documents</div></div>
      <div class="stat"><div class="stat-val">${edgeCount}</div><div class="stat-label">Visible edges</div></div>
      <div class="stat"><div class="stat-val">${DATA.docs.reduce((a,d)=>a+d.segs,0).toLocaleString()}</div><div class="stat-label">Total segments</div></div>
      <div class="stat"><div class="stat-val">${tMin.toFixed(2)}&ndash;${tMax.toFixed(2)}</div><div class="stat-label">Similarity range</div></div>`;
  }
  document.getElementById('stats').innerHTML = html;
}

// ── Detail panel ───────────────────────────────────────────
function renderDetailPanel() {
  const panel = document.getElementById('detail-panel');
  const sel = selectedDoc;

  if (!sel) {
    if (datumId) {
      panel.innerHTML = `<div class="sizing-hint" style="margin:6px 0 8px">Click a document to inspect its relationship with the datum. Top 10 most similar:</div>` +
        rankedIds.slice(0, 10).map((id, i) => {
          const d = nodeMap[id];
          const v = SIMS[id];
          return `<div class="conn-item">
            <span class="conn-name" onclick="selectedDoc='${id}';refreshAll();">${i+1}. ${d.title.slice(0, 36)}</span>
            <span class="conn-score" style="color:${strengthColor(strengthOf(v))}">${(v*100).toFixed(0)}%</span>
          </div>`;
        }).join('');
    } else {
      const clickHint = viewMode === '3rd' ? 'Click a cell or row label to inspect a document.' : 'Click a node to inspect a document.';
      panel.innerHTML = `<div class="sizing-hint" style="margin:6px 0 8px">${clickHint} Top paragraph-level overlaps across the corpus:</div>` +
        DATA.seg_pairs.slice(0, 10).map(p => `
          <div class="seg-pair">
            <div class="seg-pair-header">
              <span class="seg-pair-docs">${p.a.replace(/_/g,' ')} &harr; ${p.b.replace(/_/g,' ')}</span>
              <span class="seg-pair-count">${p.count} pairs</span>
            </div>
            ${p.examples && p.examples[0] ? `<div class="seg-pair-example">&ldquo;${p.examples[0].a_heading}&rdquo; &harr; &ldquo;${p.examples[0].b_heading}&rdquo; (sim: ${p.examples[0].sim})</div>` : ''}
          </div>`).join('');
    }
    return;
  }

  const doc = nodeMap[sel];
  if (!doc) return;
  const tc = typeColor(doc.type);

  let datumRows = '';
  let datumTag = '';
  if (datumId) {
    const v = SIMS[sel];
    const band = strengthOf(v);
    const bandLabel = { strong: 'Strong', moderate: 'Moderate', weak: 'Weak', sub: 'Below weak cutoff' }[band];
    const bc = band === 'sub' ? 'var(--text-muted)' : strengthColor(band);
    const rank = rankedIds.indexOf(sel) + 1;
    const segPair = DATA.seg_pairs.find(p => (p.a === datumId && p.b === sel) || (p.b === datumId && p.a === sel));
    datumTag = `<span class="tag" style="background:${strengthColor(band)}20;color:${bc}">${bandLabel} vs datum</span>`;
    datumRows = `
      <div class="doc-row"><span>Similarity to datum</span><span class="doc-row-val" style="color:${bc}">${v.toFixed(3)}</span></div>
      <div class="doc-row"><span>Rank vs datum</span><span class="doc-row-val">${rank} / ${rankedIds.length}</span></div>
      ${segPair ? `<div class="doc-row"><span>Segment pairs w/ datum</span><span class="doc-row-val">${segPair.count}</span></div>` : ''}`;
  }

  const conns = getEdges()
    .filter(e => e.s === sel || e.t === sel)
    .map(e => ({ id: e.s === sel ? e.t : e.s, sim: e.avg, band: strengthOf(e.avg) }))
    .sort((a, b) => b.sim - a.sim);
  // In datum mode getEdges() is range-filtered by sim-to-datum which is not
  // meaningful for corpus-wide neighbours — use the unfiltered edge list there.
  const connSource = datumId
    ? DATA.edges.filter(e => e.s === sel || e.t === sel)
        .map(e => ({ id: e.s === sel ? e.t : e.s, sim: e.avg, band: strengthOf(e.avg) }))
        .sort((a, b) => b.sim - a.sim)
    : conns;

  const segLinks = DATA.seg_pairs.filter(p => p.a === sel || p.b === sel);

  panel.innerHTML = `
    <div class="doc-card">
      <div class="doc-title">${doc.title}</div>
      <div class="doc-meta">
        <span class="tag" style="background:${tc}20;color:${tc}">${TYPE_LABELS[doc.type]}</span>
        ${datumTag}
      </div>
      ${datumRows}
      <div class="doc-row"><span>Segments</span><span class="doc-row-val">${doc.segs.toLocaleString()}</span></div>
      <div class="doc-row"><span>Words</span><span class="doc-row-val">${(doc.words/1000).toFixed(0)}k</span></div>
      <div class="doc-row"><span>Connections</span><span class="doc-row-val">${connSource.length}</span></div>
      <div class="doc-row"><span>Segment links</span><span class="doc-row-val">${doc.seg_links || 0}</span></div>
    </div>

    ${connSource.length ? `<div class="sizing-hint" style="margin:2px 0 4px">Similar documents (click to jump)</div>` +
      connSource.slice(0, 12).map(c => {
        const cd = nodeMap[c.id];
        if (!cd) return '';
        const isDatumDoc = c.id === datumId;
        const barW = Math.max(2, (c.sim - 0.5) * 100);
        return `<div class="conn-item">
          <span class="conn-name" onclick="${isDatumDoc ? '' : `selectedDoc='${c.id}';refreshAll();`}">${cd.title.slice(0, 38)}${isDatumDoc ? ' \\u2605' : ''}</span>
          <div class="conn-bar">
            <span class="conn-score" title="${c.band}" style="color:${strengthColor(c.band)}">${(c.sim * 100).toFixed(0)}%</span>
            <div class="bar"><div class="bar-fill" style="width:${barW}%;background:${typeColor(cd.type)}"></div></div>
          </div>
        </div>`;
      }).join('') : ''}`

    + (segLinks.length ? `<div class="sizing-hint" style="margin:10px 0 4px">Segment-level links</div>` +
      segLinks.map(p => `<div class="seg-pair">
        <div class="seg-pair-header">
          <span class="seg-pair-docs">&harr; ${(p.a === sel ? p.b : p.a).replace(/_/g, ' ')}</span>
          <span class="seg-pair-count">${p.count} pairs (max ${(p.max_sim*100).toFixed(0)}%)</span>
        </div>
      </div>`).join('') : '');
}

// ── Reset ──────────────────────────────────────────────────
function resetGraph() {
  selectedDoc = null;
  recomputeConn();
  if (viewMode === '2d') {
    svg.transition().duration(600).call(zoomBehavior.transform, d3.zoomIdentity);
    render2D();
    sim.alpha(0.3).restart();
  } else if (viewMode === '3d' && graph3D) {
    update3D();
    // return to the default framing: whole dataset (or all shells) in view
    setTimeout(() => { try { graph3D.zoomToFit(700, 70); } catch (e) {} }, 150);
  } else if (viewMode === '3rd') {
    const wrap = datumId ? document.getElementById('rank-wrap') : document.getElementById('grid-wrap');
    wrap.scrollTop = 0; wrap.scrollLeft = 0;
    if (datumId) rankApply(); else hmHighlightRow();
  }
  updateStats();
  renderDetailPanel();
}

__THEME_TOGGLE_JS__

// ── Init ───────────────────────────────────────────────────
document.getElementById('doc-count').textContent = DATA.docs.length;
updateThemeBtn();
buildDatumSelect();
buildDomainLegend();
buildStrengthBands();
syncSliderUI();
render2D();
updateStats();
renderDetailPanel();
highlightActiveMetric('doc_size');
window.addEventListener('resize', () => {
  if (graph3D && viewMode === '3d') {
    graph3D.width(el3d.clientWidth).height(el3d.clientHeight);
  }
});
</script>
</body>
</html>"""

    html = (html
            .replace('__DATA_JSON__', data_json)
            .replace('__LOGO_B64__', LOGO_B64)
            .replace('__THEME_CSS__', THEME_CSS)
            .replace('__THEME_JS__', THEME_JS)
            .replace('__THEME_TOGGLE_JS__', THEME_TOGGLE_JS)
            .replace('__FONTS_LINK__', FONTS_LINK)
            .replace('__MODEL_LABEL__', model_label))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Generated combined explorer (2D/3D/heatmap + datum modes) → {output_path}")


def build_heat_data(builder) -> dict:
    """Full doc×doc similarity matrix ordered by domain then id."""
    order = sorted(builder._docs.keys(), key=lambda d: (builder._docs[d]["doc_type"], d))
    idx = {did: i for i, did in enumerate(order)}
    n = len(order)
    matrix = [[0.0] * n for _ in range(n)]
    for (a, b), v in builder._doc_sim_matrix.items():
        i, j = idx[a], idx[b]
        matrix[i][j] = matrix[j][i] = round(v["avg"], 3)
    for i in range(n):
        matrix[i][i] = 1.0
    return {"ids": order, "m": matrix}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Policy KG Explorer")
    parser.add_argument("--segments", required=True, help="Path to all_segments.json")
    parser.add_argument("--emb", action="append", default=[],
                        help="Embedding model as name=path (repeatable), e.g. "
                             "--emb bge_m3=embeddings_bge_m3.npy --emb qwen3=... --emb minilm=... "
                             "Similarities are computed per model and averaged.")
    parser.add_argument("--bge", default="", help="(legacy) path to BGE embeddings .npy")
    parser.add_argument("--minilm", default="", help="(legacy) path to MiniLM embeddings .npy")
    parser.add_argument("--output", default="policy_graph.html", help="Output HTML path")
    parser.add_argument("--doc-threshold", type=float, default=0.60)
    parser.add_argument("--seg-threshold", type=float, default=0.78)
    parser.add_argument("--top-k-segments", type=int, default=300)
    parser.add_argument("--datum-options", default="",
                        help="Comma-separated doc IDs for the curated datum shortlist "
                             "(defaults to the built-in list; any doc remains selectable)")
    parser.add_argument("--skip-exports", action="store_true",
                        help="Skip JSON/GraphML/CSV exports (HTML only)")
    args = parser.parse_args()

    output_dir = Path(args.output).parent
    stem = Path(args.output).stem

    with open(args.segments, encoding="utf-8") as f:
        segments = json.load(f)

    embeddings = {}
    if args.emb:
        for spec in args.emb:
            if "=" not in spec:
                raise SystemExit(f"--emb expects name=path, got: {spec}")
            name, path = spec.split("=", 1)
            embeddings[name.strip()] = np.load(path.strip())
    else:
        if not (args.bge and args.minilm):
            raise SystemExit("Provide embeddings via --emb name=path (repeatable) "
                             "or legacy --bge/--minilm.")
        embeddings["bge"] = np.load(args.bge)
        embeddings["minilm"] = np.load(args.minilm)

    shapes = ", ".join(f"{n} {e.shape}" for n, e in embeddings.items())
    print(f"Loaded: {len(segments)} segments | embeddings: {shapes}")

    builder = PolicyGraphBuilder(
        segments, embeddings,
        doc_threshold=args.doc_threshold,
        seg_threshold=args.seg_threshold,
        top_k_segments=args.top_k_segments,
    )
    G = builder.build()

    if not args.skip_exports:
        builder.export_json(str(output_dir / f"{stem}_data.json"))
        builder.export_graphml(str(output_dir / f"{stem}.graphml"))
        builder.export_similarity_csv(str(output_dir / f"{stem}_similarity.csv"))

    viz_data = builder.get_viz_data()
    heat_data = build_heat_data(builder)
    datum_options = [d.strip() for d in args.datum_options.split(",") if d.strip()] or None
    LABELS = {"bge": "BGE-large", "bge_m3": "BGE-M3", "qwen3": "Qwen3-Embedding-0.6B",
              "minilm": "MiniLM"}
    pretty = " + ".join(LABELS.get(n, n) for n in embeddings)
    n_models = len(embeddings)
    model_label = f"{'dual' if n_models == 2 else 'tri' if n_models == 3 else str(n_models)}-model embeddings ({pretty})"
    generate_combined_html(viz_data, heat_data, args.output, datum_options=datum_options,
                           model_label=model_label)

    print(f"\nDone! Open {args.output} in your browser.")
    print("Views: 2D / 3D / Heatmap. Pick a reference datum in the sidebar to switch")
    print("into datum mode (radial layouts + ranking); any document can be the datum.")


if __name__ == "__main__":
    main()
