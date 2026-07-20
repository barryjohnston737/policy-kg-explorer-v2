#!/usr/bin/env python3
"""
merge_libraries.py — Build the unified policy document library.

Merges two sources (read-only snapshots in sources/):
  - atlas_manifest_snapshot.yaml   PolicyKit Atlas manifest (canonical schema:
                                   jurisdiction, domain[], year, legal_basis,
                                   status, source_url, source_org, ...)
  - kg_document_library_snapshot.csv  KG Explorer library (source URLs +
                                   text-quality audit: words, text_status)

Outputs:
  - master_library.yaml   canonical merged library (Atlas schema, extended)
  - master_library.csv    flat version for spreadsheets
  - GAP_REVIEW.md         everything needing human review before the corpus
                          is expanded: fuzzy matches to confirm, missing URLs,
                          pending downloads, extraction failures, suspected
                          superseded-version pairs, KG-only docs needing
                          jurisdiction/year metadata

Matching is by normalised title. Exact matches merge automatically; partial
(substring) matches merge but are flagged `confirm-match` in GAP_REVIEW.md.
Nothing in either source project is modified.
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ATLAS = REPO / "sources" / "atlas_manifest_snapshot.yaml"
KG = REPO / "sources" / "kg_document_library_snapshot.csv"

STOP = {"the", "of", "for", "and", "a", "an", "to", "in", "on", "ireland",
        "irelands", "irish", "en", "e"}


def norm(t):
    t = re.sub(r"[^a-z0-9 ]", " ", (t or "").lower())
    return " ".join(w for w in t.split() if w not in STOP)


def guess_jurisdiction(title):
    t = title.lower()
    if any(k in t for k in ("eu ", "european union", "directive", "european commission",
                            "regulation (e", "common fisheries", "cap strategic")):
        return "EU", True
    if any(k in t for k in ("convention", "marpol", "ospar", "cites", "united nations",
                            "un ", "aichi", "kyoto", "ramsar", "agenda 2030",
                            "migratory species", "biological diversity")):
        return "GL", True
    return "IE", True   # default for Irish policy corpus; always flag for review


def guess_legal_basis(title):
    t = title.lower()
    for pat, basis in [("directive", "eu_directive"), ("regulation", "regulation"),
                       (" act", "act"), ("s.i.", "statutory_instrument"),
                       ("convention", "treaty"), ("protocol", "treaty"),
                       ("strategy", "strategy"), ("plan", "plan"),
                       ("programme", "programme"), ("framework", "framework"),
                       ("order", "statutory_instrument"), ("bill", "bill"),
                       ("report", "report"), ("review", "review")]:
        if pat in t:
            return basis
    return "policy"


def main():
    atlas = yaml.safe_load(open(ATLAS, encoding="utf-8"))
    atlas_docs = atlas["documents"]
    kg_rows = list(csv.DictReader(open(KG, encoding="utf-8")))

    atlas_norm = {aid: norm(v.get("title", "")) for aid, v in atlas_docs.items()}

    # ── Build master entries from the Atlas (canonical base) ──
    master = {}
    for aid, v in atlas_docs.items():
        master[aid] = {
            "title": v.get("title", ""),
            "jurisdiction": v.get("jurisdiction", ""),
            "domain": v.get("domain", []),
            "year": v.get("year"),
            "legal_basis": v.get("legal_basis", ""),
            "status": v.get("status", "current"),
            "source_url": v.get("source_url", ""),
            "source_org": v.get("source_org", ""),
            "atlas": {
                "id": aid,
                "filename": v.get("filename", ""),
                "download_status": v.get("download_status", ""),
            },
            "kg": None,
            "review_flags": [],
        }

    # ── Match & merge KG entries ──
    exact, partial, unmatched = 0, 0, 0
    for r in kg_rows:
        nt = norm(r["title"])
        hit, kind = None, None
        for aid, at in atlas_norm.items():
            if nt and nt == at:
                hit, kind = aid, "exact"
                break
        if not hit:
            for aid, at in atlas_norm.items():
                if nt and at and len(nt) > 12 and (nt in at or at in nt):
                    hit, kind = aid, "partial"
                    break

        kg_block = {
            "id": r["doc_id"],
            "domain": r["domain"],
            "filename": r["filename"],
            "segments": int(r.get("segments") or 0),
            "words": int(r.get("words") or 0),
            "text_status": r.get("text_status", ""),
            "note": r.get("note", ""),
        }

        if hit:
            m = master[hit]
            m["kg"] = kg_block
            if kind == "partial":
                m["review_flags"].append(f"confirm-match: KG '{r['title']}' ~ Atlas '{m['title']}'")
                partial += 1
            else:
                exact += 1
            # KG library has verified URLs the Atlas may lack
            if not m["source_url"] and r["source_url"] not in ("", "TODO"):
                m["source_url"] = r["source_url"]
                m["review_flags"].append("url-from-kg-library")
        else:
            unmatched += 1
            jur, jflag = guess_jurisdiction(r["title"])
            nid = f"{jur}_{r['doc_id'][:40]}"
            master[nid] = {
                "title": r["title"],
                "jurisdiction": jur,
                "domain": [r["domain"]],
                "year": None,
                "legal_basis": guess_legal_basis(r["title"]),
                "status": "current",
                "source_url": r["source_url"] if r["source_url"] != "TODO" else "",
                "source_org": "",
                "atlas": None,
                "kg": kg_block,
                "review_flags": (["jurisdiction-guessed"] if jflag else []) + ["year-unknown", "new-from-kg"],
            }

    # ── Superseded-version candidates (same base title, different years) ──
    base_groups = defaultdict(list)
    for mid, m in master.items():
        base = re.sub(r"\b(19|20)\d{2}(\s*[-–]\s*(19|20)?\d{2,4})?\b", "", norm(m["title"])).strip()
        if base:
            base_groups[base].append(mid)
    superseded_candidates = {b: ids for b, ids in base_groups.items() if len(ids) > 1}

    # ── Write outputs ──
    out = {
        "library": {
            "name": "Unified Policy Document Library",
            "version": "1.0-draft",
            "description": "Single canonical document library shared by the PolicyKit "
                           "Atlas and the Policy KG Explorer. Merged from the Atlas "
                           "manifest (210 docs) and the KG Explorer library (96 docs).",
            "sources": {
                "atlas_manifest": "sources/atlas_manifest_snapshot.yaml",
                "kg_library": "sources/kg_document_library_snapshot.csv",
            },
        },
        "documents": master,
    }
    with open(REPO / "master_library.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True, width=100)

    with open(REPO / "master_library.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "jurisdiction", "domain", "year", "legal_basis",
                    "status", "source_url", "source_org", "in_atlas", "atlas_download",
                    "in_kg", "kg_words", "kg_text_status", "review_flags"])
        for mid, m in sorted(master.items()):
            w.writerow([
                mid, m["title"], m["jurisdiction"], ";".join(m["domain"]) if isinstance(m["domain"], list) else m["domain"],
                m["year"] or "", m["legal_basis"], m["status"], m["source_url"], m["source_org"],
                "yes" if m["atlas"] else "no",
                (m["atlas"] or {}).get("download_status", ""),
                "yes" if m["kg"] else "no",
                (m["kg"] or {}).get("words", ""),
                (m["kg"] or {}).get("text_status", ""),
                "; ".join(m["review_flags"]),
            ])

    # ── Gap review ──
    no_url = [m for m in master.values() if not m["source_url"]]
    pending = [m for m in master.values() if m["atlas"] and m["atlas"]["download_status"] == "pending"]
    poor = [m for m in master.values() if m["kg"] and m["kg"]["text_status"] in ("extraction-poor", "excluded-no-text")]
    confirms = [(mid, m) for mid, m in master.items()
                if any(f.startswith("confirm-match") for f in m["review_flags"])]
    kg_new = [(mid, m) for mid, m in master.items() if "new-from-kg" in m["review_flags"]]

    g = ["# Gap review — unified policy document library\n",
         f"Merged: {len(atlas_docs)} Atlas entries + {len(kg_rows)} KG entries "
         f"→ **{len(master)} unified documents** "
         f"({exact} exact matches, {partial} partial matches to confirm, {unmatched} KG-only additions).\n",
         "Work through the sections below, edit `master_library.csv` (or the YAML) directly, "
         "then re-run downstream builds. Nothing is consumed by either project until you sign off.\n"]

    g.append(f"\n## 1. Fuzzy matches to confirm ({len(confirms)})\n")
    g.append("These merged on partial title similarity — confirm each is genuinely the same document:\n")
    for mid, m in confirms:
        flag = next(f for f in m["review_flags"] if f.startswith("confirm-match"))
        g.append(f"- `{mid}`: {flag.replace('confirm-match: ', '')}")

    g.append(f"\n## 2. KG-only documents added ({len(kg_new)})\n")
    g.append("New to the unified library; jurisdiction was guessed and year is unknown — review both:\n")
    for mid, m in kg_new:
        g.append(f"- `{mid}` — {m['title']} (jurisdiction: {m['jurisdiction']}?, "
                 f"legal_basis: {m['legal_basis']}?)")

    g.append(f"\n## 3. Missing source URLs ({len(no_url)})\n")
    g.append("No provenance link; find and record the authoritative source:\n")
    for m in sorted(no_url, key=lambda x: x["title"]):
        g.append(f"- {m['title']}")

    g.append(f"\n## 4. Atlas downloads still pending ({len(pending)})\n")
    g.append("In the Atlas manifest but never downloaded/ingested:\n")
    for m in sorted(pending, key=lambda x: x["title"]):
        g.append(f"- {m['title']} ({m['jurisdiction']})")

    g.append(f"\n## 5. Text-extraction problems from the KG audit ({len(poor)})\n")
    for m in poor:
        g.append(f"- {m['title']}: {m['kg']['note']}")

    g.append(f"\n## 6. Suspected superseded-version pairs ({len(superseded_candidates)})\n")
    g.append("Same base title with different years/versions — mark older ones `status: superseded` "
             "or confirm both belong:\n")
    for base, ids in sorted(superseded_candidates.items()):
        g.append(f"- {', '.join(f'`{i}`' for i in ids)} — “{base}”")

    with open(REPO / "GAP_REVIEW.md", "w", encoding="utf-8") as f:
        f.write("\n".join(g) + "\n")

    # summary
    jur = Counter(m["jurisdiction"] for m in master.values())
    print(f"master library: {len(master)} documents")
    print(f"  matches: {exact} exact, {partial} to confirm | KG-only added: {unmatched}")
    print(f"  jurisdictions: {dict(jur)}")
    print(f"  missing URLs: {len(no_url)} | pending downloads: {len(pending)} | "
          f"extraction problems: {len(poor)} | version pairs to review: {len(superseded_candidates)}")


if __name__ == "__main__":
    main()
