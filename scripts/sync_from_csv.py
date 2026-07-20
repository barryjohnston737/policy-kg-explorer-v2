#!/usr/bin/env python3
"""
sync_from_csv.py — master_library.csv is the editing surface; this script
regenerates master_library.yaml and GAP_REVIEW.md from it after edits.
Run after any manual review pass.
"""

import csv
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def main():
    rows = list(csv.DictReader(open(REPO / "master_library.csv", encoding="utf-8-sig")))

    docs = {}
    for r in rows:
        rid = r["id"].strip()
        docs[rid] = {
            "title": r["title"],
            "jurisdiction": r["jurisdiction"],
            "domain": [d for d in (r["domain"] or "").split(";") if d],
            "year": int(r["year"]) if (r["year"] or "").strip().isdigit() else None,
            "legal_basis": r["legal_basis"],
            "status": r["status"] or "current",
            "source_url": r["source_url"],
            "source_org": r["source_org"],
            "in_atlas": r["in_atlas"] == "yes",
            "atlas_download": r["atlas_download"],
            "in_kg": r["in_kg"] == "yes",
            "kg_words": int(r["kg_words"]) if (r["kg_words"] or "").strip().isdigit() else None,
            "kg_text_status": r["kg_text_status"],
            "review_flags": [f.strip() for f in (r["review_flags"] or "").split(";") if f.strip()],
        }

    out = {
        "library": {
            "name": "Unified Policy Document Library",
            "version": "1.1",
            "description": "Single canonical document library shared by the PolicyKit "
                           "Atlas and the Policy KG Explorer. Edit master_library.csv, "
                           "then run sync_from_csv.py.",
        },
        "documents": docs,
    }
    with open(REPO / "master_library.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True, width=100)

    no_year = [r for r in rows if not (r["year"] or "").strip()]
    no_url = [r for r in rows if not (r["source_url"] or "").strip()]
    pending = [r for r in rows if r["atlas_download"] == "pending"]
    poor = [r for r in rows if r["kg_text_status"] in ("extraction-poor", "excluded-no-text")]
    spot = [r for r in rows if any(f in (r["review_flags"] or "")
            for f in ("year-from-title", "year-from-filename"))]

    g = ["# Gap review — unified policy document library\n",
         f"{len(rows)} documents. Remaining review items after sign-off pass:\n",
         f"\n## 1. Years still unknown ({len(no_year)})\n"]
    for r in no_year: g.append(f"- `{r['id']}` — {r['title']}")
    g.append(f"\n## 2. Years auto-derived — spot-check ({len(spot)})\n")
    g.append("Taken from the document title or source filename; verify a sample:\n")
    for r in spot: g.append(f"- `{r['id']}` — {r['title'][:60]} → {r['year']}")
    g.append(f"\n## 3. Missing source URLs ({len(no_url)})\n")
    for r in sorted(no_url, key=lambda x: x["title"]): g.append(f"- {r['title']}")
    g.append(f"\n## 4. Atlas downloads pending ({len(pending)})\n")
    for r in sorted(pending, key=lambda x: x["title"]): g.append(f"- {r['title']} ({r['jurisdiction']})")
    g.append(f"\n## 5. Text-extraction problems ({len(poor)})\n")
    for r in poor: g.append(f"- {r['title']} ({r['kg_text_status']})")

    with open(REPO / "GAP_REVIEW.md", "w", encoding="utf-8") as f:
        f.write("\n".join(g) + "\n")

    print(f"synced: {len(rows)} docs -> master_library.yaml + GAP_REVIEW.md")
    print(f"remaining: years {len(no_year)} | spot-check years {len(spot)} | "
          f"URLs {len(no_url)} | pending {len(pending)} | extraction {len(poor)}")


if __name__ == "__main__":
    main()
