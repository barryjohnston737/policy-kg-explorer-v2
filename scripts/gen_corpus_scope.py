#!/usr/bin/env python3
"""
gen_corpus_scope.py — Draft the KG Explorer corpus scope from the master library.

Scope rules (v1 proposal, for review):
  - Include all IE / EU / GL documents with status=current
  - Local (LO) tier: one representative climate action plan (Dublin City);
    all other county plans and local biodiversity plans excluded (they form a
    near-duplicate family that skews both the visual and the calibration)
  - Dedupe pairs resolved in favour of the entry that already has text
  - Every included document is tagged with where its text will come from:
    atlas-text / kg-segments / scraper-file / download-needed

Outputs: corpus_scope.csv (one row per master-library doc, include=yes/no + reason)
"""

import csv
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Optional external text sources, used to decide where each document's text will
# come from. Neither is required to run this script: if a directory is absent,
# documents that would have drawn on it are simply marked download-needed.
# Override either with an environment variable.
#
#   ATLAS_TEXT_DIR=/path/to/atlas/corpus/text  SCRAPING_DIR=/path/to/scraped  \
#       python3 scripts/gen_corpus_scope.py
#
ATLAS_TEXT = Path(os.environ.get(
    "ATLAS_TEXT_DIR", Path.home() / "Desktop" / "PolicyKit_Atlas_Backend" / "corpus" / "text"))
SCRAPING = Path(os.environ.get(
    "SCRAPING_DIR", Path.home() / "Desktop" / "scraping_dir"))

for label, p in (("ATLAS_TEXT_DIR", ATLAS_TEXT), ("SCRAPING_DIR", SCRAPING)):
    if not p.exists():
        print(f"note: {label} not found at {p} — "
              f"documents relying on it will be marked download-needed")

REPRESENTATIVE_LO = "LO_DUBLIN_CITY_CLIMATE"

# Local-tier condensation: near-identical families collapse into sampled
# composite documents (built at corpus-assembly time); unique local docs and
# the Dublin City exemplar stay individual.
LO_COMPOSITES = {
    "LO_CLIMATE_COMPOSITE": {
        "title": "Local Authority Climate Action Plans 2024\u20132029 (composite)",
        "domain": "climate",
        "members": lambda rid: "CLIMATE" in rid and rid != REPRESENTATIVE_LO,
    },
    "LO_BIODIV_COMPOSITE": {
        "title": "Local Authority Biodiversity Action Plans (composite)",
        "domain": "biodiversity",
        "members": lambda rid: "BIODIV" in rid,
    },
    "LO_CDP_COMPOSITE": {
        "title": "County Development Plans 2022\u20132028 (composite)",
        "domain": "cross_cutting",
        "members": lambda rid: rid.endswith("_CDP"),
    },
}
SEGMENTS_PER_MEMBER = 7   # even sampling per member document

# Dedupe decisions: dropped_id -> (kept_id, reason)
DEDUPE_DROPS = {
    "IE_DMAP_ORE": ("IE_DMAP_ORE_PROPOSAL", "same document; KG entry already has extracted text"),
    "IE_NCCRA": ("IE_EPA_NCCRA_MAIN_REPORT_PUBLISHED", "same document; KG entry already has extracted text"),
    "IE_OCEAN_WEALTH": ("IE_HARNESSINGOUROCEANWEALTHREPORT", "same document; KG entry already has extracted text"),
    "IE_FPM_FOREST_PLAN": ("IE_DRAFT_PLAN_FORESTS_FRESHWATER_PEARL", "same plan; KG entry already has extracted text"),
    "EU_MARINE_STRATEGY_FRAMEWORK_DIRECTIVE_P3": ("IE_MARINE_STRATEGY_P3", "same document, was mis-tiered as EU; keep the IE entry"),
    "IE_AMMONIA_CODE": ("IE_CODE_GOOD_AGRICULTURAL_PRACTICE_REDUCING", "same code; KG entry already has extracted text"),
    "IE_ARTERIAL_DRAIN_MAINT": ("IE_ARTERIAL_DRAINAGE_MAINTENANCE_PLAN_2025", "overlapping OPW maintenance documents; keep the 2025 plan with text"),
    "IE_FLOOD_RISK_GUIDE": ("IE_2009_PLANNING_SYSTEM_FLOOD_RISK", "same 2009 guidelines; KG entry already has extracted text"),
    "IE_MONUMENTS_ACT": ("IE_MONUMENTS_ACT_1930", "same act (see also extraction fix)"),
    "IE_COAST_PROTECT_ACT": ("IE_COAST_PROTECTION_ACT_1963", "same act (see also extraction fix)"),
    "IE_ARTERIAL_DRAIN_ACT": ("IE_ARTERIAL_DRAINAGE_AMENDMENT_ACT_1995", "1945+1995 acts; keep the KG 1995 entry, 1945 principal act optional"),
    "IE_HEN_HARRIER_PLAN": ("IE_HEN_HARRIER_THREAT_RESPONSE_PLAN", "same plan; KG entry already has extracted text"),
    "IE_BIODIV_ADAPT": ("IE_BIODIVERSITY_CLIMATE_CHANGE_SECTORAL_ADAPTATION", "same plan; KG entry already has extracted text"),
    "IE_MARINE_POLICY_STMT": ("IE_MARINE_PLANNING_POLICY_STATEMENT", "same statement; KG entry already has extracted text"),
    "IE_SEAFOOD_PROG": ("IE_SEAFOOD_DEVELOPMENT_PROGRAMME_2021_2027", "same programme; KG entry already has extracted text"),
    "IE_EPA_SOE": ("IE_EPA_SOE_2024", "same report; KG entry already has extracted text"),
    "GL_SDG_2030": ("GL_UN_SUSTAINABLE_DEVELOPMENT_AGENDA_2030", "same document (2030 Agenda); KG entry already has extracted text"),
    "IE_SDG_IMPL": ("IE_SDG_IMPLEMENTATION", "same plan; scraper entry already has the downloaded file"),
    "IE_COILLTE_LAND_USE": ("IE_COILLTE_FESLUP_REPORT", "same plan (FESLUP); KG entry already has extracted text"),
    "IE_AGRI_FOOD_STRATEGY": ("IE_FOOD_VISION", "same strategy (Food Vision 2030); KG entry already has extracted text"),
    "IE_AGRI_EMISSIONS": ("IE_AG_CLIMATISE", "same roadmap (Ag Climatise); KG entry already has extracted text"),
    "IE_MARINE_STRATEGY": ("IE_MARINE_STRATEGY_P3", "same document (MSFD Part 3 PoM); kept entry has KG text via twin"),
    "IE_LULUCF_PLAN": ("IE_FOREST_ACCOUNTING_PLAN_2021_2025", "same plan (NFAP/LULUCF forest reference levels); KG entry already has extracted text"),
    "IE_FOREST_ACCOUNTING": ("IE_FOREST_ACCOUNTING_PLAN_2021_2025", "same plan; KG entry already has extracted text"),
}


def main():
    rows = list(csv.DictReader(open(HERE / "master_library.csv", encoding="utf-8-sig")))
    ids = {r["id"] for r in rows}

    # sanity: only apply dedupe drops whose keep-target exists
    drops = {k: v for k, v in DEDUPE_DROPS.items() if k in ids and v[0] in ids}

    atlas_ids = {p.stem for p in ATLAS_TEXT.glob("*.txt")} if ATLAS_TEXT.exists() else set()
    scraper_stems = set()
    if SCRAPING.exists():
        for d in ["EU/downloaded_eu_docs", "GL/downloaded_docs", "GL/downloaded_docs_manual",
                  "GL/downloaded_docs_all", "IE/downloaded_ie_docs", "Ireland_mix", "LO_scraping/corpus"]:
            p = SCRAPING / d
            if p.exists():
                for f in p.rglob("*"):
                    if f.is_file() and not f.name.startswith("."):
                        scraper_stems.add(re.sub(r"\.(pdf|txt|html?)$", "", f.name, flags=re.I))

    out = []
    for r in rows:
        rid = r["id"]
        jur = r["jurisdiction"]
        include, reason = True, ""

        if rid in drops:
            include, reason = False, f"duplicate of {drops[rid][0]} — {drops[rid][1]}"
        elif r["status"] and r["status"] != "current":
            include, reason = False, f"status={r['status']}"
        elif r["kg_text_status"] == "excluded-no-text" or "MARPOL" in rid:
            include, reason = False, "MARPOL 1973 text superseded; consolidated text not freely available — excluded per review"
        elif jur == "LO":
            comp = next((cid for cid, c in LO_COMPOSITES.items() if c["members"](rid)), None)
            if rid == REPRESENTATIVE_LO:
                include, reason = True, "named exemplar of the local climate plan family"
            elif comp:
                include, reason = False, f"condensed into {comp} (sampled composite)"
            else:
                include, reason = True, "unique local-tier document"

        # where will the text come from?
        if int(r["kg_words"] or 0) > 0:
            src = "kg-segments"
        elif rid in atlas_ids:
            src = "atlas-text"
        elif rid in scraper_stems:
            src = "scraper-file"
        elif (r["source_url"] or "").strip():
            src = "download-needed"
        else:
            src = "NO-TEXT-NO-URL"

        # download-needed docs are deferred unless the file is already fetched
        # (gov.ie/NPWS bot-protection blocks automated fetch); they go to the
        # parked batch rather than blocking the build.
        if include and src == "download-needed":
            dl = HERE.parent / "corpus_build" / "downloads"
            have = any((dl / f"{rid}{e}").exists() for e in (".pdf", ".html", ".txt")) if dl.exists() else False
            if not have:
                include, reason, src = False, "text not fetchable via automation — deferred to parked batch", src

        # scraper-file docs are large peripheral EU technical instruments whose
        # PDFs OOM the extractor; deferred to the parked batch. Final build is
        # KG segments + Atlas text + composites (all clean, no heavy extraction).
        if include and src == "scraper-file":
            include, reason = False, "peripheral scraper document deferred to parked batch"

        out.append({
            "id": rid, "title": r["title"], "jurisdiction": jur,
            "domain": r["domain"], "year": r["year"],
            "include": "yes" if include else "no",
            "reason": reason,
            "text_source": src,
            "source_url": r["source_url"],
        })

    # synthetic composite entries
    for cid, c in LO_COMPOSITES.items():
        members = [r["id"] for r in rows if r["jurisdiction"] == "LO" and c["members"](r["id"])]
        if not members:
            continue
        out.append({
            "id": cid,
            "title": c["title"] + f" \u2014 {len(members)} plans",
            "jurisdiction": "LO", "domain": c["domain"], "year": 2024,
            "include": "yes",
            "reason": f"composite of {len(members)} members, {SEGMENTS_PER_MEMBER} segments sampled per member",
            "text_source": "composite:" + ",".join(members),
            "source_url": "https://www.caro.ie/knowledge-hub/local-authority-climate-action-plans-2024-2029" if "CLIMATE" in cid else "",
        })

    with open(HERE / "corpus_scope.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    inc = [o for o in out if o["include"] == "yes"]
    from collections import Counter
    print(f"scope: {len(inc)} included of {len(out)}")
    print("  by jurisdiction:", dict(Counter(o['jurisdiction'] for o in inc)))
    print("  by text source:", dict(Counter(o['text_source'] for o in inc)))
    print("  dedupe drops applied:", len([o for o in out if o['reason'].startswith('duplicate')]))
    bad = [o for o in inc if o["text_source"] == "NO-TEXT-NO-URL"]
    if bad:
        print(f"  !! included but no text and no URL ({len(bad)}):")
        for o in bad:
            print("     -", o["id"], o["title"][:50])


if __name__ == "__main__":
    main()
