# Mapping the Policy Web
## A semantic knowledge-graph tool for navigating Ireland's environmental governance

**Policy Brief · July 2026**
B. Johnston & J. Moran, Atlantic Technological University

---

### The problem

Ireland's environmental governance is delivered not through a single strategy but through
a large and growing library of separate documents: national acts and plans, EU directives
and regulations transposed into Irish law, sectoral strategies, and international
conventions. A single objective — restoring water quality, say, or meeting a 2030 emissions
target — is typically addressed across a dozen or more of these instruments at once,
authored by different departments and agencies at different times.

This fragmentation makes a basic but important question surprisingly hard to answer: **how
do these documents relate to one another?** Which cover the same ground? Which act as the
connective tissue between policy areas? Where does a newly published document sit relative
to everything already in force? Answering this by hand means reading and cross-referencing
hundreds of documents — slow, inconsistent, and rarely done comprehensively.

### What we built

We built an interactive tool that reads a corpus of **191 Irish, EU and international
environmental-governance documents**, converts every passage of every document into a
numerical representation of its meaning using modern language models, and measures how
closely each pair of documents overlaps in that meaning-space. The result is a live,
navigable map of the policy landscape — a "knowledge graph" — that a non-specialist can
explore in a web browser.

The corpus spans six policy domains: **water (53 documents), cross-cutting (52), climate
(40), biodiversity (30), agriculture (14) and forestry (2)**. It includes the most recent
material — the 2026 *A Living Land* Land Use Review, the 2024 State of the Environment
Report, the EU Nature Restoration Law — alongside foundational legislation and the
international conventions Ireland is party to. The 31 local-authority climate and
development plans are condensed into representative composite entries so that near-identical
documents do not crowd the picture.

### What it shows

The tool presents the corpus three ways — a **2D network**, a rotatable **3D view**, and a
**heatmap** — all driven by the same underlying similarity data. A user can search for any
document, filter by policy domain, adjust how strong a connection must be to appear, or pin
any document as a fixed reference point and see everything else ranked by its closeness to
it.

The central, immediate observation is visual: viewed as a whole, the corpus is a dense
tangle of connections. **This is the finding, not a defect.** Environmental governance in
Ireland is deeply interconnected; no document stands alone. The value of the tool is that it
lets a user move from that overwhelming whole to specific, answerable questions.

Some concrete results from the current corpus illustrate what it surfaces:

- **The EU Nature Restoration Law is the single strongest "bridge"** in the corpus — the
  document that most often lies on the connecting path between otherwise separate clusters of
  climate, biodiversity and water policy. This is a measurable reflection of its
  cross-sectoral design, and it flags the Law as a document whose implementation will touch
  many others.

- **The EPA State of the Environment Report 2024 is the most broadly connected document**,
  linking to 91 others. As an integrated assessment this is expected — and it makes the
  report a natural entry point for anyone trying to orient themselves in the landscape.

- **Successive and related documents are automatically identified.** Climate Action Plans
  2024 and 2025 register as near-identical (0.99 similarity); EU directives and their Irish
  transpositions pair up; several inadvertent near-duplicates in the source library were
  surfaced by the tool at a similarity of 1.0 — a useful side-effect for library curation.

- **Isolated documents are as informative as connected ones.** A handful of instruments sit
  at the edge of the map with few strong links — some because they are genuinely niche (the
  Batteries Regulation, the standalone Birds Directive), which may be entirely appropriate,
  and others because they are candidates for closer integration.

### How to use it — and how not to

The tool is a **discovery and triage instrument**. Used well, it rapidly narrows a large
document space down to the relationships worth a closer look, and makes the overall shape of
the landscape legible at a glance. Three practical uses:

1. **Orientation.** A new official, adviser or researcher can see the whole environmental
   policy landscape and its main clusters in minutes, and identify the handful of "hub"
   documents worth reading first.

2. **Situating a new document.** When a new strategy or directive appears, pinning it as the
   reference point shows immediately which existing instruments it most resembles, and
   therefore which teams and obligations it is most likely to interact with.

3. **Spotting gaps and overlaps.** Documents that connect to nothing may signal an
   integration gap; clusters of near-identical documents may signal duplication worth
   rationalising.

The single most important caveat is this: **the score starts the conversation; it does not
end it.** A strong connection means two documents *talk about the same things in similar
language*. It does **not** mean they agree, that they are mutually consistent, or that they
are legally compatible — the method cannot tell the difference between reinforcement,
duplication and contradiction. Every consequential reading must be confirmed by reading the
underlying documents. The tool is explicitly **not** a measure of policy coherence, legal
alignment, or compliance, and it produces no automated judgements. It tells you where to
look; expert judgement decides what you find.

### Method, in brief

Each document is divided into passages of roughly 350 words. Three independent,
publicly-available language models each convert every passage into a numerical
"fingerprint"; a document's fingerprint is the average of its passages. The similarity
between any two documents is the average of the three models' agreement on how close they
are — using several models guards against the quirks of any one. Connections are then sorted
into **weak, moderate and strong** bands defined statistically from the corpus itself, so
"strong" means *unusually similar for this body of documents* rather than an arbitrary
cut-off. All processing runs locally; no document data leaves the machine.

### Status and limitations

This is a **research and decision-support prototype**, not an operational or statutory tool.
Its limitations are documented transparently: it measures thematic overlap only; its
strength bands are specific to this corpus and not comparable across others; four documents
in the current build are scanned images whose text was not fully recovered and appear more
isolated than they should; and around 49 further documents identified for inclusion could
not be automatically retrieved from publisher websites and are queued for a later refresh.
None of these affect the tool's core demonstration — that modern language models can make
the structure of a large policy library visible and navigable.

### What this enables next

The underlying document library (300 catalogued documents with sources and metadata) and the
processing pipeline are reusable. The same approach could be applied to any policy domain or
extended to the full Irish statute book, to track how the landscape changes as documents are
added and withdrawn, or to compare Ireland's framework against another jurisdiction's. The
immediate contribution is a working demonstration that the interconnectedness of
environmental governance — long asserted, rarely shown — can be made concrete, measurable
and explorable.

---

*The interactive explorer, technical specification, and full document library are available
in the project repository. For access or a demonstration, contact the authors at Atlantic
Technological University.*
