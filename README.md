# KG-Noise-Lite

Does a BERT classifier detect **entity-level** and **relation-level** label
noise in KG-to-text data equally well, or is one structurally harder?

## Research question

Distant-supervision KG-to-text datasets commonly contain two kinds of label
noise: an entity in the triple doesn't match the sentence, or the relation
doesn't. This project fine-tunes `bert-base-cased` as a binary
grounded/noisy classifier and measures whether it — and a TF-IDF
cosine-similarity baseline — detect these two noise types equally well.

**Hypothesis:** entity corruption shows up as a surface lexical mismatch and
should be easy to detect (e.g. *Seoul → Busan*); relation corruption
preserves lexical overlap and only breaks the sentence semantically, so it
should be structurally harder (e.g. *capital → largest_city*).

## Data

- **Source:** [WebNLG](https://huggingface.co/datasets/GEM/web_nlg) (`GEM/web_nlg`,
  English, train split), loaded from HuggingFace's auto-converted parquet
  files — `load_dataset("GEM/web_nlg", "en")` fails under `datasets>=4`
  ("Dataset scripts are no longer supported"), so `scripts/01_load_webnlg.py`
  reads the `refs/convert/parquet` revision directly instead. Field layout is
  otherwise the dataset's original: `input` = triple strings
  (`"Subject | relation | Object"`), `target` = the reference sentence,
  `category` = domain category.
- Only single-triple examples are used (7,630 of 35,426) — a sentence
  describing multiple triples makes "which triple does this span correspond
  to" ambiguous, which is out of scope here (see Scope below).
- **Positive:** the original (triple, sentence) pair.
- **Negative**, two kinds, one of each generated per positive (so
  positive : entity_corruption : relation_corruption = 1 : 1 : 1, 22,890 rows
  total):
  - `entity_corruption` — swap the triple's subject or object (50/50) for a
    different value drawn from the corpus-wide entity pool. Sentence
    untouched.
  - `relation_corruption` — swap the triple's relation for a different value
    drawn from the corpus-wide relation pool (346 distinct relations).
    Sentence untouched.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | [`scripts/01_load_webnlg.py`](scripts/01_load_webnlg.py) | Load WebNLG, filter to single-triple examples, parse `"S \| R \| O"`, dedupe. |
| 2 | [`scripts/02_generate_negatives.py`](scripts/02_generate_negatives.py) | Generate `entity_corruption` and `relation_corruption` negatives, one of each per positive. |
| 3 | [`scripts/03_split_dataset.py`](scripts/03_split_dataset.py) | Split 70/15/15 at the `pair_id` level (all 3 rows of a pair stay together), stratified by `category`. |
| 4 | [`scripts/04_analyze_lengths.py`](scripts/04_analyze_lengths.py) | Measure tokenized `(triple_text, sentence)` length on train, recommend `max_length`. |
| 5 | [`scripts/05_train_bert.py`](scripts/05_train_bert.py) | Fine-tune `bert-base-cased` as a binary sequence-pair classifier, early stopping on validation F1 (macro). |
| 6 | [`scripts/06_tfidf_baseline.py`](scripts/06_tfidf_baseline.py) | TF-IDF + cosine-similarity baseline; threshold picked by grid search on validation F1 (macro). |
| 7 | [`scripts/07_evaluate_compare.py`](scripts/07_evaluate_compare.py) | Combine both models' test metrics into one comparison table. |
| 8 | [`scripts/08_error_analysis.py`](scripts/08_error_analysis.py) | **Core result:** recall broken down by `entity_corruption` vs `relation_corruption`, for both models, plus qualitative error cases. |

## Setup

This project has its own venv (separate from any other project on this
machine) because it needs a CUDA build of torch matched to the local driver:

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install transformers datasets scikit-learn numpy accelerate
.venv/bin/pip install -e . --no-deps
```

(`pip install torch` alone pulls whatever the latest CUDA build is — CUDA 13
at time of writing — which silently fails `torch.cuda.is_available()` on
older drivers. Pin the CUDA build to what your driver actually supports;
`nvidia-smi` reports the max CUDA version your driver handles.)

The last line installs this repo's own [`src/kg_noise/`](src/kg_noise)
package in editable mode (`--no-deps` because its runtime dependencies were
already installed on the line above) — it's what lets every
`scripts/NN_*.py` file do `from kg_noise import ...` instead of redefining
the same paths/IO/metrics helpers.

## Project layout

```
src/kg_noise/    # shared library code (paths, JSONL I/O, label constants,
                 # classification metrics, inference helpers) — imported by
                 # more than one script; installed as an editable package.
scripts/         # numbered pipeline entry points (01-10), run in order.
                 # Each one is a thin orchestration script, not a place to
                 # define reusable logic — that belongs in src/kg_noise/.
data/, models/   # pipeline inputs/outputs (see .gitignore for what's tracked).
```

## Usage

```bash
.venv/bin/python scripts/01_load_webnlg.py
.venv/bin/python scripts/02_generate_negatives.py
.venv/bin/python scripts/03_split_dataset.py
.venv/bin/python scripts/04_analyze_lengths.py
.venv/bin/python scripts/05_train_bert.py
.venv/bin/python scripts/06_tfidf_baseline.py
.venv/bin/python scripts/07_evaluate_compare.py
.venv/bin/python scripts/08_error_analysis.py
```

If the machine has more than one GPU, `scripts/05_train_bert.py` pins
`CUDA_VISIBLE_DEVICES=0` before importing torch — without it, HuggingFace
`Trainer` wraps the model in `DataParallel` across all visible GPUs and
crashes with an NCCL error on this setup.

## Results

Test set (3,435 rows: 1,145 positive / 1,145 entity_corruption / 1,145
relation_corruption):

| Metric | BERT (fine-tuned) | TF-IDF baseline |
|---|---|---|
| Accuracy | 0.9875 | 0.5872 |
| Precision (macro) | 0.9850 | 0.5418 |
| Recall (macro) | 0.9869 | 0.5430 |
| F1 (macro) | 0.9859 | 0.5422 |
| Recall (noisy) | 0.9886 | 0.6755 |
| Precision (noisy) | 0.9925 | 0.6962 |
| F1 (noisy) | 0.9906 | 0.6857 |

**Core result — recall by corruption type:**

| Model | Entity corruption recall | Relation corruption recall | Gap |
|---|---|---|---|
| BERT (fine-tuned) | 0.9965 | 0.9808 | +0.0157 |
| TF-IDF baseline | 0.7694 | 0.5817 | +0.1878 |

**The hypothesis holds for both models.** The gap is small for BERT (it
detects relation corruption well despite the harder setup) but large for
TF-IDF: mean cosine similarity for `entity_corruption` (0.104) is well below
clean pairs (0.178), but `relation_corruption` (0.166) is barely
distinguishable from clean — because relation names (`cityServed`,
`foundedBy`, ...) rarely appear as literal tokens in the sentence regardless
of whether they're correct, so a lexical-overlap baseline has almost no
signal to work with. Full breakdown, qualitative error cases, and a note on
a secondary pattern (numeric-object entity corruption is harder than
named-entity corruption) are in
[`data/errors/error_analysis_report.md`](data/errors/error_analysis_report.md).

## Scope

Deliberately excluded, left for future work:
- Other noise types (triple-order errors, multi-triple mixing, implicit
  relation noise).
- A real distant-supervision pipeline — this injects synthetic noise into
  clean WebNLG rather than working with naturally noisy data (e.g. NYT-FB).
  Steps 1–8 make no accuracy claim about real distant-supervision data; see
  the appendix below for why that number can't be produced at all.

## Appendix — qualitative NYT-FB probe (not a generalization benchmark)

[`scripts/09_nyt_case_study.py`](scripts/09_nyt_case_study.py) runs the
WebNLG-trained BERT model zero-shot on 30 random NYT-FB (`xiaobendanyn/nyt10`)
sentences, under two triple serializations — the raw Freebase relation path
(`/people/person/nationality`) and a naive natural-language version (last
path segment only, e.g. `nationality`) — to see whether relation *notation*
alone changes predictions.

**This produces no accuracy number.** NYT-FB is distant-supervision data:
there is no human-verified label for "does this sentence actually state this
triple," so there's nothing to score predictions against. The output
(`data/errors/nyt_case_study_report.md`) is a side-by-side prediction table
for manual case selection, not a metric.

Two things worth knowing before reading it:
- **The model collapsed to predicting "noisy" on 29/30 samples**, in both
  notations (0 predictions flipped between them). The one exception was the
  one case where the relation word literally appears in the sentence
  ("... founders of Endemol ..." / relation `founders`) — consistent with
  the Step 8 finding that the model leans on lexical overlap between the
  relation token and the sentence.
- **Is this just truncation?** `max_length=48` was fit to WebNLG's much
  shorter sentences (Step 4), and 25/30 NYT sentences exceed it — some by
  more than half their length. [`scripts/10_nyt_notrunc_comparison.py`](scripts/10_nyt_notrunc_comparison.py)
  re-ran the same 30 samples at `max_length=128` (no truncation for this
  set) to check. **Result: zero predicted labels changed** (0 flips in
  either notation, grounded count still 1/30 both ways) — so the near-total
  "noisy" collapse is not a truncation artifact. The one grounded case's
  *confidence* did drop substantially (0.965 → 0.581) once the full sentence
  was visible, consistent with the model leaning on local lexical overlap
  that gets diluted by more surrounding context. Full 2×2 breakdown in
  [`data/errors/nyt_notrunc_comparison_report.md`](data/errors/nyt_notrunc_comparison_report.md).
  Caveat: the model was only ever fine-tuned with fixed `max_length=48`
  batches, so position embeddings beyond index 47 never received a gradient
  update — this re-inference is a diagnostic, not evidence the model
  performs reliably at length 128.

## Future work

- Extend corruption types (order errors, multi-triple mixing, implicit
  relation noise) to build out a difficulty spectrum.
- A properly controlled distant-supervision generalization check would need
  a human-labeled mini test set (100–200 manually annotated NYT-FB rows) —
  the qualitative probe above is a substitute for that, not a replacement.
- Integrate a validated detector as a first-pass noise filter in a KG
  construction pipeline.
