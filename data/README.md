# Data directory

```
data/
  internship.csv        Production catalog: 115 synthetic sample listings (unchanged)
  candidate.csv         Single sample candidate row (unused by the app)
  external/raw/         Downloaded source files (git-ignored, 57 MB)
  processed/            Normalised sample from the external source (5,000 rows)
  training/             EMPTY on purpose — see below
  evaluation/           Model comparison output from scripts/experiment_models.py
  audit/                Dataset audit report and metrics
```

## Licence and attribution

`Uzaib52/indian-tech-career-intelligence-2026` — MIT licence, downloaded 2026-11-19,
HF revision `a6182954612f555ea3732cff30f36c8bb7901e4b`,
SHA-256 `f69a8de12e0d379419e7f9767c74a366366fabab864d20eb6f414bba21ad403c`.
Attribution: dataset by Hugging Face user `Uzaib52`.

The source is **synthetic**: 100,000 rows contain 24 unique templated descriptions and
real company names attached to fabricated postings. See `data/audit/dataset_audit.md`.
It is therefore never merged into the public catalogue and never shown to users.

## Why `data/training/` is empty

No verified source with (a) real Indian postings, (b) usable descriptions, (c) a clear
licence and (d) candidate–job relevance labels was found. Training a supervised ranker
without trustworthy labels would mean training on labels the audited data does not
contain, so the supervised-ranking experiment was **not** run. Every model that was
evaluated (TF-IDF variants, sentence embeddings) is unsupervised and is fitted on the
production catalogue at evaluation time — there is nothing to persist here.

If a labelled corpus is obtained later, put `train.csv` / `validation.csv` / `test.csv`
here, split by candidate *and* job id so no profile or posting crosses splits.

## Rebuilding from scratch

```bash
# 1. download (57 MB) and record provenance
python scripts/fetch_dataset.py                 # writes data/external/raw/ + manifest

# 2. audit the raw file (writes data/audit/*)
python scripts/audit_dataset.py \
  --csv data/external/raw/indian_tech_career_intelligence_2026.csv \
  --name indian_tech_career_2026

# 3. normalise a deterministic 5,000-row sample
python scripts/preprocess.py \
  --csv data/external/raw/indian_tech_career_intelligence_2026.csv \
  --source "Uzaib52/indian-tech-career-intelligence-2026" \
  --source-url "https://huggingface.co/datasets/Uzaib52/indian-tech-career-intelligence-2026" \
  --status synthetic_templated --sample 5000 \
  --out data/processed/jobs_sample.csv

# 4. compare models
python scripts/experiment_models.py
```

## Standard schema (`data/processed/jobs_sample.csv`)

`job_id, title, company_name, description, required_skills, required_education,
education_level, education_track, education_known, sector, location, location_raw, state,
stipend, stipend_period, job_type, source, source_url, data_status, description_quality,
is_currently_open, raw_company_name, raw_required_education, raw_skills`

* `required_education` holds only a canonical label when the source value could be
  normalised; otherwise it is empty and `education_known = false`.
* `raw_*` columns preserve the untouched source values for traceability.
* `data_status = synthetic_templated` and `is_currently_open = false` on every row — the
  records are research samples, never active openings.
