# Skill2Intern — AI-Based Internship Recommendation System

A Flask web application that recommends internships to candidates from their education,
skills, sector interest and location preference. It ships three working ranking modes —
rule-based, TF-IDF machine learning and a hybrid of both — plus a REST API and a
responsive card interface.

---

## Project Overview

Skill2Intern takes a short candidate profile and returns up to five ranked internship
matches from a curated CSV catalogue. Education eligibility is a hard filter applied
before ranking, so a recommendation can never ignore a candidate's qualification.

The project is deliberately lightweight: a single Flask process, scikit-learn for
TF-IDF/cosine similarity, no database, no model server and no external ML APIs. The
TF-IDF model is fitted once at startup (~7 ms) and reused for every request.

> **Data notice:** the catalogue is synthetic sample data built for demonstration. None of
> the listings are real, open opportunities; the API flags every record with
> `is_sample: true` and the interface labels them accordingly.

---

## Key Features

- **Rule-based recommendations** — skill overlap, sector match and location match
- **TF-IDF vectorization** with unigrams and bigrams over title, sector and required skills
- **Cosine similarity** between the candidate's skill text and every internship document
- **Hybrid ranking** combining ML relevance with the rule-based score
- **Skill normalization and synonym matching** — `MS Excel` → `excel`, `Computer Basics` →
  `basic computer`, `MySQL` → `sql`, and more
- **Education eligibility filtering** across a 0–5 qualification hierarchy, with track-aware
  handling of vocational qualifications
- **Sector and location preferences**, including a state→city mapping so "Karnataka" and
  "Bangalore" behave identically
- **Match scoring and explanations** — every card states why it was recommended
- **Responsive web interface** (vanilla HTML/CSS/JS, no frontend framework)
- **Flask REST API** with a JSON error contract
- **Health-check endpoint** at `/health` for load balancers and App Runner
- **Automated testing** — 93 unit tests using the Flask test client

---

## Technologies Used

| Layer | Technology |
| --- | --- |
| Language | Python 3.11 |
| Backend | Flask 2.3.3 |
| Data handling | pandas 2.0.3, numpy 1.24.3 |
| Machine learning | scikit-learn 1.3.0 (`TfidfVectorizer`, `cosine_similarity`) |
| Frontend | Jinja2 templates, vanilla JavaScript, CSS |
| Data store | CSV (`data/internship.csv`, 115 records) |
| Testing | Python `unittest` + Flask test client |
| Serving | `python app.py` locally, gunicorn 21.2.0 on Linux |

Runtime dependencies are pinned in `requirements.txt`. `gunicorn` is marked
`sys_platform != "win32"`, so Windows installs work unchanged.

---

## How It Works

1. The candidate submits education, skills (comma separated), sector and location.
2. Skills are normalized (lowercased, whitespace collapsed, synonyms mapped) on both the
   candidate and the catalogue side.
3. **Education eligibility is applied first.** An internship is considered only if the
   candidate's level is greater than or equal to the internship's required level.
4. Eligible internships are scored by the selected mode, ranked descending and truncated
   to the top 5.
5. The browser renders each result as a card with its score and an explanation.

**Qualification levels:** `10th Pass (0)`, `12th Pass (1)`, `ITI (2, vocational)`,
`Diploma (2)`, `Undergraduate / B.Tech / Graduate (3)`, `Postgraduate (4)`.
An ITI certificate does not satisfy a Diploma requirement; unknown qualifications are
rejected by the API rather than silently defaulted.

### API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Web interface |
| `GET` | `/health` | Health check: status, model and dataset metadata |
| `POST` | `/recommend` | Ranked recommendations (JSON) |

```http
POST /recommend
Content-Type: application/json

{
  "qualification": "B.Tech",
  "skills": "Python, SQL, Java",
  "sector_interested": "Information Technology",
  "location_interested": "Karnataka",
  "mode": "hybrid"
}
```

`mode` is optional: `rule` (default), `ml` or `hybrid`. Responses contain every CSV field
plus `score`, `mode`, `rule_score`, `tfidf_similarity`, `hybrid_score`, `matched_skills`,
`synonym_matches`, `matched_sector`, `matched_location`, `is_sample` and `explanation`.

Failures return HTTP 400 with `{"error": "..."}` — missing or malformed body, blank
profile, unknown mode, and unsupported or missing qualification.

---

## Recommendation Models

All three modes share the same eligibility filter and the same top-5 cap.

### 1. Rule-Based

Exact skill overlap after normalization:

| Rule | Points |
| --- | --- |
| Each matching skill | +15 |
| Share of required skills matched | up to +10 |
| Location match (state or city) | +10 |
| Sector match | +5 |

Internships with a score of zero are excluded; slots are never padded with ineligible
records.

### 2. ML-Based (TF-IDF + Cosine Similarity)

One document per internship is built from `title + sector + required_skills`, vectorized
with `TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)`.
The candidate's skills form the query, and cosine similarity ranks every eligible
internship. 115 documents produce 1,145 features.

### 3. Hybrid

```
rule_norm = min(rule_points / 60, 1)
ml_norm   = cosine similarity          # already 0..1
hybrid    = 0.6 * ml_norm + 0.4 * rule_norm
```

The 60-point reference is fixed, so a given internship scores identically regardless of
which other rows are eligible.

### Measured results

Evaluated on 9 profile definitions with criteria-derived relevance labels over the
115-record catalogue (`python evaluation/evaluate.py`):

| Mode | Precision@5 | Recall@5 | NDCG@5 | Eligibility violations | Median latency |
| --- | --- | --- | --- | --- | --- |
| Rule-based | 0.778 | 0.645 | 0.889 | 0 | 5.0 ms |
| ML (TF-IDF) | 0.889 | 0.715 | 1.000 | 0 | 5.1 ms |
| Hybrid 0.6/0.4 | 0.889 | 0.715 | 1.000 | 0 | 5.1 ms |

These figures come from a small synthetic benchmark, not from human relevance judgments;
they compare the modes against each other and are not a claim about real-world hiring
outcomes.

---

## Project Structure

```
app.py                        Flask app: data loading, eligibility, ranking, API routes
ml_recommender.py             TF-IDF model, skill normalization, synonyms, cosine similarity
gunicorn.conf.py              Production server configuration (PORT, workers, logging)
data/internship.csv           Internship catalogue (115 records)
templates/index.html          Profile form, mode selector and results container
static/script.js              Form submission, fetch() calls, card rendering
static/style.css              Responsive layout and card styling
tests/test_recommend.py       API and rule-based regression tests
tests/test_ml_recommender.py  TF-IDF, hybrid ranking and API integration tests
tests/test_pipeline.py        Data pipeline, normalization and qualification tests
evaluation/                   Profile definitions and Precision/Recall/NDCG harness
scripts/                      Dataset fetch, audit, preprocessing and model comparison
pipeline/                     Normalization modules and offline model variants
requirements.txt              Pinned runtime dependencies
```

---

## Installation and Setup

Requires Python 3.11.

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The app reads `data/internship.csv` at import time, so run it from the repository root.

---

## Running the Application

```bash
python app.py
```

Open <http://127.0.0.1:5000>.

Optional environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PORT` | `5000` | Bind port (App Runner supplies this in production) |
| `HOST` | `127.0.0.1` | Bind address |
| `FLASK_DEBUG` | off | `1`, `true`, `yes` or `on` enables the debugger |

```bash
PORT=8000 FLASK_DEBUG=1 python app.py
```

Production (Linux):

```bash
gunicorn -c gunicorn.conf.py app:app
```

---

## Testing

```bash
python -m unittest discover -s tests -v
```

93 tests cover the rule engine, TF-IDF vectorization, cosine similarity, skill
normalization and synonyms, education eligibility, qualification validation, duplicate
suppression, ranking order, the API contract, the health endpoint and the data pipeline.

Offline evaluation and data tooling:

```bash
python evaluation/evaluate.py           # Precision@5 / Recall@5 / NDCG@5 / latency
python scripts/experiment_models.py     # compare TF-IDF variants and embeddings
```

---

## AWS Deployment

The application runs unchanged under gunicorn on AWS App Runner.

**Start command**

```bash
gunicorn -c gunicorn.conf.py app:app
```

`gunicorn.conf.py` binds `0.0.0.0:$PORT` (default 8080), runs 2 workers × 2 threads, logs
to stdout and honours `WEB_CONCURRENCY`, `GUNICORN_THREADS`, `GUNICORN_TIMEOUT` and
`GUNICORN_LOG_LEVEL`.

**Recommended App Runner settings**

| Setting | Value |
| --- | --- |
| Build command | `pip install -r requirements.txt` |
| Start command | `gunicorn -c gunicorn.conf.py app:app` |
| Port | `8080` |
| Health check path | `/health` |
| CPU / memory | 1 vCPU / 512 MB (1 GB for headroom) |
| Environment variables | none required |

The deployed image carries no ML model files and no torch dependency: the TF-IDF matrix
(~25 KiB) is fitted per worker at import. Measured request latency is ~5 ms in-process.

No AWS resources are created by this repository. Gunicorn itself was not executed during
development (it is excluded on Windows); verify the start command in the App Runner build
log on first deploy.

---

## Future Enhancements

- Resume upload and parsing
- Skill-gap analysis showing missing skills per recommendation
- Collaborative filtering and supervised ranking — **not implemented**; no candidate–job
  relevance labels are available to train or evaluate such a model
- User accounts and persistent storage
- Live integration with an internship portal API

---

## License and Attribution

This project is provided for educational and demonstration purposes. No licence file is
included in the repository; add one before redistributing.

The catalogue is synthetic sample data authored for this project. An audit of an external
research dataset (`Uzaib52/indian-tech-career-intelligence-2026`, MIT licence) is retained
under `data/audit/` for reproducibility; that dataset was evaluated and **not** used for
training or shown to users.

Background reading that informed the ranking approach:

1. Corné de Ruijt, Sandjai Bhulai (2021) — [Job Recommender Systems: A Review](https://arxiv.org/abs/2111.13576)
2. ITM Conferences (2022) — [Job Recommendation Approaches](https://www.itm-conferences.org/articles/itmconf/abs/2022/04/itmconf_icacc2022_02002/itmconf_icacc2022_02002.htm)
3. G. Deepak — [Applying Data Mining Techniques in Job Recommender System](https://gdeepak.com/pubs/Applying_Data_Mining_Anika.pdf)
