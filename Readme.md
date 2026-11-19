# AI-Based Internship Recommendation Engine for PM Internship Scheme

### Smart India Hackathon (SIH) 2025

**Team NOVAS**

* Asif Qamar
* Fatima Aslam
* Aryan Jha
* Souvik Majee
* Avik Ghosh
* Prerna Priya

---

## What this repository is

A working Flask prototype that takes a candidate profile (education, skills, sector
interest, location preference) and returns up to five ranked internship recommendations
from a local CSV dataset.

Everything described under **Current Features** is implemented in this repository and
covered by the automated tests in `tests/`. Anything not implemented is listed under
**Planned Features** and is not claimed as working.

> **Data notice:** the internship dataset is synthetic sample data built for this
> prototype. It is not scraped from, or affiliated with, the real PM Internship Scheme
> portal, and none of the listings are real opportunities. The 15 records added for
> 10th Pass / 12th Pass / Diploma profiles are labelled `(Sample)` in the company name.

---

## Tech stack (as implemented)

| Layer | Technology |
| --- | --- |
| Backend | Python 3.11, Flask 2.3.3 |
| Data handling | pandas 2.0.3 (numpy 1.24.3 is a pandas dependency and is imported by the ML module) |
| Machine learning | scikit-learn 1.3.0 — `TfidfVectorizer` (unigrams + bigrams) + cosine similarity, fitted once per process |
| Frontend | Jinja2 template (`templates/index.html`), plain JavaScript (`static/script.js`), plain CSS (`static/style.css`) — no frontend framework |
| Data store | single CSV file, `data/internship.csv` |
| Tests | Python standard-library `unittest` + Flask test client |
| Serving | `python app.py` locally; gunicorn on Linux (`gunicorn==21.2.0`, windows-excluded marker) |
| Language | Google Translate widget embedded in the page header |

---

## Project structure

```
app.py                       Flask app: CSV loading, eligibility, ranking, /recommend API
ml_recommender.py            TF-IDF model: normalisation, synonyms, cosine similarity
data/internship.csv          Internship dataset (115 records)
data/candidate.csv           Single sample candidate row (not read by the app)
templates/index.html         Jinja2 template: profile form, mode selector, results container
static/script.js             Form submit, fetch() to /recommend, card rendering
static/style.css             Responsive card layout
tests/test_recommend.py      Rule-based regression tests for the API
tests/test_ml_recommender.py TF-IDF, hybrid ranking and API integration tests
evaluation/profiles.json     Evaluation profiles + relevance criteria
evaluation/evaluate.py       Precision@5 / Recall@5 / latency comparison harness
requirements.txt             Runtime dependencies
```

---

## Setup

Requires Python 3.10+ (developed on 3.11).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python app.py
```

The app serves on <http://127.0.0.1:5000>.

Optional environment variables (all read at startup):

| Variable | Default | Meaning |
| --- | --- | --- |
| `FLASK_DEBUG` | off | `1`, `true`, `yes` or `on` enables the Flask debugger |
| `HOST` | `127.0.0.1` | bind address |
| `PORT` | `5000` | bind port |

Example: `PORT=8000 FLASK_DEBUG=1 python app.py`

`app.py` loads `data/internship.csv` on import, so the process must run from the
repository root. If the file is missing it raises `FileNotFoundError` naming the path
instead of exiting.

---

## Recommendation logic

### 1. Input and normalisation

* `qualification`, `skills` (comma separated), `sector_interested`, `location_interested`.
* Every skill token is lower-cased and internal whitespace runs are collapsed, on both
  the candidate and the dataset side, so `"  BASIC   computer "` matches `"Basic Computer"`.
* Skills the dataset does not contain are ignored rather than treated as wildcards.

### 2. Education eligibility

Each internship's freeform `required_education` text is parsed into a level by
`parse_education_level()` (word-boundary regex, so `BA` does not match inside `MBA`).
A candidate is eligible when their level is **greater than or equal to** the required
level.

The **candidate's** qualification is validated separately by `resolve_candidate_level()`:

* An empty or whitespace-only qualification is rejected with HTTP 400
  (`"qualification is required; send one of: ..."`). It does **not** skip the filter —
  skipping would hand every candidate eligibility they never claimed.
* An unrecognised value such as `"B.Voc in Retail"` is rejected with HTTP 400
  (`"Unsupported qualification ..."`). It is not silently mapped to a bachelor's level.
* Freeform text containing a recognised keyword is still accepted, so
  `"Pursuing B.Tech in Computer Science"` resolves to level 3.

Dataset rows keep a fallback: a requirement string that cannot be parsed is treated as
level 3 so a malformed row can never become universally eligible to everyone below it.

| Level | Education |
| --- | --- |
| 0 | 10th Pass |
| 1 | 12th Pass |
| 2 | Diploma |
| 3 | B.Tech, B.Sc, B.Com, BBA, BCA, BA, B.Design, B.Pharma, Bachelor's |
| 4 | M.Tech, M.Sc, MBA, MA, M.Pharma, Master's |
| 5 | PhD |

Because the comparison is a hierarchy, a B.Tech graduate is eligible for a Diploma-level
internship, while a Diploma holder is **not** eligible for a B.Tech-level internship.

### 3. Scoring

| Rule | Points |
| --- | --- |
| Each matching skill | +15 |
| Bonus for the share of required skills matched | up to +10 (`matched / required × 10`) |
| Location matches (state or city) | +10 |
| Sector matches | +5 |

Location uses a server-side `STATE_TO_CITIES` map, so selecting a state such as
Karnataka scores exactly like selecting Bangalore.

### 4. Ranking and filtering

* Internships with no positive score are excluded — slots are never padded with
  ineligible records.
* Results are sorted by score, descending, and the top `MAX_RESULTS` (5) are returned.
* An `internship_id` is counted once even if it appears twice in the CSV.

### 5. Errors

`POST /recommend` returns JSON on every failure path — missing body, malformed JSON,
non-object JSON, a completely blank profile, an unknown `mode`, an unsupported
qualification and a missing qualification all return HTTP 400 with `{"error": "..."}`.

### Health check

```http
GET /health
```

```json
{
  "status": "ok",
  "model": {"type": "tfidf-cosine", "documents": 115, "features": 1145, "fitted": true},
  "dataset": {"records": 115, "synthetic": true}
}
```

Used as the App Runner health-check path; it performs no scoring work.

---

## Recommendation modes

The request accepts an optional `mode` field. Omitting it keeps the pre-ML API
behaviour (`rule`); the web form defaults to `hybrid`.

| Mode | Ranking score | Notes |
| --- | --- | --- |
| `rule` | rule points as-is (e.g. `34.0`) | exact skill overlap required, as before |
| `ml` | cosine similarity, 0–1 | relevance from TF-IDF text similarity only |
| `hybrid` | `0.6 × similarity + 0.4 × (rule points / 60)` | default weight set; a different `ml_weight` can be passed to `rank_internships()` |

Education eligibility is applied **before** ranking in all three modes, so a high text
similarity can never place a candidate in an internship they do not qualify for. The
`rule` mode keeps the exact-skill gate; the ML modes drop it and rely on the measured
similarity, which is what lets synonym-only matches surface.

## Machine-learning model (TF-IDF)

`ml_recommender.py`, built on scikit-learn.

* **Corpus:** one document per internship, built from the columns that exist —
  `title + sector + required_skills`. The dataset has no free-text description, so none
  is invented.
* **Vectoriser:** `TfidfVectorizer(ngram_range=(1, 2), stop_words='english',
  sublinear_tf=True, min_df=1)`. Unigrams and bigrams, 1,145 features over 115 documents.
* **Query:** the candidate's skills string, normalised the same way as the documents.
* **Similarity:** cosine similarity between the query vector and every internship vector,
  in a single sparse matrix product.
* **Synonyms:** a small table maps applicant vocabulary onto dataset wording —
  `MS Excel → excel`, `Computer Basics → basic computer`, `MySQL → sql`,
  `Customer Support → customer service`, `Tally ERP → tally`, and a dozen more. It is
  applied to both sides so the vectors stay in one vocabulary.
* **Fitted once:** `build_model()` runs at import time in `app.py` and the matrix is
  reused for every request. Scoring a request only transforms the query — no refitting,
  no model server, no external API. Fit cost is ~7 ms and the similarity matrix holds
  2,121 non-zero entries (~25 KiB).

### Hybrid scoring formula

```
rule_norm  = min(rule_points / 60, 1)          # 60 = reference maximum
ml_norm    = cosine_similarity                  # already 0..1
hybrid     = 0.6 * ml_norm + 0.4 * rule_norm
```

The 60-point reference is fixed, not computed per request, so a given internship
receives the same score for a given profile regardless of which other rows are eligible.

### Match explanations

Every returned record carries `mode`, `rule_score`, `tfidf_similarity`, `hybrid_score`,
`matched_skills`, `matched_sector`, `matched_location` and a generated `explanation`
string, for example:

> Hybrid score 0.33 = 60% text similarity (0.17) + 40% rule score (34 points, scaled to
> 0.57). Matched 2 of 5 required skills (python, sql); sector preference matches.

When the only reason a row matched is the synonym table, the explanation says so
explicitly and names both sides of the mapping:

> Hybrid score 0.17 = 60% text similarity (0.29) + 40% rule score (0 points, scaled to
> 0.00). Matched 1 required skill(s) through synonym mapping ('computer basics' means
> 'basic computer').

The same pairs are returned structurally in `synonym_matches`:
`[{"input": "computer basics", "dataset_skill": "basic computer"}]`.

The similarity value is described as a cosine similarity between 0 and 1. It is **not** a
probability of being selected for the internship, and the UI says so above the results.

### API

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

`mode` is optional: `rule` (default), `ml` or `hybrid`.

Response: a JSON array (max 5), ordered by descending `score`, each item containing all
CSV columns plus the computed score and its breakdown:

```json
[
  {
    "internship_id": "INT-DS-25-026",
    "company_name": "LogicLoop AI",
    "title": "Data Analyst Intern",
    "sector": "Data Science & Analytics",
    "required_skills": "SQL, Excel, Tableau, Python, Communication",
    "required_education": "Pursuing B.Sc in Statistics",
    "location": "Gurgaon",
    "stipend": 32000,
    "score": 0.328,
    "mode": "hybrid",
    "rule_score": 34.0,
    "tfidf_similarity": 0.169,
    "hybrid_score": 0.328,
    "matched_skills": ["python", "sql"],
    "matched_sector": false,
    "matched_location": false,
    "explanation": "Hybrid score 0.33 = 60% text similarity (0.17) + 40% rule score (34 points, scaled to 0.57). Matched 2 of 5 required skills (python, sql)."
  }
]
```

---

## Dataset

`data/internship.csv` — 115 records, one header row.

| Column | Example |
| --- | --- |
| `internship_id` | `INT-IT-25-001` |
| `company_name` | `Nexus Technologies` |
| `title` | `Software Development Intern` |
| `sector` | `Information Technology` |
| `required_skills` | `Python, Java, Git, SQL, Data Structures` |
| `required_education` | `Pursuing B.Tech in Computer Science` |
| `location` | `Bangalore` |
| `stipend` | `₹35000/month` |

* **ID format:** `INT-<SECTOR CODE>-<YY>-<NNN>`, for example `INT-IT-25-001`.
* **Sectors (10):** Information Technology, Marketing & Advertising, Data Science &
  Analytics, Human Resources, E-commerce, Finance, FinTech, EdTech, Design, Healthcare.
* **Cities (7):** Bangalore, Mumbai, Pune, Hyderabad, Gurgaon, Delhi, Chennai.
* **Records 001–100:** synthetic listings for degree-level profiles.
* **Records 101–115:** 15 demo records added so that 10th Pass, 12th Pass and Diploma
  profiles return results — five per level, using the practical skills `Excel`,
  `Data Entry`, `Typing`, `Tally`, `Basic Computer`, `Communication`, `Customer Service`,
  `Troubleshooting`, `Inventory`. Their `company_name` ends in `(Sample)`.

`stipend` is stored as `₹<amount>/month` and converted to an integer when the CSV is loaded.

Every one of the 115 records is synthetic. Recommendations are returned with
`is_sample: true`, and the page displays a demo-data banner plus a badge on every card so
the UI never implies these are real, open listings.

---

## Model evaluation

`python evaluation/evaluate.py` scores the three approaches against 9 profiles whose
relevant internships are defined by the criteria in `evaluation/profiles.json`
(education level ≤ profile level **and** at least one criterion skill appears as a whole
word in the record's skills). The labels are computed from the CSV, never from a
recommender's output.

| Approach | Mean Precision@5 | Mean Recall@5 | Eligibility violations |
| --- | --- | --- | --- |
| Rule-based | 0.778 | 0.645 | 0 |
| TF-IDF (`ml`) | 0.889 | 0.715 | 0 |
| Hybrid 0.6 / 0.4 (default) | 0.889 | 0.715 | 0 |
| Hybrid 0.5 / 0.5 | 0.889 | 0.715 | 0 |
| Hybrid 0.7 / 0.3 | 0.889 | 0.715 | 0 |
| Hybrid 0.4 / 0.6 | 0.889 | 0.715 | 0 |

Per profile (hybrid 0.6/0.4 and `ml` produce the same top-5 on this dataset):

| Profile | Returned | P@5 | R@5 | Rule-based P@5 |
| --- | --- | --- | --- | --- |
| 10th Pass, Excel + Data Entry | 3 | 0.60 | 1.00 | 0.60 |
| 10th Pass, Customer Service + Communication | 4 | 0.80 | 1.00 | 0.80 |
| 12th Pass, Tally + Excel | 3 | 0.60 | 1.00 | 0.60 |
| 12th Pass, Basic Computer + Troubleshooting | 5 | 1.00 | 1.00 | 1.00 |
| 12th Pass, "MS Excel, Computer Basics" | 5 | 1.00 | 0.62 | **0.00** (no results) |
| Diploma, Troubleshooting + Basic Computer | 5 | 1.00 | 0.56 | 1.00 |
| Diploma, Inventory + Excel | 5 | 1.00 | 0.50 | 1.00 |
| B.Tech, Python + SQL | 5 | 1.00 | 0.42 | 1.00 |
| B.Tech, Excel + Data Analysis | 5 | 1.00 | 0.33 | 1.00 |

How to read this honestly:

* The entire ML advantage comes from the synonym profile. The rule engine returns **no
  results** for "MS Excel, Computer Basics" because neither phrase exists verbatim in the
  dataset; TF-IDF with the synonym table finds the five matching rows.
* On every other profile the two approaches agree, and all four hybrid weightings produce
  identical top-5s. Changing 60/40 to 70/30 or 40/60 does **not** change the ranking on
  this dataset — the ML weight is not shown to be better than the alternatives.
* Precision@5 is capped by the label count: a profile with 3 relevant records cannot
  score above 0.60, so the mean P@5 is a property of the label sets as much as the model.
* Recall drops on the B.Tech profiles because they have 12–15 relevant rows and only 5
  slots; that is a ranking-pool limitation, not necessarily a ranking error.
* Latency (median, in-process, 20 × 9 calls, latest run): rule 5.24 ms, `ml` 5.05 ms,
  hybrid 5.55 ms. All three are within noise of each other.
* 9 profiles and 115 synthetic records is far too small to claim the ML model is
  generally better. The rule engine is the baseline; the ML method is not worse, and it
  fixes one real vocabulary gap.

---

## External data: verification and outcome

Full evidence in `data/audit/dataset_audit.md`. Summary:

| Source | Licence | Records | India | Verdict |
| --- | --- | --- | --- | --- |
| `Uzaib52/indian-tech-career-intelligence-2026` | MIT | 100,000 (card claims 1,000,000) | 100% | **Rejected as training data** |
| `xanderios/linkedin-job-postings` | MIT | 33,246 | ~0% (US) | Rejected — no Indian locations |
| `NextGig-Rocks/global-job-postings-multi-ats` | CC-BY-4.0 | 112,816 | ~0% (North America) | Rejected — no Indian locations |
| `SuhaibAtef/tech-job-postings-labeled` | none declared | 8,000 | unknown | Rejected — unclear reuse rights |

The first source is the one the task pointed at. It exists and its MIT tag is real, but
measurement contradicts the description:

* **24 unique job descriptions in 100,000 rows** (99.98% duplicate, 99.52% near-duplicate
  at Jaccard ≥ 0.9), all matching a single template.
* 12 companies, 24 titles, 6 distinct skill bundles, 8 cities, **1 industry**.
* Near-uniform distributions (education 33,414 / 33,349 / 33,237; cities ≈12.5k each) —
  generated, not observed.
* Real employers (TCS, Cognizant, Ola Electric) attached to fabricated postings.
* **Zero records** for 10th Pass, 12th Pass, ITI or Diploma; zero non-IT sectors.

Retained: a deterministic 5,000-row sample normalised into
`data/processed/jobs_sample.csv` for pipeline validation, every row flagged
`data_status = synthetic_templated`, `is_currently_open = false`. **No external row was
used to train or tune anything.**

## Data pipeline

```
scripts/fetch_dataset.py     download + sha256 + provenance manifest
scripts/audit_dataset.py     missing %, duplicates, near-duplicates, distributions
scripts/preprocess.py        canonical schema, qualification + skill + location mapping
scripts/experiment_models.py model comparison -> data/evaluation/model_comparison.{json,md}
pipeline/skills_normalizer.py  skill/city/state canonicalisation
pipeline/qualifications.py     qualification levels + tracks + eligibility policy
pipeline/models.py             TF-IDF variants and sentence-embedding ranker
```

Standard schema (`data/processed/jobs_sample.csv`): `job_id, title, company_name,
description, required_skills, required_education, education_level, education_track,
education_known, sector, location, location_raw, state, stipend, stipend_period, job_type,
source, source_url, data_status, description_quality, is_currently_open,
raw_company_name, raw_required_education, raw_skills`. Raw values are preserved in the
`raw_*` columns; missing source fields are left empty rather than invented.

**Qualification policy** (deliberately not a plain hierarchy):

| Label | Level | Track |
| --- | --- | --- |
| 10th Pass | 0 | academic |
| 12th Pass | 1 | academic |
| ITI | 2 | vocational |
| Diploma | 2 | academic |
| Undergraduate / B.Tech / Graduate | 3 | academic |
| Postgraduate | 4 | academic |

An ITI certificate does **not** satisfy a Diploma requirement; a Diploma holder is
eligible for an ITI-level role. Unrecognised values are `None` → the record is excluded
from eligibility-sensitive ranking instead of being guessed.

## Model experiments (same 9-profile criteria benchmark)

| Model | P@5 | R@5 | NDCG@5 | Violations | Diversity | Latency | ΔRAM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Rule-based (production baseline) | 0.778 | 0.645 | 0.889 | 0 | 0.61 | 5.0 ms | — |
| Hybrid 0.6/0.4 (production default) | **0.889** | **0.715** | **1.000** | 0 | 0.65 | 5.1 ms | — |
| A0 TF-IDF title+sector+skills, 1–2 | 0.889 | 0.715 | 1.000 | 0 | 0.65 | 1.5 ms | +0.2 MB |
| A1 TF-IDF skills only, 1–1 | 0.889 | 0.715 | 1.000 | 0 | 0.65 | 1.5 ms | +0.0 MB |
| A2 TF-IDF title+skills, 1–2 | 0.889 | 0.715 | 1.000 | 0 | 0.65 | 1.5 ms | +0.1 MB |
| A3 TF-IDF title+sector+skills, 1–1 | 0.889 | 0.715 | 1.000 | 0 | 0.65 | 1.5 ms | +0.1 MB |
| B Sentence embeddings (all-MiniLM-L6-v2) | 0.867 | 0.703 | 0.980 | 0 | 0.67 | 7.7 ms | **+381 MB** |

Reading:

* **No model beats the production hybrid.** The TF-IDF variants tie it exactly; the
  sentence-transformer is slightly worse (0.867 vs 0.889) at 381 MB of extra RAM.
* The benchmark saturates (NDCG@5 = 1.000) for every TF-IDF variant, so it cannot
  discriminate between them — a tie is not evidence of improvement.
* The external description column adds no signal: 5,000 documents collapse to **24 unique
  TF-IDF vectors**, and the mean cosine between different job titles is 0.234.
* **Model C (supervised ranking) was not built:** no source provides candidate–job
  relevance labels, and inventing them would be the fabrication this project forbids.
* Decision: **the production recommender is unchanged.** No new API mode was added,
  because nothing cleared the "meaningful improvement" bar.

Model B is evaluation-only: `sentence-transformers` and `torch` are **not** in
`requirements.txt`, so the deployed app stays on scikit-learn only.

---

## Current features (implemented and tested)

* Profile form with education dropdown (10th Pass through MA), free-text skills,
  sector dropdown and state/UT location dropdown.
* Education eligibility filtering across the 0–5 level hierarchy, including 10th Pass,
  12th Pass and Diploma profiles.
* Skill matching that ignores case and redundant whitespace, and works for
  comma-separated skill lists.
* Sector matching and state-or-city location matching.
* Ranked recommendations (up to 5), no duplicates, no ineligible padding.
* Result cards showing title, company, required skills, sector, location, stipend,
  match score and internship ID.
* Three recommendation modes — rule-based, TF-IDF ML, hybrid — selectable in the form and
  via the `mode` API field.
* TF-IDF model (unigrams + bigrams, cosine similarity) fitted once at startup, with a
  skill synonym table.
* Per-card explanation of the actual match (skills matched, sector/location match,
  text similarity and its weighted contribution).
* Loading, empty and error states in the UI.
* JSON error contract on `/recommend`, including unknown-mode, unsupported-qualification
  and missing-qualification validation.
* `GET /health` health-check endpoint returning model and dataset status.
* Every recommendation is returned with `is_sample: true`, and the UI shows a page-level
  demo banner plus a per-card "Sample listing — not a real opening" badge.
* Google Translate widget in the header for on-page translation.

## Planned features (not implemented)

These appear in the original proposal but **do not exist in this repository**:

* Resume upload and resume parsing.
* Skill-gap analysis (missing skills per recommendation).
* Collaborative filtering or any behavioural learning. The implemented model is
  content-based TF-IDF only — it learns nothing from user history, clicks or outcomes.
* A trained/learned ranking model (the hybrid weights are hand-set constants).
* React.js or React Native frontend; swipe interactions.
* User accounts, persistence or a database.
* Live integration with the PM Internship Scheme portal or any real listing feed.
* Native multilingual UI strings (only the Google Translate widget is present).
* "Other" sector/qualification options.

## Known limitations

* In `rule` mode matching is exact-token after normalisation: `MS Excel` does not match
  `Excel`. The ML modes bridge this with a fixed, hand-written synonym table — there is
  no learned embedding, fuzzy matching or spelling correction.
* The TF-IDF corpus is `title + sector + required_skills`; there are no descriptions, so
  similarity reflects wording overlap more than genuine role fit.
* 30 of the 36 location options have no internships in the current dataset and will
  return no location-matched rows.
* Both 10th Pass and 12th Pass candidates are eligible for the same low-level rows by
  design of the hierarchy; only the demo records are written specifically for them.
* Similarity and hybrid scores are ranking values, not probabilities, and the system does
  not predict whether a candidate will be selected for an internship.
* With 115 synthetic records the model is not evidence of real-world ranking quality;
  evaluation is a 9-profile smoke test.
* All listings are synthetic samples. The API flags them (`is_sample`) and the UI labels
  them, but the data itself was not removed or replaced with real postings.
* The hybrid weights are hand-set; no section of the evaluation shows one weighting
  beating another on this dataset.
* No verified external corpus was obtained. The one India-focused source tested is
  templated (24 unique descriptions in 100,000 rows) and has **zero** records for
  10th Pass, 12th Pass, ITI or Diploma, so the catalogue's coverage of those groups is
  still only the 15 hand-written sample rows.
* No candidate–job relevance labels exist in any verified source, so a supervised ranker
  cannot be trained or honestly evaluated.
* Sentence embeddings were measured but rejected: slightly worse ranking on the benchmark
  and +381 MB RAM, which is not justified for a CPU-only deployment.

---

## Testing

```bash
python -m unittest discover -s tests -v
```

93 tests: the rule-based suite, the TF-IDF/hybrid suite, and a pipeline suite covering
skill normalisation, qualification tracks, eligibility, dataset validation, duplicate
detection, model inference and metric maths. They use the Flask test client and the real
CSV; dataset tests skip cleanly when the external sample is absent.

Extra commands:

```bash
python scripts/fetch_dataset.py        # download + provenance manifest
python scripts/audit_dataset.py --csv <file> --name <name>
python scripts/preprocess.py --help    # normalisation into the standard schema
python scripts/experiment_models.py    # model comparison table
```

Offline evaluation (not part of the unit suite):

```bash
python evaluation/evaluate.py
```

---

## Deployment readiness (AWS App Runner)

The app runs unchanged under gunicorn; nothing in the code is AWS-specific and no
credentials, bucket names or service endpoints appear in the source.

**Start command**

```bash
gunicorn -c gunicorn.conf.py app:app
```

`gunicorn.conf.py` binds `0.0.0.0:$PORT` (App Runner injects `PORT`, default 8080), two
workers, two threads, stdout logging, and honours `WEB_CONCURRENCY`,
`GUNICORN_THREADS`, `GUNICORN_TIMEOUT`, `GUNICORN_LOG_LEVEL` overrides.

App Runner health check: path `/health`, HTTP 200.

**Requirements**

* Python 3.11 base image; `pip install -r requirements.txt`. gunicorn is pinned with a
  `sys_platform != "win32"` marker so Windows development installs still work.
* The model is fitted once per worker at import time (~7 ms, ~25 KiB sparse matrix), so
  worker count affects memory only marginally; two workers is plenty.
* `data/internship.csv` must ship with the image — it is read at import and the process
  must run from the repository root.
* No external model API, no GPU, no background jobs.

Minimum App Runner settings: source = repository or container image, build command
`pip install -r requirements.txt`, start command above, port 8080, 1 vCPU / 2 GB is far
more than required.

**Inference footprint.** The deployed app loads Flask + pandas + scikit-learn and a
115 × 1,145 TF-IDF matrix (~25 KiB), with no torch and no embedding model. Measured model
fit is ~7 ms and request latency ~5 ms in-process. 512 MB is comfortable; 1 GB gives
headroom for pandas. The embedding experiment (+381 MB) is deliberately **not** part of
the deployment.

**Verification status:** the WSGI target `app:app` and the Flask routes were exercised
here, and `gunicorn.conf.py` was executed to confirm its values. Gunicorn itself has
**not** been run: it is excluded on Windows and no Linux runtime was available on this
machine (Docker Desktop's Linux engine did not start; the only WSL distro is
`docker-desktop` with no Python). Confirm the start command on the App Runner build log
before treating it as proven.

No deployment was performed and no AWS resource was created from this repository.

---

## Workflow

1. Candidate submits a profile and selects a mode.
2. `app.py` normalises skills, resolves the education level and computes the TF-IDF query.
3. Every internship row is checked for education eligibility first.
4. Eligible rows are scored by the selected mode, ranked and truncated to the top 5.
5. The browser renders each result as a card with its scoring explanation.

## Flowchart

<img width="500" alt="Application flow" src="Flowchart%20(2).png">

## Prototype preview

<img width="500" alt="Frontend sample" src="Frontend%20Sample.png">

---

## Impact and benefits

* Personalised suggestions instead of browsing every listing.
* Lower barrier for candidates from rural, tribal and underserved regions.
* Rule-based scoring keeps the deployment lightweight and explainable.
* The scoring layer can later be extended with a learned model without changing the UI.

---

## References

1. Corné de Ruijt, Sandjai Bhulai (2021) - [Job Recommender Systems: A Review](https://arxiv.org/abs/2111.13576)
2. ITM Conferences (2022) - [Job Recommendation Approaches](https://www.itm-conferences.org/articles/itmconf/abs/2022/04/itmconf_icacc2022_02002/itmconf_icacc2022_02002.htm)
3. G. Deepak - [Applying Data Mining Techniques in Job Recommender System](https://gdeepak.com/pubs/Applying_Data_Mining_Anika.pdf)
