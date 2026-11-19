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

| Method | Path | Purpose | Auth |
| --- | --- | --- | --- |
| `GET` | `/` | Web interface | login required |
| `GET` | `/health` | Health check: status, model and dataset metadata | public |
| `POST` | `/recommend` | Ranked recommendations (JSON) | login required |
| `GET` | `/login` | Login form | public |
| `POST` | `/login` | Start a session | public |
| `GET` | `/register` | Registration form | public |
| `POST` | `/register` | Create an account and start a session | public |
| `POST` | `/logout` | End the session | login required |
| `GET` | `/api/me` | Current session user as JSON | login required |
| `GET` | `/jobs` | Live job search page (JSearch) | login required |
| `GET` | `/api/jobs` | Proxy one JSearch query as JSON | login required |

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

`/recommend` requires a logged-in session. Anonymous API calls return HTTP 401 with
`{"error": "Authentication required. Please log in."}` rather than an HTML redirect, so
script clients can detect it.

Other failures return HTTP 400 with `{"error": "..."}` — missing or malformed body, blank
profile, unknown mode, and unsupported or missing qualification.

---

## Authentication

Session-based login built on Flask signed cookies — no extra dependency, no server-side
session store.

* **Registration** — `POST /register` with `name`, `email`, `password`, `confirm`.
  Emails are lower-cased and unique; passwords must be at least 8 characters.
* **Hashing** — `werkzeug.security` scrypt hashes (`scrypt:32768:8:1$...`). Plaintext
  passwords are never written to disk.
* **Storage** — SQLite at `data/app.db` (created on first use, git-ignored):
  `users(id, name, email UNIQUE, password_hash, created_at)`.
* **Session** — signed cookie, 7-day lifetime, `HttpOnly`, `SameSite=Lax`. Set
  `COOKIE_SECURE=1` when serving over HTTPS.
* **Gating** — `/` and `/recommend` require login. `/health` stays public so the App
  Runner health check keeps passing. Anonymous page requests are redirected to
  `/login?next=...`; API requests get a JSON 401.
* **Enumeration safety** — wrong password and unknown email return the identical
  message `Incorrect email or password.`

Limitations of this prototype: no email verification, no password reset, no rate
limiting and no CSRF token (the `SameSite=Lax` cookie blocks cross-site form posts, which
covers the current JSON/form routes). Add those before exposing the app to untrusted
traffic.

---

## Live Job Search (JSearch / RapidAPI)

`/jobs` adds a second job source next to the built-in synthetic internship catalogue: real
public postings from JSearch, proxied through this server so the RapidAPI key never reaches
the browser.

```http
GET /api/jobs?q=python+developer&location=bangalore&country=in
    &date_posted=week&employment_type=FULLTIME&work_mode=remote&num_pages=1
# next batch: append the cursor from the previous response
GET /api/jobs?q=python+developer&location=bangalore&country=in&num_pages=1&cursor=<next_cursor>
```

| Parameter | Accepted values | Notes |
| --- | --- | --- |
| `q` | free text, max 120 chars | job title, skill or keyword |
| `location` | free text, max 80 chars | sent as the documented `location` (Google UULE) **and** folded into `query` |
| `country` | ISO-3166 alpha-2 (`in`, `us`, `gb`, ...) | defaults to `JSEARCH_DEFAULT_COUNTRY` (`in`) |
| `date_posted` | `all`, `today`, `3days`, `week`, `month` | passed straight to JSearch |
| `employment_type` | `FULLTIME`, `PARTTIME`, `CONTRACTOR`, `INTERN` (CSV) | passed straight to JSearch |
| `work_mode` | `all`, `remote`, `onsite` | `remote` sends `work_from_home=true`; `onsite` filters on `job_is_remote` |
| `cursor` | opaque string from `next_cursor` | cursor pagination (documented for `/search-v2`) |
| `page` | 1-20 | only sent to the legacy `/search` endpoint |
| `num_pages` | 1-20 | page + num_pages must not exceed 20 |

Anything outside those enumerations is rejected with HTTP 400 naming the accepted values —
filters are never silently dropped. `q` and `location` cannot both be empty.

### Which upstream endpoint actually answers

Verified against the live API with a real key on this deployment:

| Endpoint | Result |
| --- | --- |
| `GET /search` | **404** `{"message":"Endpoint '/search' does not exist"}` |
| `GET /search-v2` | **200**, `data: {jobs: [...], cursor: "..."}` |

So JSearch now serves cursor-paginated `/search-v2` only. `jobs_api.py` calls `/search-v2`
first, sends `cursor` for the next batch, and falls back to the legacy `/search` (with `page`)
if a deployment answers 404 for `/search-v2`. If both 404, the route returns 502 with a
message naming the subscription. The response field `endpoint` tells you which one answered.

Documented parameters are filtered per endpoint: `/search-v2` never receives `page`, and
`/search` never receives `cursor` or `location`.

`location` is Google's UULE search origin. Measured against the live API with a real key:
`query="python developer in bangalore"` alone returned a mix across sources, while adding
`location=Bangalore, Karnataka, India` returned 10/10 Bengaluru postings with **zero overlap**
with the previous set. Raw city input (`bangalore`, `Bengaluru, Karnataka, India`) resolves
correctly; a location Google does not recognise (`zzz`) returns zero jobs, so the UI empty
state matters here.

Documented but deliberately **not** exposed: `job_requirements` and `radius` (no matching
control in the requested filter set), `exclude_job_publishers` and `fields` (no product need).
Add them to `_upstream_params()` if a control is introduced.

### Where the API key lives

`RAPIDAPI_KEY` is read on the server only (`jobs_api.py`) and is never included in any
response, template or static file. The browser calls `/api/jobs`; that route is what talks to
`https://jsearch.p.rapidapi.com/search`.

Create `.env` next to `app.py` (`.env` is git-ignored, `.env.example` is committed):

```bash
cp .env.example .env      # then put your real key in it
# RAPIDAPI_KEY=YOUR_RAPIDAPI_KEY
```

`env_loader.load_env()` reads that file at startup and **never overrides** variables already
set in the real environment, so App Runner / systemd settings win.

### Quota protection

* In-process TTL cache (default 600 s, `JOBS_CACHE_TTL`) keyed on the normalised query;
  identical searches — including different users' — cost one API credit.
* Cache holds max 256 entries, evicting expired then oldest first.
* Client-side 450 ms debounce on the keyword/location inputs, and no request fires until the
  user has searched once (no API call on page load).
* 12 s upstream timeout (`JOBS_API_TIMEOUT`), and every upstream failure maps to an explicit
  JSON status: 429 rate limit (with `Retry-After: 60`), 503 bad/missing key, 502 provider
  error, 504 timeout.

### Entry level only (default on)

This is an internship product, so listings that demand experience are removed before anything is
shown. Two layers, because the payload carries no structured experience field (0 of 29 sampled
records had `job_required_experience`):

1. **Upstream** — `job_requirements=no_experience`, a documented `/search-v2` parameter. Measured
   live, it removed every senior title from page 1 and returned fresher-oriented postings.
2. **Local** (`experience_filter.py`) — Google's requirement matching is advisory, so the filter is
   repeated on our side: seniority titles (Senior, Sr., Lead, Principal, Staff, Architect, Manager,
   Engineer II/III, SDE 2/3/4), years stated in the title (`|8 to 14yrs|`), and descriptions that
   ask for `N+ years of experience` / `experience of N years` / `minimum N years`.

Titles saying intern / internship / fresher / trainee / graduate / entry-level / "no experience"
override body mentions — an internship posting that happens to say "our team has 10 years of
experience" is kept. Every drop is reported in `filtered_samples` with a reason, and the UI shows
"N roles hidden (experience required)".

| Setting | Default | Meaning |
| --- | --- | --- |
| `entry_level` (query param) | `1` | `0` disables both layers and returns everything |
| `JOBS_MAX_EXPERIENCE_YEARS` (env) | `0` | years still tolerated by the text rule (`1` keeps "1 year" roles) |

Live example (`q=software developer`, Bangalore): default → 6 fresher listings, 4 hidden with
reasons (`title contains seniority marker "senior"`, `asks for 1+ years of experience`);
`entry_level=0` → the same 10 results including "Java developer … |5+ years|" and
"Senior Software Engineer". `q=internship developer` → 4 intern listings, 0 hidden.

### Matching listings against the candidate profile

`/jobs` accepts the same profile the recommender form collects, and every returned listing is
scored against it (score is computed from cached provider data, so it costs no extra API credit):

```http
GET /jobs?skills=python,sql,django&qualification=B.Tech&sector=Information+Technology&location=Karnataka
GET /api/jobs?q=python&skills=python,sql&sector=Information+Technology
```

How a listing is scored (`job_matching.py`, pure functions):

| Component | Weight | Meaning |
| --- | --- | --- |
| Skill coverage | 55% | share of profile skills found in the posting text |
| TF-IDF cosine | 35% | similarity between the skill text and title + description |
| Sector mention | 10% | the preferred sector appears in the posting |

Skill matching reuses the recommender's normalisation and synonym table, so `MySQL` counts as
`sql` and `ReactJS` as `react`, and matching is word-boundary aware (`java` does not match
`javascript`). Each card reports the percentage, the skills that matched, and a sentence naming
the numbers used. When a profile is present, results are sorted best match first; the search is
unaffected when no profile is sent (`match` is `null`).

Arriving from the recommender pre-fills the keyword with the first two profile skills — measured
live, a three-skill query returned 1 result where two skills returned 10.

What cannot be matched: live JSearch records carry no structured skill list
(`job_required_skills` is empty on every result sampled) and no education requirement, so skill
matching runs over the description text and your qualification is advisory only — it is not used
to filter or score live postings, and the UI says so.

### Honest limitations

* JSearch exposes `work_from_home` and `job_is_remote`; it has **no hybrid flag**, so the UI
  offers remote and on-site only and says so.
* The search payload already contains the full description, so no extra `/job-details` call
  (and no extra credit) is spent to show details.
* Live results require a subscribed key. Without one the page renders a clear configuration
  banner and `/api/jobs` returns 503 `not_configured`.

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
auth.py                       Login backend: SQLite users, password hashing, session gating
jobs_api.py                   JSearch client: validation, TTL cache, normalisation, error mapping
env_loader.py                 Minimal .env reader (stdlib, never overrides real env vars)
templates/jobs.html           Live job search page, filters and profile strip
job_matching.py               Profile scoring for live listings (skills + TF-IDF + sector)
experience_filter.py          Drops postings that require experience (internships only)
static/jobs.js                Debounced search, filters, load-more, error/empty states
tests/test_jobs_api.py        JSearch integration tests (mocked upstream)
tests/test_job_matching.py    Profile scoring and its /api/jobs integration
.env.example                  Documented environment variables without secrets
templates/login.html          Login form
templates/register.html       Registration form
tests/test_auth.py            Registration, login, logout and route-gating tests
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
| `SECRET_KEY` | generated | Session signing key; generated once into `data/secret_key` if unset |
| `COOKIE_SECURE` | off | `1` marks the session cookie `Secure` (use behind HTTPS) |
| `S2I_DB_PATH` | `data/app.db` | SQLite file holding user accounts |
| `RAPIDAPI_KEY` | *(none)* | RapidAPI key for JSearch; without it `/api/jobs` returns 503 |
| `JSEARCH_DEFAULT_COUNTRY` | `in` | Country used when the request omits `country` |
| `JOBS_CACHE_TTL` | `600` | Seconds a job search stays cached (`0` disables) |
| `JOBS_API_TIMEOUT` | `12` | Upstream JSearch request timeout in seconds |
| `JOBS_MAX_EXPERIENCE_YEARS` | `0` | Years tolerated by the entry-level text filter |

```bash
PORT=8000 FLASK_DEBUG=1 python app.py
```

The development server logs every request. Lines reporting `"POST /" 405` come from other
software probing port 5000 (Steam's client does this continuously) — no route here accepts
`POST /`. `quiet_probe_noise()` drops exactly that pattern from the Werkzeug access log; the
response is still 405 and every real request is still logged.

Production (Linux):

```bash
gunicorn -c gunicorn.conf.py app:app
```

---

## Testing

```bash
python -m unittest discover -s tests -v
```

231 tests cover registration, login, logout, password hashing, route gating, the rule
engine, TF-IDF vectorization, cosine similarity, skill normalization and synonyms,
education eligibility, qualification validation, duplicate suppression, ranking order,
the API contract, the health endpoint, the data pipeline, and the JSearch integration
(parameter validation, cursor pagination, endpoint fallback, upstream error mapping, caching,
the `/api/jobs` route, and profile scoring). Upstream HTTP is mocked in the JSearch tests, so no
network or key is needed to run them.

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
