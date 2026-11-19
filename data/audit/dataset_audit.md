# Dataset audit — 2026-11-19

All numbers below were measured on the downloaded file, not taken from the dataset
card. Reproduction: `python scripts/audit_dataset.py --csv <file> --name <name>`.

## 1. Source verification

| Source | Exists | Declared licence | Measured size | India coverage | Text usability | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `Uzaib52/indian-tech-career-intelligence-2026` | Yes (HF API 200) | MIT (card tag) | 100,000 rows × 33 cols (card claims 1,000,000) | 100% (`location_country = India`) | **24 unique descriptions in 100,000 rows** | **Rejected as a training corpus** — see §5 |
| `xanderios/linkedin-job-postings` | Yes | MIT | 33,246 rows | ~0 (US postings: "Little River, SC") | Real descriptions | Rejected — no Indian locations |
| `NextGig-Rocks/global-job-postings-multi-ats` | Yes | CC-BY-4.0 | 112,816 rows | ~0 (Workday/ATS North-America) | Real descriptions | Rejected — no Indian locations |
| `SuhaibAtef/tech-job-postings-labeled` | Yes | **None declared** | 8,000 rows | unverified | unknown | Rejected — reuse rights unclear |

No source with (a) real Indian postings, (b) usable descriptions, (c) a clear licence and
(d) candidate–job relevance labels was found. **No candidate–job relevance labels exist in
any of them**, which blocks the supervised-ranking experiment (Model C) by definition.

## 2. Retrieval record

| Field | Value |
| --- | --- |
| Dataset | `Uzaib52/indian-tech-career-intelligence-2026` |
| File | `indian_tech_career_intelligence_2026.csv` (57 MB) |
| HF revision | `a6182954612f555ea3732cff30f36c8bb7901e4b` |
| SHA-256 | `f69a8de12e0d379419e7f9767c74a366366fabab864d20eb6f414bba21ad403c` |
| Downloaded (UTC) | 2026-11-19T14:24:34Z |
| Licence | MIT (declared in the dataset card; no separate licence file in the repo) |
| Retention | Raw file kept locally only (`data/external/raw/`, git-ignored); 5,000-row deterministic sample retained in `data/processed/` |

## 3. Schema (33 columns, all present)

`job_title, job_category, company_name, company_size, industry, company_type, location_city,
location_state, location_country, work_mode, salary_minimum, salary_maximum, salary_currency,
stipend_amount, experience_level, experience_required, education_required, employment_type,
contract_type, skills_extracted, skill_count, job_description, posted_date,
application_deadline, source_platform, applicant_count, job_benefits, technical_domain,
interview_difficulty, probation_period, shift_type, is_active, career_growth_index`

The source provides **no** application URL, **no** candidate–job relevance label, and **no**
verified employer identity.

## 4. Quality metrics (100,000 rows)

| Metric | Value |
| --- | --- |
| Total records | 100,000 |
| Columns | 33 |
| Columns 100% missing | 0 |
| Columns partially missing | 1 (`probation_period`, 33.5%) |
| Exact duplicate rows | 0 (0.00%) |
| Unique job descriptions | **24** |
| Duplicate description rate | **99.98%** |
| Templated descriptions | **100%** |
| Near-duplicate rate (Jaccard ≥ 0.9, 5-shingles, first 5,000 rows) | **99.52%** |
| Unique descriptions after near-dedup | **24** |
| Unique job titles | 24 |
| Unique companies | 12 |
| Unique `skills_extracted` values | 6 |
| Unique skills after splitting | 39 |
| Unique cities | 8 |
| Industry values | 1 (`Information Technology`) |
| Records marked active | not usable as evidence (synthetic dates 2026) |

Every description follows one sentence template:

> "We are hiring a passionate **{job_title}** to join our engineering cluster. Hands-on
> experience with core tech workflows preferred."

## 5. Distribution and synthetic-vs-observed classification

| Dimension | Distribution |
| --- | --- |
| Education | Bachelor's Degree 33,414 · MCA/MSc 33,349 · B.E./B.Tech 33,237 — exactly 3 values, each ~⅓ |
| Employment type | Full-time 59,871 · Internship 40,129 |
| Work mode | Remote 40,018 · Hybrid 39,654 · On-site 20,328 |
| Country | India 100,000 |
| Cities | Chennai 12,636 · Noida 12,622 · Gurgaon 12,521 · Delhi NCR 12,515 · Pune 12,452 · Mumbai 12,443 · Hyderabad 12,439 · Bangalore 12,372 — near-uniform |
| Top skills | python 33,212 · linux 33,134 · html/css/js/react/node/express/mongodb/tailwind 16,872 each |

**Synthetic: 100%.** Observed: 0%. Evidence:

1. 24 unique descriptions across 100,000 rows, all matching one template.
2. Only 6 distinct `skills_extracted` bundles, identical skill sets repeated per title.
3. Education, work mode, city and employment distributions are near-uniform — the
   signature of a random generator, not a labour market.
4. Only one industry (Information Technology) for a "tech career" dataset.
5. Internships carry full-time salaries (e.g. "Web Development Intern", ₹549,068,
   `employment_type = Full-time`).
6. **Real employer names (TCS, Cognizant, Ola Electric) are attached to fabricated
   postings** — an attribution risk independent of the MIT licence.

## 6. Coverage gaps for the prioritised user groups

| User group | Records in source | Share |
| --- | --- | --- |
| 10th Pass | 0 | 0% |
| 12th Pass | 0 | 0% |
| ITI | 0 | 0% |
| Diploma | 0 | 0% |
| Undergraduate / B.Tech / Graduate | 100,000 | 100% |
| Postgraduate | 0 (MCA/MSc rows normalise to Postgraduate — 33,349) | 33% |

Non-technical sectors: **0 records** (single industry value). No retail, BPO, healthcare,
accounts, logistics or government listings — exactly the segments the 10th/12th/ITI
candidates need.

## 7. Download and retention decision

* The file is **not** committed (57 MB) and is ignored via `.gitignore`.
* A deterministic 5,000-row sample (`random_state=42`) was normalised into
  `data/processed/jobs_sample.csv` **for pipeline validation only**. Every row carries
  `data_status = synthetic_templated` and `is_currently_open = false`.
* **Zero rows were used to train or tune a model.** The TF-IDF/embedding experiments are
  unsupervised and run on the production catalog; the external sample is used only to
  demonstrate why its description column cannot support a text-relevance model
  (5,000 documents collapse to 24 unique TF-IDF vectors; mean cosine between different
  job titles 0.234, max 0.502 — see `data/evaluation/model_comparison.json`).

## 8. Licence and attribution notes

* MIT permits reuse with attribution; the attribution is recorded here and in
  `data/README.md`.
* MIT does **not** resolve the fabrication problem: the dataset must not be presented as
  real job openings, and its employer attributions are unverifiable. That is why it is
  excluded from any user-facing catalogue.
* No personal data (no candidate names, emails, phone numbers or resumes) is present in
  the columns reviewed; `applicant_count` is an aggregate count.
