"""Profile matching for live JSearch listings.

The internship recommender ranks rows from ``data/internship.csv``; live postings
come from JSearch and carry no structured skill list (``job_required_skills`` is
empty on every result I sampled). So the candidate profile is matched against the
posting *text* — title, company and description — using the same normalisation and
synonym table as the recommender, plus TF-IDF cosine similarity.

Pure functions, no Flask objects, so the scoring is unit-testable and the same
code path serves the API and the page.

Score = 55% skill coverage + 35% TF-IDF text similarity + 10% sector mention.
Every component is reported back so the UI can explain the number.
"""
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ml_recommender import normalize_text

WEIGHT_SKILLS = 0.55
WEIGHT_TEXT = 0.35
WEIGHT_SECTOR = 0.10

MAX_SKILLS = 12
MAX_DESCRIPTION_FOR_MATCHING = 4000   # truncate before vectorising; cost control


def parse_skills(skills):
    """Split comma-separated input into normalised, de-duplicated skill phrases."""
    phrases = []
    for chunk in str(skills or '').split(','):
        phrase = normalize_text(chunk)
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return phrases[:MAX_SKILLS]


def build_profile(skills='', qualification='', sector='', location=''):
    """Describe the candidate. ``active`` is False when there is nothing to match on."""
    phrases = parse_skills(skills)
    sector_text = normalize_text(sector)
    return {
        'skills': phrases,
        'skill_text': ' '.join(phrases),
        'qualification': ' '.join(str(qualification or '').split()),
        'sector': sector_text,
        'location': ' '.join(str(location or '').split()),
        'active': bool(phrases or sector_text),
    }


def _job_document(job):
    parts = [job.get('title'), job.get('company'),
             (job.get('description') or '')[:MAX_DESCRIPTION_FOR_MATCHING]]
    return normalize_text(' '.join(part for part in parts if part))


def _skill_hits(document, phrases):
    """Skills present in the posting text, matched on word boundaries.

    The document is normalised first, so a posting that says "MySQL" is scored as
    "sql" and one that says "ReactJS" as "react" — the same synonym table the
    internship recommender uses.
    """
    hits = []
    for phrase in phrases:
        if not phrase:
            continue
        pattern = r'(?<![a-z0-9])' + re.escape(phrase) + r'(?![a-z0-9])'
        if re.search(pattern, document):
            hits.append(phrase)
    return hits


def score_job(job, profile, similarity=0.0):
    """Return a copy of ``job`` with match fields attached."""
    placed = dict(job)
    if not profile.get('active'):
        placed['match'] = None
        return placed

    document = _job_document(job)
    skills = profile['skills']
    hits = _skill_hits(document, skills)
    coverage = (len(hits) / len(skills)) if skills else 0.0
    sector_match = bool(profile['sector']) and profile['sector'] in document

    score = (WEIGHT_SKILLS * coverage
             + WEIGHT_TEXT * max(0.0, min(1.0, similarity))
             + (WEIGHT_SECTOR if sector_match else 0.0))

    parts = []
    if skills:
        parts.append(f'matches {len(hits)} of {len(skills)} skills'
                     + (f" ({', '.join(hits)})" if hits else ''))
    if sector_match:
        parts.append(f'sector "{profile["sector"]}" appears in the posting')
    parts.append(f'text similarity {similarity:.2f}')

    placed['match'] = {
        'score': round(score, 3),
        'percent': int(round(score * 100)),
        'matched_skills': hits,
        'missing_skills': [skill for skill in skills if skill not in hits],
        'skill_coverage': round(coverage, 3),
        'text_similarity': round(similarity, 3),
        'sector_match': sector_match,
        'explanation': '; '.join(parts).capitalize() + '.',
    }
    return placed


def _similarities(jobs, profile):
    """Cosine similarity between the profile's skill text and each posting."""
    if not profile['skills'] or not jobs:
        return [0.0] * len(jobs)

    documents = [_job_document(job) for job in jobs]
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english',
                                     sublinear_tf=True)
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        # Empty vocabulary: nothing but stop words in the postings.
        return [0.0] * len(jobs)

    query = vectorizer.transform([profile['skill_text']])
    return [float(value) for value in cosine_similarity(query, matrix)[0]]


def rank_jobs(jobs, profile):
    """Attach match data and sort best-first when a profile is active."""
    if not profile.get('active'):
        return [score_job(job, profile) for job in jobs]

    similarities = _similarities(jobs, profile)
    scored = [score_job(job, profile, similarity) for job, similarity in zip(jobs, similarities)]
    # Stable sort: equal scores keep the provider's own ordering.
    return sorted(scored, key=lambda job: job['match']['score'], reverse=True)
