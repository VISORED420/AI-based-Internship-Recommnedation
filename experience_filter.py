"""Keep experienced roles out of an internship-focused search.

JSearch exposes no structured experience field on this deployment — 0 of 29 sampled
records carried ``job_required_experience`` — so filtering has two parts:

1. Upstream: ``job_requirements=no_experience`` (a documented `/search-v2` parameter).
   Measured live, it removed every senior title from page 1.
2. Here: a local pass over the title and description, because Google's requirement
   matching is advisory. It drops seniority titles (Senior, Lead, Principal, Staff,
   Architect, Manager, Engineer II/III, SDE 2 ...) and postings that state a minimum
   of more than ``max_years`` years of experience.

Roles whose title says "intern" or "fresher" are never dropped by the text rules —
they are exactly what this product is for.

Pure functions, no Flask objects; every drop carries a reason for auditability.
"""
import os
import re

SENIOR_TITLE_PATTERN = re.compile(
    r'\b(senior|sr\.?|lead|principal|staff|architect|manager|head|director|vp|chief|'
    r'engineer\s+(?:ii|iii|iv|2|3|4)|sde\s*(?:2|3|4))\b',
    re.IGNORECASE,
)

INTERN_TITLE_PATTERN = re.compile(
    r'\b(intern|internship|fresher|trainee|graduate)\b'
    r'|\bno\s+experience\b'
    r'|\bentry[\s-]?level\b'
    r'|\b0\s*(?:-|–|to)\s*\d\s*(?:years?|yrs?)\b',
    re.IGNORECASE,
)

# "5+ years", "3-5 years", "8 to 14yrs", "minimum 2 years", "at least 1 year"
_TITLE_YEARS = re.compile(
    r'(\d{1,2})\s*(?:\+|plus)?\s*(?:-|to|–|—)?\s*(\d{1,2})?\s*(?:years?|yrs?)\b',
    re.IGNORECASE,
)
_DESCRIPTION_YEARS = [
    # "5+ years of experience", "3 years experience"
    re.compile(r'(\d{1,2})\s*(?:\+|plus)?\s*(?:years?|yrs?)\b[^.\n]{0,40}\bexperience',
               re.IGNORECASE),
    # "experience of 4 years", "experience: 2+ years"
    re.compile(r'\bexperience\b[^.\n]{0,40}?(\d{1,2})\s*(?:\+|plus)?\s*(?:years?|yrs?)',
               re.IGNORECASE),
    # "minimum 3 years", "at least 2 years"
    re.compile(r'\b(?:minimum|at\s+least|min\.?)\s+(?:of\s+)?(\d{1,2})\s*\+?\s*'
               r'(?:years?|yrs?)', re.IGNORECASE),
]

DEFAULT_MAX_YEARS = 0


def max_experience_years():
    """Threshold from JOBS_MAX_EXPERIENCE_YEARS (default 0 — no experience at all)."""
    try:
        value = int(os.environ.get('JOBS_MAX_EXPERIENCE_YEARS', ''))
    except (TypeError, ValueError):
        return DEFAULT_MAX_YEARS
    return value if value >= 0 else DEFAULT_MAX_YEARS


def _first_group(match):
    for group in match.groups():
        if group:
            try:
                return int(group)
            except (TypeError, ValueError):
                continue
    return None


def years_mentioned(job):
    """Lowest number of years the posting asks for, or None if it asks for none."""
    title = job.get('title') or ''
    title_match = _TITLE_YEARS.search(title)
    if title_match:
        years = _first_group(title_match)
        if years is not None:
            return years

    if INTERN_TITLE_PATTERN.search(title):
        return None   # "Internship", "Fresher" overrides loose mentions in the body

    description = (job.get('description') or '')[:4000]
    found = []
    for pattern in _DESCRIPTION_YEARS:
        for match in pattern.finditer(description):
            years = _first_group(match)
            if years is not None:
                found.append(years)
    return min(found) if found else None


def experience_signal(job, max_years=None):
    """Return {'experienced': bool, 'min_years': int|None, 'reason': str} for one listing."""
    if max_years is None:
        max_years = max_experience_years()

    title = job.get('title') or ''
    if INTERN_TITLE_PATTERN.search(title):
        return {'experienced': False, 'min_years': None, 'reason': 'title says intern/fresher'}

    seniority = SENIOR_TITLE_PATTERN.search(title)
    if seniority:
        return {'experienced': True, 'min_years': None,
                'reason': f'title contains seniority marker "{seniority.group(0).lower()}"'}

    years = years_mentioned(job)
    if years is not None and years > max_years:
        return {'experienced': True, 'min_years': years,
                'reason': f'asks for {years}+ years of experience'}

    return {'experienced': False, 'min_years': years,
            'reason': 'no experience requirement found'}


def filter_internships(jobs, max_years=None):
    """Split listings into (kept, dropped). Dropped entries carry a reason."""
    kept, dropped = [], []
    for job in jobs:
        signal = experience_signal(job, max_years)
        if signal['experienced']:
            dropped.append({'id': job.get('id'), 'title': job.get('title') or 'Untitled job',
                            'reason': signal['reason']})
        else:
            kept.append(job)
    return kept, dropped
