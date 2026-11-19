"""Qualification normalisation and eligibility for the data pipeline.

The eight requested buckets are mapped to a level **and** a track. Level alone is not
enough: an ITI certificate and a Diploma sit at the same level but are not
interchangeable for every role, so the eligibility rule also looks at the track.

Policies encoded here (documented so they can be reviewed, not silently assumed):

* A higher academic level satisfies a lower academic requirement.
* A vocational requirement (ITI) is satisfied by ITI or by any academic qualification at
  the same level or above — a Diploma/B.Tech holder can take an ITI-level role.
* An academic requirement is **not** satisfied by the vocational track at the same level
  (ITI does not stand in for a Diploma). This is conservative on purpose.
* Anything unrecognised maps to None -> the record/profile is "unknown" and must be
  excluded from eligibility-sensitive ranking until reviewed.
"""
import re

TRACK_ACADEMIC = 'academic'
TRACK_VOCATIONAL = 'vocational'

# canonical label -> (level, track)
QUALIFICATIONS = {
    '10th Pass': (0, TRACK_ACADEMIC),
    '12th Pass': (1, TRACK_ACADEMIC),
    'ITI': (2, TRACK_VOCATIONAL),
    'Diploma': (2, TRACK_ACADEMIC),
    'Undergraduate': (3, TRACK_ACADEMIC),   # currently pursuing a degree
    'B.Tech': (3, TRACK_ACADEMIC),
    'Graduate': (3, TRACK_ACADEMIC),        # any completed bachelor's
    'Postgraduate': (4, TRACK_ACADEMIC),
}

# Freeform text -> canonical label. Order matters: more specific patterns first.
PATTERNS = [
    (r'\biti\b|industrial training institute', 'ITI'),
    (r'\bdiploma\b|polytechnic', 'Diploma'),
    (r'\bph\.?\s?d\b|\bdoctorate\b', 'Postgraduate'),
    (r'\bm\.?tech\b|\bmba\b|\bmca\b|\bm\.?sc\b|\bm\.?com\b|\bm\.?a\b|\bmaster', 'Postgraduate'),
    (r'\bb\.?tech\b|\bb\.?e\.?\b|bachelor of engineering', 'B.Tech'),
    (r'\b10th\b|\bssc\b|matriculation|high school', '10th Pass'),
    (r'\b12th\b|\bhsc\b|higher secondary|intermediate', '12th Pass'),
    (r"\bbachelor|graduate|b\.?sc\b|b\.?com\b|bca\b|bba\b|\bba\b|degree", 'Graduate'),
    (r'\bundergraduate|pursuing|final year|student', 'Undergraduate'),
]

_COMPILED = [(re.compile(pattern, re.IGNORECASE), label) for pattern, label in PATTERNS]


def normalize_qualification(raw):
    """Return a canonical qualification label, or None when unrecognised/missing."""
    text = str(raw or '').strip()
    if not text:
        return None

    # exact canonical label (case-insensitive)
    for label in QUALIFICATIONS:
        if text.lower() == label.lower():
            return label

    for pattern, label in _COMPILED:
        if pattern.search(text):
            return label
    return None


def level_of(label):
    """Numeric level for a canonical label, or None."""
    entry = QUALIFICATIONS.get(label)
    return entry[0] if entry else None


def track_of(label):
    entry = QUALIFICATIONS.get(label)
    return entry[1] if entry else None


def is_eligible(candidate_label, required_label):
    """Eligibility under the documented policy.

    Returns True / False, or None when either side is unknown (caller must treat the
    record as unverified rather than guessing).
    """
    candidate_level = level_of(candidate_label)
    required_level = level_of(required_label)
    if candidate_level is None or required_level is None:
        return None

    candidate_track = track_of(candidate_label)
    required_track = track_of(required_label)

    if candidate_level < required_level:
        return False

    if required_track == TRACK_ACADEMIC and candidate_track == TRACK_VOCATIONAL:
        # ITI does not substitute for a Diploma/lower-degree requirement.
        return candidate_level > required_level

    return True
