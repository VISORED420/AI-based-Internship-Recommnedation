"""TF-IDF relevance model for the internship recommender.

The dataset has no free-text description column, so each internship's document is
built from the columns that do exist: title, sector and required skills.

The vectorizer is fitted exactly once per process, when the module user calls
build_model() at startup; scoring a request only transforms the query and does one
cosine-similarity multiplication (no refitting).

Nothing here talks to the network or to an external model API.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Common ways applicants write a skill -> the wording used in the dataset.
# Applied to both the internship documents and the candidate query so the two
# sides stay in the same vocabulary.
SKILL_SYNONYMS = {
    'ms excel': 'excel',
    'microsoft excel': 'excel',
    'advanced excel': 'excel',
    'excel sheet': 'excel',
    'spreadsheet': 'excel',
    'basic computer knowledge': 'basic computer',
    'basic computer skills': 'basic computer',
    'computer basics': 'basic computer',
    'computer knowledge': 'basic computer',
    'computer literacy': 'basic computer',
    'data entry operation': 'data entry',
    'data entry operator': 'data entry',
    'communication skills': 'communication',
    'communications': 'communication',
    'verbal communication': 'communication',
    'customer support': 'customer service',
    'customer care': 'customer service',
    'client service': 'customer service',
    'typing speed': 'typing',
    'technical support': 'troubleshooting',
    'hardware troubleshooting': 'troubleshooting',
    'inventory management': 'inventory',
    'stock management': 'inventory',
    'tally erp': 'tally',
    'tally erp 9': 'tally',
    'mysql': 'sql',
    'postgresql': 'sql',
    'sql server': 'sql',
    'python3': 'python',
    'reactjs': 'react',
    'node js': 'node.js',
}

# Longest phrases first so "basic computer knowledge" wins over "basic computer".
_SYNONYM_PATTERNS = [
    (re.compile(r'\b' + re.escape(phrase) + r'\b'), canonical)
    for phrase, canonical in sorted(SKILL_SYNONYMS.items(), key=lambda item: -len(item[0]))
]

_TOKEN_CLEANUP = re.compile(r'[^a-z0-9+#.\- ]')
_WHITESPACE = re.compile(r'\s+')


def normalize_text(text):
    """Lower-case, strip punctuation, apply skill synonyms and collapse whitespace."""
    value = str(text or '').lower()
    value = _TOKEN_CLEANUP.sub(' ', value)
    value = _WHITESPACE.sub(' ', value).strip()
    for pattern, canonical in _SYNONYM_PATTERNS:
        value = pattern.sub(canonical, value)
    return _WHITESPACE.sub(' ', value).strip()


def synonym_expansions(skills_text):
    """Return [(input_phrase, canonical_skill)] for phrases that only match via synonyms.

    Each comma-separated chunk is normalised; if the result differs from the chunk the
    user typed, the difference came from the synonym table, which is what the match
    explanation reports back to the candidate.
    """
    expansions = []
    for chunk in str(skills_text or '').split(','):
        phrase = ' '.join(chunk.lower().split())
        if not phrase:
            continue
        canonical = normalize_text(phrase)
        if canonical and canonical != phrase:
            expansions.append((phrase, canonical))
    return expansions


class TfidfRecommender:
    """Cosine-similarity ranker over TF-IDF vectors of internship documents."""

    def __init__(self, internships, fit=True):
        self.internship_ids = list(internships['internship_id'])
        self.documents = [self._make_document(row) for row in internships.itertuples(index=False)]
        self.canonical_skills_by_id = {
            internship_id: self._canonical_skills(skills)
            for internship_id, skills in zip(internships['internship_id'],
                                             internships['required_skills'])
        }
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),         # unigrams and bigrams
            stop_words='english',
            sublinear_tf=True,
            min_df=1,
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z0-9.+#\-]{1,}\b',
        )
        self.matrix = None
        self.fitted = False
        if fit and self.documents:
            self.matrix = self.vectorizer.fit_transform(self.documents)
            self.fitted = True

    @staticmethod
    def _make_document(row):
        title = getattr(row, 'title', '') or ''
        sector = getattr(row, 'sector', '') or ''
        skills = getattr(row, 'required_skills', '') or ''
        return normalize_text(f'{title} {sector} {skills}')

    @staticmethod
    def _canonical_skills(required_skills):
        """Normalised skill set for one row, used to explain synonym matches."""
        skills = set()
        for part in str(required_skills or '').split(','):
            canonical = normalize_text(part)
            if canonical:
                skills.add(canonical)
        return skills

    @property
    def vocabulary_size(self):
        return len(self.vectorizer.vocabulary_) if self.fitted else 0

    @property
    def document_count(self):
        return len(self.internship_ids)

    def similarities(self, query_text):
        """Cosine similarity of the candidate's skills against every internship document.

        Returns a float array aligned with `self.internship_ids`, or zeros when the
        query has no usable tokens (an empty query is not a perfect match).
        """
        if not self.fitted:
            return np.zeros(len(self.internship_ids), dtype=float)

        query = normalize_text(query_text)
        if not query:
            return np.zeros(len(self.internship_ids), dtype=float)

        query_vector = self.vectorizer.transform([query])
        if query_vector.nnz == 0:
            return np.zeros(len(self.internship_ids), dtype=float)

        return cosine_similarity(query_vector, self.matrix).ravel()

    def similarity_by_id(self, query_text):
        """Same as similarities(), keyed by internship_id."""
        return dict(zip(self.internship_ids, self.similarities(query_text)))


def build_model(internships):
    """Fit the model once. Call at application startup, not per request."""
    return TfidfRecommender(internships)
