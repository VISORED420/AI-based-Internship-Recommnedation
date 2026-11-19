"""Offline model variants for the recommendation experiment.

These are evaluated by scripts/experiment_models.py and are NOT wired into the Flask
app. A variant only gets promoted to an optional API mode if it beats the baseline on
the evaluation set.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _clean(text):
    value = str(text or '').lower()
    value = re.sub(r'<[^>]+>', ' ', value)
    value = re.sub(r'[^a-z0-9+#./\- ]', ' ', value)
    return re.sub(r'\s+', ' ', value).strip()


class TfidfVariant:
    """Configurable TF-IDF ranker so feature/ngram choices can be compared."""

    def __init__(self, name, fields, ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                 skills_repeat=1, stop_words='english'):
        self.name = name
        self.fields = tuple(fields)
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.skills_repeat = skills_repeat
        self.stop_words = stop_words
        self.vectorizer = None
        self.matrix = None

    def _document(self, record):
        parts = []
        for field in self.fields:
            value = _clean(record.get(field, ''))
            if not value:
                continue
            repeat = self.skills_repeat if field == 'required_skills' else 1
            parts.extend([value] * repeat)
        return ' '.join(parts)

    def fit(self, frame):
        documents = [self._document(record) for record in frame.to_dict('records')]
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
            stop_words=self.stop_words,
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z0-9.+#\-]{1,}\b',
        )
        self.matrix = self.vectorizer.fit_transform(documents)
        return self

    def scores(self, query_text):
        if self.matrix is None:
            raise RuntimeError('fit() must be called first')
        query = _clean(query_text)
        if not query:
            return np.zeros(self.matrix.shape[0])
        vector = self.vectorizer.transform([query])
        if vector.nnz == 0:
            return np.zeros(self.matrix.shape[0])
        return cosine_similarity(vector, self.matrix).ravel()


class SentenceEmbeddingVariant:
    """Sentence-Transformer ranker. Lazy-loads the model; records its own cost."""

    def __init__(self, model_id='sentence-transformers/all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer  # local import: heavy
        self.name = f'embeddings:{model_id.split("/")[-1]}'
        self.model = SentenceTransformer(model_id)
        self.embeddings = None

    def _document(self, record):
        return _clean(' '.join(str(record.get(field, '')) for field in
                               ('title', 'sector', 'required_skills')))

    def fit(self, frame):
        documents = [self._document(record) for record in frame.to_dict('records')]
        self.embeddings = self.model.encode(documents, batch_size=32,
                                            show_progress_bar=False, convert_to_numpy=True)
        return self

    def scores(self, query_text):
        query = self.model.encode([_clean(query_text)], convert_to_numpy=True)
        return cosine_similarity(query, self.embeddings).ravel()


class TfidfOnDescriptions:
    """Fits on the external dataset's description column to measure whether it carries signal."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english',
                                          token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z0-9.+#\-]{1,}\b')
        self.matrix = None

    def fit(self, frame):
        documents = [_clean(value) for value in frame['description']]
        self.matrix = self.vectorizer.fit_transform(documents)
        self.titles = list(frame['title'])
        return self

    def diagnostics(self):
        """How much information the description column actually contains."""
        matrix = self.matrix
        unique_rows = len({row for row in map(str, matrix.toarray().round(4))})
        titles = self.titles
        # mean cosine between descriptions belonging to different job titles
        index_by_title = {}
        for position, title in enumerate(titles):
            index_by_title.setdefault(title, []).append(position)
        cross = []
        title_keys = list(index_by_title)[:24]
        for i, left in enumerate(title_keys):
            for right in title_keys[i + 1:]:
                a = index_by_title[left][0]
                b = index_by_title[right][0]
                cross.append(float(cosine_similarity(matrix[a], matrix[b])[0][0]))
        return {
            'documents': int(matrix.shape[0]),
            'unique_description_vectors': unique_rows,
            'mean_cosine_between_different_titles': round(float(np.mean(cross)), 4) if cross else None,
            'max_cosine_between_different_titles': round(float(np.max(cross)), 4) if cross else None,
            'titles_compared': len(title_keys),
        }
