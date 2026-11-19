"""Tests for the TF-IDF recommender, the hybrid ranking and the /recommend API.

Run from the repository root:

    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import app as flask_app  # noqa: E402
import _auth_support  # noqa: E402
from ml_recommender import TfidfRecommender, build_model, normalize_text  # noqa: E402


def _level(text):
    return flask_app.parse_education_level(text)


class TfidfModelTests(unittest.TestCase):

    def test_model_is_fitted_once_at_startup(self):
        model = flask_app.ml_model
        self.assertIsInstance(model, TfidfRecommender)
        self.assertTrue(model.fitted)
        self.assertEqual(model.document_count, len(flask_app.internships_df))
        self.assertGreater(model.vocabulary_size, 0)

    def test_vectorizer_uses_unigrams_and_bigrams(self):
        self.assertEqual(flask_app.ml_model.vectorizer.ngram_range, (1, 2))
        matrix = flask_app.ml_model.matrix
        self.assertEqual(matrix.shape[0], flask_app.ml_model.document_count)

    def test_similarities_are_cosine_values_per_document(self):
        model = flask_app.ml_model
        sims = model.similarities('Python, SQL')
        self.assertEqual(len(sims), model.document_count)
        self.assertGreaterEqual(float(sims.min()), 0.0)
        self.assertLessEqual(float(sims.max()), 1.0 + 1e-9)

    def test_identical_text_scores_higher_than_unrelated_text(self):
        model = flask_app.ml_model
        related = model.similarity_by_id('Python, Java, Git, SQL, Data Structures')
        unrelated = model.similarity_by_id('inventory, warehouse ops')
        self.assertGreater(related['INT-IT-25-001'], unrelated['INT-IT-25-001'])
        self.assertGreater(related['INT-IT-25-001'], 0.5)

    def test_unseen_vocabulary_scores_zero(self):
        model = flask_app.ml_model
        sims = model.similarities('Underwater Basket Weaving')
        self.assertEqual(float(sims.max()), 0.0)

    def test_empty_query_scores_zero(self):
        self.assertEqual(float(flask_app.ml_model.similarities('').max()), 0.0)

    def test_model_is_not_rebuilt_between_requests(self):
        model = flask_app.ml_model
        vectorizer_id = id(model.vectorizer)
        matrix_id = id(model.matrix)
        flask_app.rank_internships(qualification='B.Tech', skills='Python', mode='hybrid')
        flask_app.rank_internships(qualification='10th Pass', skills='Excel', mode='ml')
        self.assertIs(model, flask_app.ml_model)
        self.assertEqual(id(model.vectorizer), vectorizer_id)
        self.assertEqual(id(model.matrix), matrix_id)

    def test_build_model_is_independent_of_the_running_app(self):
        small = flask_app.internships_df.head(5).copy()
        model = build_model(small)
        self.assertEqual(model.document_count, 5)
        self.assertEqual(len(model.similarities('python')), 5)


class NormalisationTests(unittest.TestCase):

    def test_case_and_whitespace_normalisation(self):
        self.assertEqual(normalize_text('  Python   ,  SQL  '), 'python sql')
        self.assertEqual(normalize_text('BASIC   COMPUTER'), 'basic computer')

    def test_synonyms_map_to_dataset_vocabulary(self):
        self.assertEqual(normalize_text('MS Excel'), 'excel')
        self.assertEqual(normalize_text('Basic Computer Knowledge'), 'basic computer')
        self.assertEqual(normalize_text('Customer Support'), 'customer service')
        self.assertEqual(normalize_text('MySQL'), 'sql')

    def test_synonym_query_finds_rows_the_rule_engine_cannot(self):
        ml_rows = flask_app.rank_internships(qualification='12th Pass',
                                             skills='MS Excel, Computer Basics', mode='ml')
        rule_rows = flask_app.rank_internships(qualification='12th Pass',
                                               skills='MS Excel, Computer Basics', mode='rule')
        self.assertEqual(rule_rows, [])
        self.assertTrue(ml_rows)
        self.assertGreater(len(ml_rows), 0)


class RankingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = _auth_support.logged_in_client()

    def test_education_eligibility_holds_in_every_mode(self):
        for qualification in ('10th Pass', '12th Pass', 'Diploma'):
            candidate_level = _level(qualification)
            for mode in flask_app.MODES:
                rows = flask_app.rank_internships(qualification=qualification,
                                                  skills='Excel, Communication', mode=mode)
                for row in rows:
                    self.assertLessEqual(_level(row['required_education']), candidate_level,
                                         f'{mode}: {row["internship_id"]}')

    def test_ml_similarity_cannot_override_education(self):
        # 'pandas' only exists on a level-4 record; a 10th Pass candidate must not see it.
        for mode in flask_app.MODES:
            ids = [r['internship_id'] for r in flask_app.rank_internships(
                qualification='10th Pass', skills='pandas', mode=mode)]
            self.assertNotIn('INT-DS-25-004', ids)

    def test_ml_results_are_ordered_by_similarity(self):
        rows = flask_app.rank_internships(qualification='B.Tech', skills='Python, SQL', mode='ml')
        self.assertGreater(len(rows), 1)
        scores = [r['score'] for r in rows]
        self.assertEqual(scores, sorted(scores, reverse=True), scores)

    def test_hybrid_results_are_ordered_by_hybrid_score(self):
        rows = flask_app.rank_internships(qualification='B.Tech', skills='Python, SQL', mode='hybrid')
        scores = [r['score'] for r in rows]
        self.assertEqual(scores, sorted(scores, reverse=True), scores)

    def test_hybrid_score_is_the_weighted_sum_of_its_parts(self):
        rows = flask_app.rank_internships(qualification='B.Tech', skills='Python, SQL', mode='hybrid')
        for row in rows:
            rule_norm = min(row['rule_score'] / flask_app.RULE_SCORE_SCALE, 1.0)
            expected = flask_app.HYBRID_ML_WEIGHT * row['tfidf_similarity'] + \
                flask_app.HYBRID_RULE_WEIGHT * rule_norm
            self.assertAlmostEqual(row['hybrid_score'], expected, places=2)

    def test_hybrid_weights_change_the_score_not_the_eligibility(self):
        default = flask_app.rank_internships(qualification='B.Tech', skills='Python, SQL',
                                             mode='hybrid', ml_weight=0.6)
        rule_heavy = flask_app.rank_internships(qualification='B.Tech', skills='Python, SQL',
                                                mode='hybrid', ml_weight=0.2)
        self.assertTrue(default and rule_heavy)
        for row in default:
            self.assertLessEqual(_level(row['required_education']), 3)
        for row in rule_heavy:
            self.assertLessEqual(_level(row['required_education']), 3)
        self.assertNotEqual([r['score'] for r in default], [r['score'] for r in rule_heavy])

    def test_no_duplicate_recommendations_in_any_mode(self):
        for mode in flask_app.MODES:
            for skills in ('Python, SQL', 'Excel, Data Entry', 'Troubleshooting'):
                ids = [r['internship_id'] for r in flask_app.rank_internships(
                    qualification='Diploma', skills=skills, mode=mode)]
                self.assertEqual(len(ids), len(set(ids)), (mode, skills))

    def test_at_most_five_results(self):
        for mode in flask_app.MODES:
            rows = flask_app.rank_internships(qualification='MBA',
                                              skills='Excel, Python, SQL, Communication',
                                              mode=mode)
            self.assertLessEqual(len(rows), flask_app.MAX_RESULTS)

    def test_empty_results_when_nothing_is_relevant(self):
        for mode in flask_app.MODES:
            rows = flask_app.rank_internships(qualification='B.Tech',
                                              skills='Underwater Basket Weaving', mode=mode)
            self.assertEqual(rows, [], mode)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            flask_app.rank_internships(qualification='B.Tech', skills='Python', mode='magic')

    def test_unknown_qualification_raises(self):
        with self.assertRaises(ValueError):
            flask_app.rank_internships(qualification='B.Voc in Retail',
                                       skills='Python, SQL', mode='hybrid')

    def test_empty_qualification_raises(self):
        with self.assertRaises(ValueError):
            flask_app.rank_internships(qualification='', skills='Python', mode='rule')
        with self.assertRaises(ValueError):
            flask_app.rank_internships(qualification='   ', skills='Python', mode='ml')

    def test_freeform_qualification_still_equals_plain(self):
        freeform = flask_app.rank_internships(qualification='Pursuing B.Tech in CS',
                                              skills='Python, SQL', mode='hybrid')
        plain = flask_app.rank_internships(qualification='B.Tech',
                                           skills='Python, SQL', mode='hybrid')
        self.assertEqual([r['internship_id'] for r in freeform],
                         [r['internship_id'] for r in plain])

    def test_every_record_is_flagged_as_sample_data(self):
        rows = flask_app.rank_internships(qualification='B.Tech', skills='Python', mode='hybrid')
        self.assertTrue(rows)
        for row in rows:
            self.assertIs(row['is_sample'], True)


class RecommendApiMlTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = _auth_support.logged_in_client()

    def post(self, **payload):
        return self.client.post('/recommend', json=payload)

    def test_default_mode_is_still_rule_based(self):
        rows = self.post(qualification='B.Tech', skills='Python, SQL').get_json()
        self.assertTrue(rows)
        self.assertTrue(all(r['mode'] == 'rule' for r in rows))

    def test_each_mode_is_accepted_and_echoed(self):
        for mode in flask_app.MODES:
            response = self.post(qualification='B.Tech', skills='Python, SQL', mode=mode)
            self.assertEqual(response.status_code, 200)
            rows = response.get_json()
            self.assertTrue(rows, mode)
            self.assertTrue(all(r['mode'] == mode for r in rows), mode)

    def test_unknown_mode_returns_json_400(self):
        response = self.post(qualification='B.Tech', skills='Python', mode='magic')
        self.assertEqual(response.status_code, 400)
        self.assertTrue(response.content_type.startswith('application/json'))
        self.assertIn('error', response.get_json())

    def test_hybrid_response_carries_the_explanation_fields(self):
        rows = self.post(qualification='B.Tech', skills='Python, SQL',
                         sector_interested='Information Technology', mode='hybrid').get_json()
        self.assertTrue(rows)
        first = rows[0]
        for key in ('score', 'mode', 'rule_score', 'tfidf_similarity', 'hybrid_score',
                    'matched_skills', 'matched_sector', 'matched_location', 'explanation'):
            self.assertIn(key, first, key)
        self.assertIn('Hybrid score', first['explanation'])
        self.assertIn('text similarity', first['explanation'].lower())
        self.assertNotIn('probability', first['explanation'].lower())

    def test_ml_mode_explanation_names_text_similarity(self):
        rows = self.post(qualification='B.Tech', skills='Python, SQL', mode='ml').get_json()
        self.assertTrue(rows)
        self.assertIn('Text similarity', rows[0]['explanation'])

    def test_api_never_returns_ineligible_records(self):
        for mode in flask_app.MODES:
            rows = self.post(qualification='10th Pass', skills='Excel, Data Entry',
                             mode=mode).get_json()
            for row in rows:
                self.assertLessEqual(_level(row['required_education']), 0, row['internship_id'])

    def test_api_empty_results_is_an_empty_list(self):
        for mode in flask_app.MODES:
            response = self.post(qualification='B.Tech', skills='Underwater Basket Weaving',
                                 mode=mode)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json(), [])

    def test_health_endpoint(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content_type.startswith('application/json'))
        body = response.get_json()
        self.assertEqual(body['status'], 'ok')
        self.assertTrue(body['model']['fitted'])
        self.assertEqual(body['model']['documents'], len(flask_app.internships_df))
        self.assertEqual(body['dataset']['records'], len(flask_app.internships_df))
        self.assertTrue(body['dataset']['synthetic'])

    def test_synonym_match_is_explained_to_the_user(self):
        rows = self.post(qualification='12th Pass', skills='MS Excel, Computer Basics',
                         mode='hybrid').get_json()
        self.assertTrue(rows)
        explained = [row for row in rows if row['synonym_matches']]
        self.assertTrue(explained, 'expected at least one synonym-based match')

        first = explained[0]
        pair = first['synonym_matches'][0]
        canonical = flask_app.ml_model.canonical_skills_by_id[first['internship_id']]

        self.assertIn(pair['dataset_skill'], canonical)
        self.assertIn(pair['input'], first['explanation'])
        self.assertIn(pair['dataset_skill'], first['explanation'])
        self.assertIn('synonym', first['explanation'].lower())

    def test_api_rejects_unknown_qualification(self):
        for mode in flask_app.MODES:
            response = self.post(qualification='B.Voc in Retail', skills='Python', mode=mode)
            self.assertEqual(response.status_code, 400, mode)
            self.assertTrue(response.content_type.startswith('application/json'))
            self.assertIn('Unsupported qualification', response.get_json()['error'])

    def test_api_rejects_empty_qualification(self):
        response = self.post(qualification='', skills='Python')
        self.assertEqual(response.status_code, 400)
        self.assertIn('qualification is required', response.get_json()['error'])


if __name__ == '__main__':
    unittest.main()
