"""Regression tests for the /recommend endpoint using the Flask test client.

Run from the repository root:

    python -m unittest discover -s tests -v

The module fixes its own working directory and sys.path, because app.py reads
'data/internship.csv' by relative path at import time.
"""
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import app as flask_app  # noqa: E402


def _level(education_text):
    return flask_app.parse_education_level(education_text)


def _ids(rows):
    return [row['internship_id'] for row in rows]


def _scores(rows):
    return [row['score'] for row in rows]


class RecommendApiTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = flask_app.app.test_client()

    def post(self, **payload):
        return self.client.post('/recommend', json=payload)

    def results(self, **payload):
        response = self.post(**payload)
        self.assertEqual(response.status_code, 200, response.data)
        return response.get_json()

    def assert_json_error(self, response):
        self.assertEqual(response.status_code, 400, response.status_code)
        self.assertTrue(response.content_type.startswith('application/json'), response.content_type)
        body = response.get_json()
        self.assertIsInstance(body, dict)
        self.assertTrue(body.get('error'), body)

    # --- 1-4: the four profiles from the demo dataset ----------------------

    def test_10th_pass_with_basic_computer_returns_results(self):
        rows = self.results(qualification='10th Pass', skills='Basic Computer')
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertLessEqual(_level(row['required_education']), 0, row['required_education'])
            self.assertIn('basic computer', row['required_skills'].lower())

    def test_12th_pass_with_excel_returns_results(self):
        rows = self.results(qualification='12th Pass', skills='Excel')
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertLessEqual(_level(row['required_education']), 1, row['required_education'])
            self.assertIn('excel', row['required_skills'].lower())

    def test_diploma_with_troubleshooting_returns_results(self):
        rows = self.results(qualification='Diploma', skills='Troubleshooting')
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertLessEqual(_level(row['required_education']), 2, row['required_education'])
            self.assertIn('troubleshooting', row['required_skills'].lower())

    def test_btech_with_python_and_sql_returns_it_rows(self):
        rows = self.results(qualification='B.Tech', skills='Python, SQL')
        self.assertGreater(len(rows), 0)
        self.assertIn('INT-IT-25-001', _ids(rows))
        top = rows[0]
        self.assertEqual(top['title'], 'Software Development Intern')

    # --- 5-6: skill normalisation ----------------------------------------

    def test_skill_matching_is_case_insensitive(self):
        mixed = self.results(qualification='10th Pass', skills='bASIC COMPUTER')
        plain = self.results(qualification='10th Pass', skills='Basic Computer')
        self.assertTrue(mixed)
        self.assertEqual(_ids(mixed), _ids(plain))
        self.assertEqual(_scores(mixed), _scores(plain))

    def test_skill_matching_ignores_extra_whitespace(self):
        messy = self.results(qualification='12th Pass', skills='  excel  ,  data   entry ')
        clean = self.results(qualification='12th Pass', skills='Excel,Data Entry')
        self.assertTrue(messy)
        self.assertEqual(_ids(messy), _ids(clean))
        self.assertEqual(_scores(messy), _scores(clean))

    # --- 7: unknown qualification ----------------------------------------

    def test_unknown_qualification_is_rejected(self):
        response = self.post(qualification='B.Voc in Retail', skills='Python')
        self.assertEqual(response.status_code, 400)
        self.assertTrue(response.content_type.startswith('application/json'))
        self.assertIn('Unsupported qualification', response.get_json()['error'])

    def test_empty_qualification_is_rejected_not_granted(self):
        response = self.post(qualification='', skills='Python')
        self.assertEqual(response.status_code, 400)
        self.assertTrue(response.content_type.startswith('application/json'))
        self.assertIn('qualification is required', response.get_json()['error'])

    def test_freeform_qualification_is_still_accepted(self):
        # Freeform text containing a known degree keyword is not rejected.
        rows = self.results(qualification='Pursuing B.Tech in Computer Science', skills='Python')
        self.assertTrue(rows)

    # --- 8: education eligibility ----------------------------------------

    def test_diploma_is_not_eligible_for_btech_rows(self):
        diploma = _ids(self.results(qualification='Diploma', skills='Python, SQL'))
        btech = _ids(self.results(qualification='B.Tech', skills='Python, SQL'))
        self.assertIn('INT-IT-25-001', btech)
        self.assertNotIn('INT-IT-25-001', diploma)
        for row in self.results(qualification='Diploma', skills='Python, SQL'):
            self.assertLessEqual(_level(row['required_education']), 2)

    def test_btech_is_eligible_for_lower_level_rows(self):
        # A B.Tech graduate satisfies a 10th Pass / 12th Pass / Diploma requirement.
        rows = self.results(qualification='B.Tech', skills='Troubleshooting')
        self.assertTrue(any(_level(row['required_education']) == 2 for row in rows))

    def test_higher_level_qualification_unlocks_higher_level_rows(self):
        # 'pandas' exists only on a level-4 row, so only a master's-level candidate sees it.
        self.assertEqual(self.results(qualification='B.Tech', skills='pandas'), [])
        mba = _ids(self.results(qualification='MBA', skills='pandas'))
        self.assertIn('INT-DS-25-004', mba)

    # --- 9: ranking -------------------------------------------------------

    def test_results_are_ordered_by_descending_score(self):
        rows = self.results(qualification='B.Tech', skills='Python, SQL')
        self.assertGreater(len(rows), 1)
        scores = _scores(rows)
        self.assertEqual(scores, sorted(scores, reverse=True), scores)

    def test_ranked_by_score_not_by_qualification(self):
        # Same eligible row scores identically for two qualifications; the level only
        # decides which rows are eligible, not how they rank.
        btech = {r['internship_id']: r['score'] for r in self.results(
            qualification='B.Tech', skills='Python, SQL')}
        mba = {r['internship_id']: r['score'] for r in self.results(
            qualification='MBA', skills='Python, SQL')}
        shared = set(btech) & set(mba)
        self.assertTrue(shared, 'expected at least one internship eligible for both')
        for internship_id in shared:
            self.assertEqual(btech[internship_id], mba[internship_id], internship_id)

    # --- 10: duplicates and caps -----------------------------------------

    def test_no_duplicate_recommendations(self):
        for payload in ({'qualification': 'B.Tech', 'skills': 'Python, SQL'},
                        {'qualification': '10th Pass', 'skills': 'Excel, Data Entry'},
                        {'qualification': 'Diploma', 'skills': 'Troubleshooting, Basic Computer'}):
            ids = _ids(self.results(**payload))
            self.assertEqual(len(ids), len(set(ids)), payload)

    def test_never_returns_more_than_five(self):
        rows = self.results(qualification='MBA', skills='Excel, Python, SQL, Communication')
        self.assertLessEqual(len(rows), 5)

    def test_skills_the_dataset_does_not_contain_return_nothing(self):
        self.assertEqual(self.results(qualification='B.Tech', skills='Underwater Basket Weaving'), [])

    # --- location mapping -------------------------------------------------

    def test_karnataka_equals_bangalore_ids_and_scores(self):
        base = {'qualification': 'B.Tech', 'skills': 'Python, Java, SQL',
                'sector_interested': 'Information Technology'}
        state = self.results(**dict(base, location_interested='Karnataka'))
        city = self.results(**dict(base, location_interested='Bangalore'))
        self.assertTrue(state)
        self.assertEqual(_ids(state), _ids(city))
        self.assertEqual(_scores(state), _scores(city))

    # --- qualification text parsing --------------------------------------

    def test_freeform_qualification_equals_plain(self):
        for plain, freeform in (('B.Tech', 'Pursuing B.Tech in CS'),
                                ('MBA', 'Pursuing MBA in Finance'),
                                ('B.Sc', 'Pursuing B.Sc in Statistics')):
            plain_rows = self.results(qualification=plain, skills='Python, Java, SQL')
            free_rows = self.results(qualification=freeform, skills='Python, Java, SQL')
            self.assertEqual(_ids(plain_rows), _ids(free_rows), (plain, freeform))

    # --- error contract ---------------------------------------------------

    def test_error_missing_body(self):
        self.assert_json_error(self.client.post('/recommend'))

    def test_error_malformed_json(self):
        self.assert_json_error(self.client.post('/recommend', data='notjson',
                                                content_type='application/json'))

    def test_error_non_object_body(self):
        self.assert_json_error(self.post_json_list())

    def post_json_list(self):
        return self.client.post('/recommend', json=[{'skills': 'Python'}])

    def test_error_blank_profile(self):
        self.assert_json_error(self.client.post('/recommend', json={}))
        self.assert_json_error(self.client.post('/recommend', json={
            'qualification': '', 'skills': '', 'sector_interested': '', 'location_interested': ''}))
        self.assert_json_error(self.client.post('/recommend', json={
            'qualification': '  ', 'skills': '   ', 'sector_interested': ' ',
            'location_interested': '  '}))

    def test_error_null_fields(self):
        self.assert_json_error(self.client.post('/recommend', json={
            'qualification': None, 'skills': None, 'sector_interested': None,
            'location_interested': None}))

    # --- success shape ----------------------------------------------------

    def test_valid_payload_returns_list(self):
        rows = self.results(qualification='B.Tech', skills='Python')
        self.assertIsInstance(rows, list)
        self.assertTrue(rows)
        self.assertTrue({'internship_id', 'title', 'score'} <= set(rows[0]))


if __name__ == '__main__':
    unittest.main()
