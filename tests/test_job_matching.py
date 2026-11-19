"""Tests for profile matching against live JSearch listings.

Covers the pure scorer (``job_matching``) and its wiring into ``/api/jobs``,
with the upstream HTTP mocked so no key or network is needed.
"""
import json
import os
import sys
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, TESTS_DIR)
os.chdir(REPO_ROOT)

import app as flask_app  # noqa: E402
import job_matching  # noqa: E402
import jobs_api  # noqa: E402
import _auth_support  # noqa: E402

RAW_JOBS = [
    {
        'job_id': 'A',
        'job_title': 'Python Developer',
        'employer_name': 'Acme',
        'job_description': 'Build REST APIs in Python and SQL. Experience with Django required.',
        'job_apply_link': 'https://example.com/a',
        'job_is_remote': False,
        'job_city': 'Bengaluru',
        'job_country': 'IN',
    },
    {
        'job_id': 'B',
        'job_title': 'Frontend Engineer',
        'employer_name': 'Beta',
        'job_description': 'React and TypeScript for a design system. No backend work.',
        'job_apply_link': 'https://example.com/b',
        'job_is_remote': True,
        'job_city': 'Pune',
        'job_country': 'IN',
    },
    {
        'job_id': 'C',
        'job_title': 'Warehouse Associate',
        'employer_name': 'Gamma',
        'job_description': 'Pick and pack orders. Forklift licence preferred.',
        'job_apply_link': 'https://example.com/c',
        'job_is_remote': False,
        'job_city': 'Delhi',
        'job_country': 'IN',
    },
]

# The scorer consumes the normalised shape the API returns, so build the fixtures
# through the same function that produces it.
JOBS = [jobs_api.normalize_job(raw) for raw in RAW_JOBS]


def _payload(jobs=None):
    return json.dumps({'status': 'OK', 'data': {'jobs': jobs or RAW_JOBS,
                                                 'cursor': ''}}).encode()


class ParseSkillsTests(unittest.TestCase):

    def test_splits_normalises_and_dedupes(self):
        self.assertEqual(job_matching.parse_skills('Python,  python , SQL'),
                         ['python', 'sql'])

    def test_applies_the_existing_synonym_table(self):
        self.assertEqual(job_matching.parse_skills('MySQL, ReactJS, MS Excel'),
                         ['sql', 'react', 'excel'])

    def test_caps_the_number_of_skills(self):
        self.assertEqual(len(job_matching.parse_skills(','.join(f'skill{i}' for i in range(20)))),
                         job_matching.MAX_SKILLS)

    def test_blank_input_yields_nothing(self):
        self.assertEqual(job_matching.parse_skills(''), [])
        self.assertEqual(job_matching.parse_skills(None), [])


class BuildProfileTests(unittest.TestCase):

    def test_active_when_skills_present(self):
        self.assertTrue(job_matching.build_profile(skills='python')['active'])

    def test_active_when_only_sector_present(self):
        self.assertTrue(job_matching.build_profile(sector='FinTech')['active'])

    def test_inactive_when_empty(self):
        profile = job_matching.build_profile()
        self.assertFalse(profile['active'])
        self.assertEqual(profile['skills'], [])

    def test_qualification_and_location_do_not_activate(self):
        # JSearch exposes no education requirement, and location is a search filter.
        self.assertFalse(job_matching.build_profile(
            qualification='B.Tech', location='Bangalore')['active'])


class ScoreJobTests(unittest.TestCase):

    def setUp(self):
        self.profile = job_matching.build_profile(skills='python, sql, react')

    def test_skill_coverage_is_fraction_of_profile_skills_found(self):
        job = job_matching.score_job({'title': 'Python Developer',
                                      'description': 'python and sql'}, self.profile)
        self.assertEqual(job['match']['skill_coverage'], round(2 / 3, 3))
        self.assertEqual(job['match']['matched_skills'], ['python', 'sql'])
        self.assertEqual(job['match']['missing_skills'], ['react'])

    def test_score_top_of_its_range_for_a_perfect_text_match(self):
        job = job_matching.score_job({'title': 'python sql react', 'description': ''},
                                     self.profile, similarity=1.0)
        expected = job_matching.WEIGHT_SKILLS + job_matching.WEIGHT_TEXT
        self.assertAlmostEqual(job['match']['score'], round(expected, 3), places=3)
        self.assertEqual(job['match']['percent'], int(round(expected * 100)))

    def test_word_boundaries_prevent_false_positives(self):
        job = job_matching.score_job({'title': 'JavaScript Developer',
                                      'description': 'javascript everywhere'},
                                     job_matching.build_profile(skills='java'))
        self.assertEqual(job['match']['matched_skills'], [])

    def test_synonyms_match_the_posting_wording(self):
        job = job_matching.score_job({'title': 'Backend Engineer',
                                      'description': 'You will use MySQL daily'},
                                     job_matching.build_profile(skills='sql'))
        self.assertEqual(job['match']['matched_skills'], ['sql'])

    def test_sector_hit_is_reported_and_scored(self):
        job = job_matching.score_job({'title': 'Analyst', 'description': 'fintech payments'},
                                     job_matching.build_profile(skills='sql', sector='FinTech'))
        self.assertTrue(job['match']['sector_match'])
        self.assertIn('fintech', job['match']['explanation'].lower())

    def test_explanation_names_the_numbers_used(self):
        job = job_matching.score_job(JOBS[0], self.profile, similarity=0.42)
        explanation = job['match']['explanation']
        self.assertIn('Matches 2 of 3 skills', explanation)
        self.assertIn('0.42', explanation)

    def test_inactive_profile_attaches_no_match(self):
        job = job_matching.score_job(JOBS[0], job_matching.build_profile())
        self.assertIsNone(job['match'])

    def test_input_job_is_not_mutated(self):
        original = dict(JOBS[0])
        job_matching.score_job(JOBS[0], self.profile)
        self.assertEqual(JOBS[0], original)

    def test_missing_fields_do_not_crash(self):
        job = job_matching.score_job({}, self.profile)
        self.assertEqual(job['match']['matched_skills'], [])
        self.assertEqual(job['match']['percent'], 0)


class RankJobsTests(unittest.TestCase):

    def setUp(self):
        self.profile = job_matching.build_profile(skills='python, sql')

    def test_best_match_first(self):
        ranked = job_matching.rank_jobs(JOBS, self.profile)
        self.assertEqual(ranked[0]['id'], 'A')
        scores = [job['match']['score'] for job in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_order_is_preserved_without_a_profile(self):
        ranked = job_matching.rank_jobs(JOBS, job_matching.build_profile())
        self.assertEqual([job['id'] for job in ranked], ['A', 'B', 'C'])
        self.assertTrue(all(job['match'] is None for job in ranked))

    def test_irrelevant_job_scores_lowest(self):
        ranked = job_matching.rank_jobs(JOBS, self.profile)
        self.assertEqual(ranked[-1]['id'], 'C')

    def test_similarity_is_bounded(self):
        for job in job_matching.rank_jobs(JOBS, self.profile):
            self.assertGreaterEqual(job['match']['text_similarity'], 0.0)
            self.assertLessEqual(job['match']['text_similarity'], 1.0)


class ApiProfileTests(unittest.TestCase):

    def setUp(self):
        jobs_api.clear_cache()
        self.env = mock.patch.dict(os.environ, {'RAPIDAPI_KEY': 'test-key', 'JOBS_CACHE_TTL': '600'})
        self.env.start()
        self.client = _auth_support.logged_in_client()

    def tearDown(self):
        self.env.stop()
        jobs_api.clear_cache()

    def _stub(self, recorded=None):
        calls = recorded if recorded is not None else []

        def _fake(url, headers, timeout):
            calls.append(url)
            return 200, _payload()

        return _fake, calls

    def test_profile_is_echoed_and_jobs_are_scored(self):
        stub, _ = self._stub()
        with mock.patch.object(jobs_api, '_http_get', stub):
            response = self.client.get('/api/jobs?q=developer&skills=python,sql')
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload['profile']['active'])
        self.assertEqual(payload['profile']['skills'], ['python', 'sql'])
        self.assertTrue(all(job['match'] for job in payload['jobs']))
        percents = [job['match']['percent'] for job in payload['jobs']]
        self.assertEqual(percents, sorted(percents, reverse=True))

    def test_without_a_profile_match_is_null(self):
        stub, _ = self._stub()
        with mock.patch.object(jobs_api, '_http_get', stub):
            payload = self.client.get('/api/jobs?q=developer').get_json()
        self.assertFalse(payload['profile']['active'])
        self.assertTrue(all(job['match'] is None for job in payload['jobs']))

    def test_profile_skills_replace_a_blank_keyword(self):
        stub, calls = self._stub()
        with mock.patch.object(jobs_api, '_http_get', stub):
            response = self.client.get('/api/jobs?skills=python,sql,react')
        self.assertEqual(response.status_code, 200)
        # Two skills: a three-skill query over-constrains JSearch (1 vs 10 live results).
        self.assertIn('query=python+sql', calls[0])
        self.assertNotIn('react', calls[0])

    def test_keyword_still_wins_over_profile_skills(self):
        stub, calls = self._stub()
        with mock.patch.object(jobs_api, '_http_get', stub):
            self.client.get('/api/jobs?q=data+analyst&skills=python')
        self.assertIn('query=data+analyst', calls[0])

    def test_profile_does_not_leak_into_other_users_results(self):
        stub, _ = self._stub()
        with mock.patch.object(jobs_api, '_http_get', stub):
            self.client.get('/api/jobs?q=developer&skills=python')
            second = self.client.get('/api/jobs?q=developer').get_json()
        # Second response is served from cache but must be scored with its own profile.
        self.assertFalse(second['profile']['active'])
        self.assertTrue(all(job['match'] is None for job in second['jobs']))

    def test_page_renders_the_profile_strip(self):
        response = self.client.get('/jobs?skills=python,sql&qualification=B.Tech')
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn('Matching your profile', body)
        self.assertIn('python, sql', body)
        self.assertIn('value="python sql"', body)   # pre-filled keyword

    def test_page_without_profile_has_no_strip(self):
        body = self.client.get('/jobs').get_data(as_text=True)
        self.assertNotIn('Matching your profile', body)


if __name__ == '__main__':
    unittest.main()
