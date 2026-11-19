"""Tests for dropping experienced roles from an internship-focused search.

Live probing showed the deployment carries no structured experience field, so the
rules here run on titles and description text; the documented
``job_requirements=no_experience`` parameter is asserted separately on the request.
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
import experience_filter  # noqa: E402
import jobs_api  # noqa: E402
import _auth_support  # noqa: E402

RAW_JOBS = [
    {'job_id': 'INTERN', 'job_title': 'Software Development Internship',
     'employer_name': 'Acme', 'job_description': 'Learn on the job. No experience needed.',
     'job_apply_link': 'https://example.com/1', 'job_city': 'Bengaluru', 'job_country': 'IN'},
    {'job_id': 'FRESHER', 'job_title': 'Junior Software Developer (Fresher)',
     'employer_name': 'Beta', 'job_description': '0-1 years of experience welcome.',
     'job_apply_link': 'https://example.com/2', 'job_city': 'Bengaluru', 'job_country': 'IN'},
    {'job_id': 'SENIOR', 'job_title': 'Senior Software Engineer',
     'employer_name': 'Gamma', 'job_description': 'Design distributed systems.',
     'job_apply_link': 'https://example.com/3', 'job_city': 'Bengaluru', 'job_country': 'IN'},
    {'job_id': 'YEARS', 'job_title': 'Java Developer',
     'employer_name': 'Delta',
     'job_description': 'You need 5+ years of experience building services.',
     'job_apply_link': 'https://example.com/4', 'job_city': 'Bengaluru', 'job_country': 'IN'},
    {'job_id': 'TITLE_YEARS', 'job_title': 'Software Engineer- Java |8 to 14yrs| Bangalore',
     'employer_name': 'Epsilon', 'job_description': 'Backend work.',
     'job_apply_link': 'https://example.com/5', 'job_city': 'Bengaluru', 'job_country': 'IN'},
]


def _job(title, description=''):
    return {'id': title, 'title': title, 'description': description}


class TitleRuleTests(unittest.TestCase):

    def test_seniority_titles_are_dropped(self):
        title = ['Senior Software Engineer', 'Sr. Developer', 'Lead Backend Engineer',
                 'Principal Engineer', 'Staff Engineer', 'Software Architect',
                 'Engineering Manager', 'Software Engineer II', 'SDE 2', 'VP Engineering']
        for role in title:
            signal = experience_filter.experience_signal(_job(role))
            self.assertTrue(signal['experienced'], role)
            self.assertIn('seniority', signal['reason'])

    def test_intern_and_fresher_titles_are_kept(self):
        for role in ['Software Development Internship', 'Data Science Intern',
                     'Graduate Trainee', 'Junior Developer (Fresher)']:
            self.assertFalse(experience_filter.experience_signal(_job(role))['experienced'],
                             role)

    def test_years_in_the_title_are_read(self):
        self.assertEqual(experience_filter.years_mentioned(_job('Java |8 to 14yrs| Bangalore')), 8)
        self.assertEqual(experience_filter.years_mentioned(_job('Developer 3+ years')), 3)

    def test_no_experience_and_entry_level_titles_override_body_mentions(self):
        # Live case: "Software Programmer (No experience required)" with a body that
        # mentioned 1+ years was being dropped before this rule.
        for role in ['Software Programmer (No experience required) (Bengaluru)',
                     'Entry Level Developer', 'Developer 0-1 years']:
            job = _job(role, 'You need 1+ years of experience.')
            self.assertFalse(experience_filter.experience_signal(job)['experienced'], role)

    def test_intern_title_overrides_body_mentions(self):
        job = _job('Software Engineer Intern', 'Our team has 10 years of experience.')
        self.assertFalse(experience_filter.experience_signal(job)['experienced'])


class DescriptionRuleTests(unittest.TestCase):

    def test_common_year_phrasings_are_detected(self):
        cases = [
            ('You need 5+ years of experience.', 5),
            ('Experience of 4 years in backend.', 4),
            ('Minimum 3 years working with Java.', 3),
            ('At least 2 years in a similar role.', 2),
        ]
        for text, expected in cases:
            self.assertEqual(experience_filter.years_mentioned(_job('Developer', text)),
                             expected, text)

    def test_unrelated_year_mentions_are_ignored(self):
        text = ('The company was founded 12 years ago and serves 5 years of warranty. '
                'Our office moved 3 years back.')
        self.assertIsNone(experience_filter.years_mentioned(_job('Developer', text)))

    def test_mentions_at_the_threshold_are_kept(self):
        job = _job('Developer', 'Minimum 2 years of experience.')
        self.assertFalse(experience_filter.experience_signal(job, max_years=2)['experienced'])
        self.assertTrue(experience_filter.experience_signal(job, max_years=1)['experienced'])

    def test_default_threshold_is_zero(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('JOBS_MAX_EXPERIENCE_YEARS', None)
            self.assertEqual(experience_filter.max_experience_years(), 0)

    def test_threshold_from_environment(self):
        with mock.patch.dict(os.environ, {'JOBS_MAX_EXPERIENCE_YEARS': '2'}):
            self.assertEqual(experience_filter.max_experience_years(), 2)

    def test_garbage_threshold_falls_back_to_zero(self):
        with mock.patch.dict(os.environ, {'JOBS_MAX_EXPERIENCE_YEARS': 'lots'}):
            self.assertEqual(experience_filter.max_experience_years(), 0)


class FilterTests(unittest.TestCase):

    def test_split_reports_reasons(self):
        jobs = [jobs_api.normalize_job(raw) for raw in RAW_JOBS]
        kept, dropped = experience_filter.filter_internships(jobs)
        self.assertEqual([job['id'] for job in kept], ['INTERN', 'FRESHER'])
        self.assertEqual({entry['title'] for entry in dropped},
                         {'Senior Software Engineer', 'Java Developer',
                          'Software Engineer- Java |8 to 14yrs| Bangalore'})
        for entry in dropped:
            self.assertTrue(entry['reason'])

    def test_no_jobs_dropped_when_threshold_is_high(self):
        jobs = [jobs_api.normalize_job(raw) for raw in RAW_JOBS]
        kept, dropped = experience_filter.filter_internships(jobs, max_years=20)
        # Seniority titles still go; only the years-based rule loosens.
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]['title'], 'Senior Software Engineer')


class ApiWiringTests(unittest.TestCase):

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
        body = json.dumps({'status': 'OK',
                           'data': {'jobs': RAW_JOBS, 'cursor': ''}}).encode()

        def _fake(url, headers, timeout):
            calls.append(url)
            return 200, body

        return _fake, calls

    def test_entry_level_is_on_by_default_and_sends_the_documented_parameter(self):
        stub, calls = self._stub()
        with mock.patch.object(jobs_api, '_http_get', stub):
            payload = self.client.get('/api/jobs?q=developer').get_json()
        self.assertIn('job_requirements=no_experience', calls[0])
        self.assertTrue(payload['entry_level'])
        self.assertEqual(payload['filtered_out'], 3)
        titles = [job['title'] for job in payload['jobs']]
        self.assertNotIn('Senior Software Engineer', titles)

    def test_entry_level_can_be_switched_off(self):
        stub, calls = self._stub()
        with mock.patch.object(jobs_api, '_http_get', stub):
            payload = self.client.get('/api/jobs?q=developer&entry_level=0').get_json()
        self.assertNotIn('job_requirements', calls[0])
        self.assertFalse(payload['entry_level'])
        self.assertEqual(payload['filtered_out'], 0)
        self.assertEqual(payload['count'], 5)

    def test_filtered_samples_name_the_reason(self):
        stub, _ = self._stub()
        with mock.patch.object(jobs_api, '_http_get', stub):
            payload = self.client.get('/api/jobs?q=developer').get_json()
        self.assertTrue(payload['filtered_samples'])
        self.assertIn('reason', payload['filtered_samples'][0])

    def test_count_reflects_the_filtered_list(self):
        stub, _ = self._stub()
        with mock.patch.object(jobs_api, '_http_get', stub):
            payload = self.client.get('/api/jobs?q=developer').get_json()
        self.assertEqual(payload['count'], len(payload['jobs']))

    def test_page_renders_the_checkbox_checked(self):
        body = self.client.get('/jobs').get_data(as_text=True)
        self.assertIn('id="entry_level"', body)
        self.assertIn('checked', body)

    def test_page_checkbox_unchecked_when_disabled(self):
        body = self.client.get('/jobs?entry_level=0').get_data(as_text=True)
        self.assertIn('id="entry_level"', body)
        self.assertNotIn('name="entry_level" value="1"\n                               checked', body)


class UpstreamParameterTests(unittest.TestCase):

    def test_documented_values_are_enforced(self):
        with self.assertRaises(jobs_api.JobsApiError) as ctx:
            jobs_api.validate_search(keyword='x', job_requirements=['five_years'])
        self.assertEqual(ctx.exception.status, 400)
        self.assertIn('no_experience', ctx.exception.message)

    def test_values_are_sent_comma_separated(self):
        plan = jobs_api.validate_search(keyword='x',
                                        job_requirements='no_experience,under_3_years_experience')
        params = jobs_api._upstream_params(plan, 'search-v2')
        self.assertEqual(params['job_requirements'],
                         'no_experience,under_3_years_experience')

    def test_requirements_are_part_of_the_cache_key(self):
        jobs_api.clear_cache()
        with mock.patch.dict(os.environ, {'RAPIDAPI_KEY': 'k', 'JOBS_CACHE_TTL': '600'}):
            calls = []

            def _fake(url, headers, timeout):
                calls.append(url)
                return 200, json.dumps({'status': 'OK',
                                        'data': {'jobs': RAW_JOBS, 'cursor': ''}}).encode()

            with mock.patch.object(jobs_api, '_http_get', _fake):
                jobs_api.search_jobs(keyword='x')
                jobs_api.search_jobs(keyword='x', job_requirements=('no_experience',))
            self.assertEqual(len(calls), 2)
        jobs_api.clear_cache()


if __name__ == '__main__':
    unittest.main()
