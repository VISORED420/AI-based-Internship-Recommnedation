"""Tests for the JSearch integration: validation, request shape, caching, routes.

Upstream HTTP is always mocked (``jobs_api._http_get``), so the suite needs no
network and no real RapidAPI key. Network behaviour that cannot be proven
without a live key is reported as unverified in the README, not faked here.
"""
import json
import os
import sys
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, TESTS_DIR)   # so _auth_support resolves under any invocation
os.chdir(REPO_ROOT)

import app as flask_app  # noqa: E402
import jobs_api  # noqa: E402
import _auth_support  # noqa: E402  (registers a user; sets S2I_DB_PATH)


SAMPLE_JOBS = [
    {
        'job_id': 'JOB-1',
        'job_title': 'Python Developer',
        'employer_name': 'Nexus Labs',
        'employer_logo': 'https://example.com/logo.png',
        'job_publisher': 'LinkedIn',
        'job_employment_type': 'Full-time',
        'job_employment_types': ['FULLTIME'],
        'job_apply_link': 'https://example.com/apply/1',
        'job_apply_is_direct': True,
        'job_description': 'Build APIs with Python and PostgreSQL. ' * 20,
        'job_is_remote': False,
        'job_posted_at_datetime_utc': '2026-05-19T00:00:00.000Z',
        'job_posted_at': '2 days ago',
        'job_city': 'Bangalore',
        'job_state': 'Karnataka',
        'job_country': 'IN',
        'job_min_salary': 20000,
        'job_max_salary': 35000,
        'job_salary_currency': 'INR',
        'job_salary_period': 'MONTH',
        'job_required_skills': ['Python', 'SQL'],
        'job_google_link': 'https://google.com/jobs/1',
    },
    {
        'job_id': 'JOB-2',
        'job_title': 'Backend Engineer (Remote)',
        'employer_name': 'CloudWorks',
        'job_publisher': 'Indeed',
        'job_employment_type': 'Contractor',
        'job_employment_types': ['CONTRACTOR'],
        'job_apply_link': '',
        'job_google_link': 'https://google.com/jobs/2',
        'job_description': 'Remote contract role.',
        'job_is_remote': True,
        'job_posted_at_datetime_utc': '2026-05-20T06:00:00.000Z',
        'job_city': '',
        'job_state': '',
        'job_country': 'IN',
    },
]


def _payload(jobs=None, cursor=''):
    return {'status': 'OK', 'request_id': 'test', 'data': jobs if jobs is not None else SAMPLE_JOBS}


class JobsTestBase(unittest.TestCase):

    def setUp(self):
        jobs_api.clear_cache()
        env = {'RAPIDAPI_KEY': 'test-key-not-real', 'JOBS_CACHE_TTL': '600'}
        self.env_patch = mock.patch.dict(os.environ, env, clear=False)
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        jobs_api.clear_cache()

    def fake_http(self, jobs=None, status=200, body=None, calls=None):
        """Return a stub for _http_get that records the URLs it was given."""
        recorded = calls if calls is not None else []

        def _stub(url, headers, timeout):
            recorded.append({'url': url, 'headers': headers, 'timeout': timeout})
            payload = body if body is not None else json.dumps(_payload(jobs)).encode()
            return status, payload

        return _stub, recorded


class ValidationTests(JobsTestBase):

    def test_requires_keyword_or_location(self):
        with self.assertRaises(jobs_api.JobsApiError) as ctx:
            jobs_api.validate_search()
        self.assertEqual(ctx.exception.status, 400)
        self.assertIn('job title, skill or location', ctx.exception.message)

    def test_keyword_and_location_build_one_query(self):
        plan = jobs_api.validate_search(keyword='python developer', location='bangalore')
        self.assertEqual(plan['query'], 'python developer in bangalore')

    def test_keyword_only_query(self):
        self.assertEqual(jobs_api.validate_search(keyword='data analyst')['query'], 'data analyst')

    def test_location_only_query(self):
        self.assertEqual(jobs_api.validate_search(location='mumbai')['query'], 'jobs in mumbai')

    def test_unknown_date_posted_lists_accepted_values(self):
        with self.assertRaises(jobs_api.JobsApiError) as ctx:
            jobs_api.validate_search(keyword='x', date_posted='yesterday')
        self.assertEqual(ctx.exception.status, 400)
        for value in jobs_api.DATE_POSTED_VALUES:
            self.assertIn(value, ctx.exception.message)

    def test_unknown_employment_type_is_rejected(self):
        with self.assertRaises(jobs_api.JobsApiError) as ctx:
            jobs_api.validate_search(keyword='x', employment_types=['GIG'])
        self.assertIn('FULLTIME', ctx.exception.message)

    def test_employment_types_are_upper_cased_and_deduplicated(self):
        plan = jobs_api.validate_search(keyword='x',
                                        employment_types='intern, INTERN,parttime')
        self.assertEqual(plan['employment_types'], ['INTERN', 'PARTTIME'])

    def test_unknown_work_mode_is_rejected(self):
        with self.assertRaises(jobs_api.JobsApiError) as ctx:
            jobs_api.validate_search(keyword='x', work_mode='hybrid')
        self.assertIn('remote', ctx.exception.message)

    def test_page_bounds(self):
        for bad in (0, -1, 21, 'abc'):
            with self.assertRaises(jobs_api.JobsApiError, msg=bad):
                jobs_api.validate_search(keyword='x', page=bad)

    def test_num_pages_bounds(self):
        with self.assertRaises(jobs_api.JobsApiError):
            jobs_api.validate_search(keyword='x', num_pages=21)

    def test_page_plus_num_pages_window(self):
        with self.assertRaises(jobs_api.JobsApiError):
            jobs_api.validate_search(keyword='x', page=20, num_pages=2)

    def test_country_must_be_two_letters(self):
        for bad in ('usa', 'india', '1n'):
            with self.assertRaises(jobs_api.JobsApiError, msg=bad):
                jobs_api.validate_search(keyword='x', country=bad)

    def test_country_defaults_from_env(self):
        with mock.patch.dict(os.environ, {'JSEARCH_DEFAULT_COUNTRY': 'gb'}):
            self.assertEqual(jobs_api.validate_search(keyword='x')['country'], 'gb')

    def test_query_is_length_capped(self):
        plan = jobs_api.validate_search(keyword='a' * 400)
        self.assertLessEqual(len(plan['keyword']), jobs_api.MAX_KEYWORD_LENGTH)


class NormalisationTests(JobsTestBase):

    def test_full_record_maps_to_card_fields(self):
        job = jobs_api.normalize_job(SAMPLE_JOBS[0])
        self.assertEqual(job['title'], 'Python Developer')
        self.assertEqual(job['company'], 'Nexus Labs')
        self.assertEqual(job['location'], 'Bangalore, Karnataka, IN')
        self.assertEqual(job['employment_type'], 'Full-time')
        self.assertEqual(job['apply_link'], 'https://example.com/apply/1')
        self.assertFalse(job['is_remote'])
        self.assertEqual(job['required_skills'], ['Python', 'SQL'])
        self.assertIn('per month', job['salary'])

    def test_salary_uses_min_and_max(self):
        job = jobs_api.normalize_job(SAMPLE_JOBS[0])
        self.assertIn('20,000', job['salary'])
        self.assertIn('35,000', job['salary'])

    def test_missing_fields_degrade_gracefully(self):
        job = jobs_api.normalize_job({'job_id': 'X'})
        self.assertEqual(job['title'], 'Untitled job')
        self.assertEqual(job['company'], 'Employer not listed')
        self.assertEqual(job['location'], '')
        self.assertEqual(job['salary'], '')
        self.assertEqual(job['apply_link'], '')
        self.assertEqual(job['required_skills'], [])

    def test_apply_link_falls_back_to_google_link(self):
        job = jobs_api.normalize_job(SAMPLE_JOBS[1])
        self.assertEqual(job['apply_link'], 'https://google.com/jobs/2')

    def test_location_does_not_repeat_identical_parts(self):
        job = jobs_api.normalize_job({'job_city': 'Delhi', 'job_state': 'Delhi',
                                      'job_country': 'IN'})
        self.assertEqual(job['location'], 'Delhi, IN')

    def test_non_dict_record_is_skipped(self):
        self.assertIsNone(jobs_api.normalize_job('not a job'))

    def test_salary_single_value(self):
        job = jobs_api.normalize_job({'job_min_salary': 50000, 'job_max_salary': None,
                                      'job_salary_currency': 'USD'})
        self.assertIn('50,000', job['salary'])
        self.assertNotIn('-', job['salary'])


class RequestTests(JobsTestBase):

    def test_documented_parameters_and_headers_are_sent(self):
        stub, calls = self.fake_http()
        with mock.patch.object(jobs_api, '_http_get', stub):
            jobs_api.search_jobs(keyword='python', location='bangalore',
                                 date_posted='week', employment_types=['FULLTIME'],
                                 work_mode='remote', page=2)
        url = calls[0]['url']
        headers = calls[0]['headers']
        self.assertTrue(url.startswith(jobs_api.SEARCH_URL + '?'))
        self.assertIn('/search-v2?', url)
        self.assertIn('query=python+in+bangalore', url)
        self.assertIn('num_pages=1', url)
        self.assertIn('date_posted=week', url)
        self.assertIn('employment_types=FULLTIME', url)
        self.assertIn('work_from_home=true', url)
        self.assertIn('location=bangalore', url)   # documented UULE parameter
        self.assertEqual(headers['X-RapidAPI-Host'], 'jsearch.p.rapidapi.com')
        self.assertEqual(headers['X-RapidAPI-Key'], 'test-key-not-real')

    def test_location_is_omitted_when_the_user_did_not_supply_one(self):
        stub, calls = self.fake_http()
        with mock.patch.object(jobs_api, '_http_get', stub):
            jobs_api.search_jobs(keyword='python')
        self.assertNotIn('location=', calls[0]['url'])

    def test_legacy_endpoint_does_not_receive_the_v2_only_location_param(self):
        responses = [(404, b'{"message":"Endpoint \'/search-v2\' does not exist"}'),
                     (200, json.dumps(_payload()).encode())]
        recorded = []

        def _stub(url, headers, timeout):
            recorded.append(url)
            return responses[len(recorded) - 1]

        with mock.patch.object(jobs_api, '_http_get', _stub):
            jobs_api.search_jobs(keyword='python', location='bangalore')
        self.assertIn('location=bangalore', recorded[0])   # /search-v2
        self.assertNotIn('location=', recorded[1])         # legacy /search

    def test_no_undocumented_parameters_are_sent(self):
        stub, calls = self.fake_http()
        with mock.patch.object(jobs_api, '_http_get', stub):
            jobs_api.search_jobs(keyword='python', work_mode='onsite')
        url = calls[0]['url']
        self.assertNotIn('work_from_home', url)   # on-site has no API parameter
        for invented in ('hybrid', 'salary_min', 'company', 'page_size', 'sort_by'):
            self.assertNotIn(invented, url)

    def test_remote_filter_drops_records_flagged_not_remote(self):
        # work_from_home=true is sent upstream, and the documented job_is_remote
        # flag is also enforced locally because Google's own matching is loose.
        stub, _ = self.fake_http()
        with mock.patch.object(jobs_api, '_http_get', stub):
            result = jobs_api.search_jobs(keyword='python', work_mode='remote')
        self.assertEqual([job['id'] for job in result['jobs']], ['JOB-2'])

    def test_onsite_filter_drops_remote_jobs(self):
        stub, _ = self.fake_http()
        with mock.patch.object(jobs_api, '_http_get', stub):
            result = jobs_api.search_jobs(keyword='python', work_mode='onsite')
        self.assertEqual([job['id'] for job in result['jobs']], ['JOB-1'])

    def test_api_key_is_not_part_of_the_result(self):
        stub, _ = self.fake_http()
        with mock.patch.object(jobs_api, '_http_get', stub):
            result = jobs_api.search_jobs(keyword='python')
        self.assertNotIn('test-key-not-real', json.dumps(result))

    def test_429_is_reported_as_rate_limited(self):
        stub, _ = self.fake_http(status=429, body=b'{"message":"Too many requests"}')
        with mock.patch.object(jobs_api, '_http_get', stub):
            with self.assertRaises(jobs_api.JobsApiError) as ctx:
                jobs_api.search_jobs(keyword='python')
        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(ctx.exception.code, 'rate_limited')

    def test_401_is_reported_as_key_problem(self):
        stub, _ = self.fake_http(status=401, body=b'{"message":"Invalid API key"}')
        with mock.patch.object(jobs_api, '_http_get', stub):
            with self.assertRaises(jobs_api.JobsApiError) as ctx:
                jobs_api.search_jobs(keyword='python')
        self.assertEqual(ctx.exception.status, 503)
        self.assertEqual(ctx.exception.code, 'unauthorized')

    def test_500_is_reported_as_upstream_failure(self):
        stub, _ = self.fake_http(status=500, body=b'oops')
        with mock.patch.object(jobs_api, '_http_get', stub):
            with self.assertRaises(jobs_api.JobsApiError) as ctx:
                jobs_api.search_jobs(keyword='python')
        self.assertEqual(ctx.exception.status, 502)

    def test_unreadable_body_is_reported(self):
        stub, _ = self.fake_http(status=200, body=b'<html>not json</html>')
        with mock.patch.object(jobs_api, '_http_get', stub):
            with self.assertRaises(jobs_api.JobsApiError) as ctx:
                jobs_api.search_jobs(keyword='python')
        self.assertEqual(ctx.exception.code, 'bad_response')

    def test_provider_error_envelope_is_surfaced(self):
        body = json.dumps({'status': 'ERROR',
                           'error': {'message': 'query is required', 'code': 400}}).encode()
        stub, _ = self.fake_http(status=200, body=body)
        with mock.patch.object(jobs_api, '_http_get', stub):
            with self.assertRaises(jobs_api.JobsApiError) as ctx:
                jobs_api.search_jobs(keyword='python')
        self.assertIn('query is required', ctx.exception.message)

    def test_placeholder_key_counts_as_unconfigured(self):
        for placeholder in ('YOUR_RAPIDAPI_KEY', 'your_rapidapi_key', 'CHANGEME', 'xxx-key'):
            with mock.patch.dict(os.environ, {'RAPIDAPI_KEY': placeholder}):
                self.assertFalse(jobs_api.is_configured(), placeholder)
        with mock.patch.dict(os.environ, {'RAPIDAPI_KEY': 'a' * 32}):
            self.assertTrue(jobs_api.is_configured())

    def test_placeholder_key_returns_503_without_calling_upstream(self):
        stub, calls = self.fake_http()
        with mock.patch.dict(os.environ, {'RAPIDAPI_KEY': 'YOUR_RAPIDAPI_KEY'}):
            with mock.patch.object(jobs_api, '_http_get', stub):
                with self.assertRaises(jobs_api.JobsApiError) as ctx:
                    jobs_api.search_jobs(keyword='python')
        self.assertEqual(ctx.exception.code, 'not_configured')
        self.assertEqual(calls, [])

    def test_missing_key_returns_503_without_calling_upstream(self):
        stub, calls = self.fake_http()
        with mock.patch.dict(os.environ, {'RAPIDAPI_KEY': ''}):
            with mock.patch.object(jobs_api, '_http_get', stub):
                with self.assertRaises(jobs_api.JobsApiError) as ctx:
                    jobs_api.search_jobs(keyword='python')
        self.assertEqual(ctx.exception.status, 503)
        self.assertEqual(ctx.exception.code, 'not_configured')
        self.assertEqual(calls, [])

    def test_network_timeout_maps_to_504(self):
        # Patch the transport, not _http_get, so the real error mapping runs.
        import urllib.error

        def _boom(request, timeout=None):
            raise urllib.error.URLError(TimeoutError('timed out'))

        with mock.patch.object(jobs_api.urllib.request, 'urlopen', _boom):
            with self.assertRaises(jobs_api.JobsApiError) as ctx:
                jobs_api.search_jobs(keyword='python')
        self.assertEqual(ctx.exception.status, 504)
        self.assertEqual(ctx.exception.code, 'timeout')

    def test_dns_failure_maps_to_502(self):
        import urllib.error

        def _boom(request, timeout=None):
            raise urllib.error.URLError('Name or service not known')

        with mock.patch.object(jobs_api.urllib.request, 'urlopen', _boom):
            with self.assertRaises(jobs_api.JobsApiError) as ctx:
                jobs_api.search_jobs(keyword='python')
        self.assertEqual(ctx.exception.status, 502)
        self.assertEqual(ctx.exception.code, 'unreachable')

    def test_v2_response_shape_is_understood(self):
        body = json.dumps({'status': 'OK',
                           'data': {'jobs': SAMPLE_JOBS, 'cursor': 'abc'}}).encode()
        stub, _ = self.fake_http(body=body)
        with mock.patch.object(jobs_api, '_http_get', stub):
            result = jobs_api.search_jobs(keyword='python')
        self.assertEqual(len(result['jobs']), 2)
        self.assertEqual(result['next_cursor'], 'abc')

    def test_has_more_follows_the_reported_cursor(self):
        body = json.dumps({'status': 'OK',
                           'data': {'jobs': SAMPLE_JOBS * 5, 'cursor': 'NEXT-PAGE'}}).encode()
        stub, _ = self.fake_http(body=body)
        with mock.patch.object(jobs_api, '_http_get', stub):
            result = jobs_api.search_jobs(keyword='python')
        self.assertEqual(len(result['jobs']), 10)
        self.assertEqual(result['next_cursor'], 'NEXT-PAGE')
        self.assertTrue(result['has_more'])

    def test_has_more_is_false_without_a_cursor(self):
        body = json.dumps({'status': 'OK', 'data': {'jobs': SAMPLE_JOBS}}).encode()
        stub, _ = self.fake_http(body=body)
        with mock.patch.object(jobs_api, '_http_get', stub):
            result = jobs_api.search_jobs(keyword='python')
        self.assertEqual(result['next_cursor'], '')
        self.assertFalse(result['has_more'])

    def test_cursor_is_sent_as_a_documented_parameter(self):
        stub, calls = self.fake_http()
        with mock.patch.object(jobs_api, '_http_get', stub):
            jobs_api.search_jobs(keyword='python', cursor='CUR-123')
        self.assertIn('cursor=CUR-123', calls[0]['url'])
        self.assertNotIn('page=', calls[0]['url'])

    def test_cursor_is_part_of_the_cache_key(self):
        stub, calls = self.fake_http()
        with mock.patch.object(jobs_api, '_http_get', stub):
            jobs_api.search_jobs(keyword='python')
            jobs_api.search_jobs(keyword='python', cursor='CUR-123')
        self.assertEqual(len(calls), 2)

    def test_falls_back_to_legacy_search_when_v2_is_missing(self):
        responses = [
            (404, b'{"message":"Endpoint \'/search-v2\' does not exist"}'),
            (200, json.dumps(_payload()).encode()),
        ]
        recorded = []

        def _stub(url, headers, timeout):
            recorded.append(url)
            return responses[len(recorded) - 1]

        with mock.patch.object(jobs_api, '_http_get', _stub):
            result = jobs_api.search_jobs(keyword='python', page=2)
        self.assertEqual(result['endpoint'], 'search')
        self.assertEqual(len(recorded), 2)
        self.assertIn('/search-v2?', recorded[0])
        self.assertIn('/search?', recorded[1])
        self.assertIn('page=2', recorded[1])   # page is documented for the legacy endpoint

    def test_reports_both_endpoints_missing(self):
        responses = [(404, b'{"message":"Endpoint \'/search-v2\' does not exist"}'),
                     (404, b'{"message":"Endpoint \'/search\' does not exist"}')]
        state = {'n': 0}

        def _stub(url, headers, timeout):
            response = responses[state['n']]
            state['n'] += 1
            return response

        with mock.patch.object(jobs_api, '_http_get', _stub):
            with self.assertRaises(jobs_api.JobsApiError) as ctx:
                jobs_api.search_jobs(keyword='python')
        self.assertEqual(ctx.exception.status, 502)
        self.assertIn('neither', ctx.exception.message)


class CacheTests(JobsTestBase):

    def test_identical_searches_hit_upstream_once(self):
        stub, calls = self.fake_http()
        with mock.patch.object(jobs_api, '_http_get', stub):
            first = jobs_api.search_jobs(keyword='python', location='pune')
            second = jobs_api.search_jobs(keyword='python', location='pune')
        self.assertEqual(len(calls), 1)
        self.assertFalse(first['cached'])
        self.assertTrue(second['cached'])
        self.assertEqual(first['jobs'], second['jobs'])

    def test_different_page_is_not_a_cache_hit(self):
        stub, calls = self.fake_http()
        with mock.patch.object(jobs_api, '_http_get', stub):
            jobs_api.search_jobs(keyword='python', page=1)
            jobs_api.search_jobs(keyword='python', page=2)
        self.assertEqual(len(calls), 2)

    def test_zero_ttl_disables_caching(self):
        stub, calls = self.fake_http()
        with mock.patch.dict(os.environ, {'JOBS_CACHE_TTL': '0'}):
            with mock.patch.object(jobs_api, '_http_get', stub):
                jobs_api.search_jobs(keyword='python')
                jobs_api.search_jobs(keyword='python')
        self.assertEqual(len(calls), 2)

    def test_case_insensitive_cache_key(self):
        stub, calls = self.fake_http()
        with mock.patch.object(jobs_api, '_http_get', stub):
            jobs_api.search_jobs(keyword='Python')
            jobs_api.search_jobs(keyword='python')
        self.assertEqual(len(calls), 1)


class RouteTests(JobsTestBase):

    def setUp(self):
        super().setUp()
        self.client = _auth_support.logged_in_client()

    def test_api_requires_login(self):
        anonymous = flask_app.app.test_client()
        response = anonymous.get('/api/jobs?q=python')
        self.assertEqual(response.status_code, 401)
        self.assertTrue(response.content_type.startswith('application/json'))

    def test_page_requires_login(self):
        anonymous = flask_app.app.test_client()
        response = anonymous.get('/jobs')
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login', response.headers['Location'])

    def test_page_renders_filters(self):
        response = self.client.get('/jobs')
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        for marker in ('id="q"', 'id="location"', 'id="date_posted"',
                       'id="employment_type"', 'id="work_mode"'):
            self.assertIn(marker, body)

    def test_search_returns_jobs_json(self):
        stub, _ = self.fake_http()
        with mock.patch.object(jobs_api, '_http_get', stub):
            response = self.client.get('/api/jobs?q=python&location=bangalore&country=in')
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['jobs'][0]['title'], 'Python Developer')
        self.assertEqual(payload['source'], 'JSearch (RapidAPI)')
        self.assertEqual(payload['endpoint'], 'search-v2')

    def test_invalid_filter_returns_json_400(self):
        response = self.client.get('/api/jobs?q=python&date_posted=yesterday')
        self.assertEqual(response.status_code, 400)
        self.assertTrue(response.content_type.startswith('application/json'))
        self.assertIn('date_posted', response.get_json()['error'])

    def test_blank_search_returns_json_400(self):
        response = self.client.get('/api/jobs')
        self.assertEqual(response.status_code, 400)
        self.assertIn('job title, skill or location', response.get_json()['error'])

    def test_rate_limit_is_passed_through_with_retry_after(self):
        stub, _ = self.fake_http(status=429, body=b'{"message":"quota"}')
        with mock.patch.object(jobs_api, '_http_get', stub):
            response = self.client.get('/api/jobs?q=python')
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers.get('Retry-After'), '60')
        self.assertEqual(response.get_json()['code'], 'rate_limited')

    def test_unconfigured_provider_returns_503(self):
        with mock.patch.dict(os.environ, {'RAPIDAPI_KEY': ''}):
            response = self.client.get('/api/jobs?q=python')
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()['code'], 'not_configured')

    def test_health_reports_configuration_without_the_key(self):
        response = self.client.get('/health')
        body = response.get_json()
        self.assertIn('jobs_api', body)
        self.assertTrue(body['jobs_api']['configured'])
        self.assertNotIn('test-key-not-real', json.dumps(body))


class EnvLoaderTests(unittest.TestCase):

    def test_parses_quotes_comments_and_export(self):
        from env_loader import parse_env_file
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.env', delete=False,
                                         encoding='utf-8') as handle:
            handle.write('# comment\nRAPIDAPI_KEY="abc123"\nexport OTHER=\'x y\'\n'
                         'EMPTY=\nBADLINE\n')
            path = handle.name
        try:
            values = parse_env_file(path)
        finally:
            os.unlink(path)
        self.assertEqual(values['RAPIDAPI_KEY'], 'abc123')
        self.assertEqual(values['OTHER'], 'x y')
        self.assertEqual(values['EMPTY'], '')
        self.assertNotIn('BADLINE', values)

    def test_does_not_override_existing_env(self):
        from env_loader import load_env
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.env', delete=False,
                                         encoding='utf-8') as handle:
            handle.write('PYTEST_FAKE_KEY=from-file\n')
            path = handle.name
        try:
            with mock.patch.dict(os.environ, {'PYTEST_FAKE_KEY': 'from-shell'}):
                load_env(path)
                self.assertEqual(os.environ['PYTEST_FAKE_KEY'], 'from-shell')
        finally:
            os.unlink(path)
            os.environ.pop('PYTEST_FAKE_KEY', None)

    def test_missing_file_is_not_an_error(self):
        from env_loader import load_env
        self.assertEqual(load_env('/definitely/not/here/.env'), [])


if __name__ == '__main__':
    unittest.main()
