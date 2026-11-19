"""JSearch (RapidAPI) job-search client.

Server-side only: the RapidAPI key never reaches the browser, every upstream
call is made here, and identical searches are served from a small in-memory
TTL cache so a burst of users costs one API credit instead of one per request.

Only parameters documented for JSearch ``GET /search`` are sent:
``query``, ``page``, ``num_pages``, ``country``, ``language``, ``date_posted``,
``work_from_home``, ``employment_types``, ``job_requirements``, ``radius``,
``exclude_job_publishers``, ``fields``.

Configuration (environment)
---------------------------
RAPIDAPI_KEY           RapidAPI key. Without it every search returns 503.
JSEARCH_DEFAULT_COUNTRY  ISO-3166 alpha-2 country used when none is given (default ``in``).
JOBS_CACHE_TTL         Seconds a response stays cached (default 600).
JOBS_API_TIMEOUT       Upstream request timeout in seconds (default 12).
"""
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

RAPIDAPI_HOST = 'jsearch.p.rapidapi.com'
# Verified live: this RapidAPI deployment serves only cursor-paginated /search-v2 and
# answers /search with 404 "Endpoint '/search' does not exist". Older deployments are the
# reverse, so both are attempted in order and the one that answered is reported.
SEARCH_ENDPOINTS = (
    ('search-v2', 'https://' + RAPIDAPI_HOST + '/search-v2'),
    ('search', 'https://' + RAPIDAPI_HOST + '/search'),
)
SEARCH_URL = SEARCH_ENDPOINTS[0][1]

# --- documented enumerations (anything else is rejected, never silently ignored)
DATE_POSTED_VALUES = ('all', 'today', '3days', 'week', 'month')
EMPLOYMENT_TYPES = ('FULLTIME', 'PARTTIME', 'CONTRACTOR', 'INTERN')
# Documented values for /search-v2 job_requirements. 'no_experience' is the only one that
# fits an internship product; 'no_degree' returned 0 results in a live probe.
JOB_REQUIREMENTS = ('under_3_years_experience', 'more_than_3_years_experience',
                    'no_experience', 'no_degree')
WORK_MODES = ('all', 'remote', 'onsite')
# JSearch exposes work_from_home only; there is no hybrid flag in the response,
# so 'hybrid' is deliberately not offered.

MAX_PAGE = 20           # documented page range for /search is 1-20
RESULTS_PER_PAGE = 10   # documented page size
MAX_KEYWORD_LENGTH = 120
MAX_LOCATION_LENGTH = 80
DEFAULT_NUM_PAGES = 1

CACHE_MAX_ENTRIES = 256

_CACHE = {}
_CACHE_LOCK = threading.Lock()


class JobsApiError(Exception):
    """Carries the HTTP status and machine-readable code the route should return."""

    def __init__(self, message, status=502, code='upstream'):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code


# --- configuration ----------------------------------------------------------

def api_key():
    return (os.environ.get('RAPIDAPI_KEY') or '').strip()


PLACEHOLDER_MARKERS = ('your_rapidapi_key', 'your_rapidapi', 'changeme', '<your', 'xxx')


def is_configured():
    """True when a usable-looking key is present. The key itself is never returned."""
    key = api_key()
    if not key:
        return False
    lowered = key.lower()
    return not any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def cache_ttl():
    try:
        value = int(os.environ.get('JOBS_CACHE_TTL', ''))
    except (TypeError, ValueError):
        return 600
    return value if value >= 0 else 600


def request_timeout():
    try:
        value = float(os.environ.get('JOBS_API_TIMEOUT', ''))
    except (TypeError, ValueError):
        return 12.0
    return value if value > 0 else 12.0


def default_country():
    value = (os.environ.get('JSEARCH_DEFAULT_COUNTRY') or 'in').strip().lower()
    return value if len(value) == 2 and value.isalpha() else 'in'


# --- validation -------------------------------------------------------------

def _clean(value, limit):
    text = ' '.join(str(value or '').split())
    return text[:limit]


def _as_int(value, field, default, minimum, maximum):
    if value is None or str(value).strip() == '':
        return default
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        raise JobsApiError(f'{field} must be a whole number.', 400, 'invalid_request')
    if number < minimum or number > maximum:
        raise JobsApiError(f'{field} must be between {minimum} and {maximum}.', 400, 'invalid_request')
    return number


def validate_search(keyword='', location='', country=None, date_posted='all',
                    employment_types=(), work_mode='all', page=1,
                    num_pages=DEFAULT_NUM_PAGES, cursor='', job_requirements=()):
    """Validate and normalise user input into a request plan.

    Unknown enum values raise JobsApiError(400) that names the accepted values —
    a filter is never dropped silently, because that would answer a narrower
    question with the wrong jobs.
    """
    keyword = _clean(keyword, MAX_KEYWORD_LENGTH)
    location = _clean(location, MAX_LOCATION_LENGTH)

    if not keyword and not location:
        raise JobsApiError(
            'Enter a job title, skill or location to search.', 400, 'invalid_request')

    # The API takes one free-form query; location belongs inside it.
    if keyword and location:
        query = f'{keyword} in {location}'
    elif keyword:
        query = keyword
    else:
        query = f'jobs in {location}'

    # Never truncate here: 'usa' must fail loudly instead of becoming 'us'.
    country = ' '.join(str(country or '').split()).lower() or default_country()
    if len(country) != 2 or not country.isalpha():
        raise JobsApiError(
            'country must be a two-letter ISO-3166 code such as in, us or gb.',
            400, 'invalid_request')

    date_posted = _clean(date_posted, 10).lower() or 'all'
    if date_posted not in DATE_POSTED_VALUES:
        raise JobsApiError(
            'date_posted must be one of: ' + ', '.join(DATE_POSTED_VALUES) + '.',
            400, 'invalid_request')

    if isinstance(employment_types, str):
        employment_types = [part for part in employment_types.split(',')]
    cleaned_types = []
    for raw_type in employment_types or ():
        value = str(raw_type or '').strip().upper()
        if not value:
            continue
        if value not in EMPLOYMENT_TYPES:
            raise JobsApiError(
                'employment_types must be a comma-separated subset of: '
                + ', '.join(EMPLOYMENT_TYPES) + '.', 400, 'invalid_request')
        if value not in cleaned_types:
            cleaned_types.append(value)

    if isinstance(job_requirements, str):
        job_requirements = [part for part in job_requirements.split(',')]
    cleaned_requirements = []
    for raw_value in job_requirements or ():
        value = str(raw_value or '').strip().lower()
        if not value:
            continue
        if value not in JOB_REQUIREMENTS:
            raise JobsApiError(
                'job_requirements must be a comma-separated subset of: '
                + ', '.join(JOB_REQUIREMENTS) + '.', 400, 'invalid_request')
        if value not in cleaned_requirements:
            cleaned_requirements.append(value)

    work_mode = _clean(work_mode, 10).lower() or 'all'
    if work_mode not in WORK_MODES:
        raise JobsApiError(
            'work_mode must be one of: ' + ', '.join(WORK_MODES) + '.',
            400, 'invalid_request')

    page = _as_int(page, 'page', 1, 1, MAX_PAGE)
    num_pages = _as_int(num_pages, 'num_pages', DEFAULT_NUM_PAGES, 1, MAX_PAGE)
    if page + num_pages - 1 > MAX_PAGE:
        raise JobsApiError(
            f'page + num_pages must not exceed {MAX_PAGE}.', 400, 'invalid_request')

    return {
        'query': query,
        'cursor': _clean(cursor, 4000),
        'keyword': keyword,
        'location': location,
        'country': country,
        'language': '',
        'date_posted': date_posted,
        'employment_types': cleaned_types,
        'job_requirements': cleaned_requirements,
        'work_mode': work_mode,
        'page': page,
        'num_pages': num_pages,
    }


# --- upstream call ----------------------------------------------------------

def _upstream_params(plan, endpoint='search-v2'):
    """Translate the validated plan into the parameters documented for that endpoint."""
    params = {
        'query': plan['query'],
        'num_pages': plan['num_pages'],
        'country': plan['country'],
        'date_posted': plan['date_posted'],
    }
    if endpoint == 'search':
        params['page'] = plan['page']          # documented for the legacy endpoint
    else:
        if plan['cursor']:
            params['cursor'] = plan['cursor']  # documented for /search-v2
        if plan['location']:
            # Documented for /search-v2 only: Google's UULE search origin. Measured live,
            # it changes the result set completely (0/10 overlap with query-only) and a
            # location Google does not recognise returns zero jobs.
            params['location'] = plan['location']
    if plan['language']:
        params['language'] = plan['language']
    if plan['employment_types']:
        params['employment_types'] = ','.join(plan['employment_types'])
    if plan['job_requirements']:
        params['job_requirements'] = ','.join(plan['job_requirements'])
    if plan['work_mode'] == 'remote':
        params['work_from_home'] = 'true'
    return params


def _http_get(url, headers, timeout):
    """GET a URL and return (status, body bytes). Network faults raise JobsApiError."""
    request = urllib.request.Request(url, headers=headers, method='GET')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:  # 4xx/5xx still carry a body
        try:
            body = exc.read()
        except Exception:  # pragma: no cover - defensive
            body = b''
        return exc.code, body
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, 'reason', exc)
        if 'timed out' in str(reason).lower() or isinstance(reason, TimeoutError):
            raise JobsApiError('The job provider did not respond in time. Try again.',
                               504, 'timeout')
        raise JobsApiError('Could not reach the job provider. Check the network and try again.',
                           502, 'unreachable')


def _request(plan):
    key = api_key()
    if not is_configured():
        # Catches both a missing key and an unfilled placeholder such as YOUR_RAPIDAPI_KEY,
        # so the UI never reports a provider 401 it could have predicted.
        raise JobsApiError(
            'Job search is not configured: set RAPIDAPI_KEY in the server environment '
            'or .env file.', 503, 'not_configured')

    headers = {
        'X-RapidAPI-Key': key,
        'X-RapidAPI-Host': RAPIDAPI_HOST,
        'Accept': 'application/json',
    }

    for endpoint, url in SEARCH_ENDPOINTS:
        query_string = urllib.parse.urlencode(_upstream_params(plan, endpoint))
        status, body = _http_get(url + '?' + query_string, headers, request_timeout())
        if status == 404 and b'does not exist' in body.lower():
            continue   # this deployment does not expose that endpoint; try the other
        return _interpret(status, body), endpoint

    raise JobsApiError(
        'The JSearch subscription exposes neither /search-v2 nor /search. '
        'Check the subscription on RapidAPI.', 502, 'upstream')


def _interpret(status, body):
    """Turn an upstream status + body into a payload dict, or raise JobsApiError."""
    if status == 429:
        raise JobsApiError(
            'The job provider rate limit was reached. Please wait a moment and retry.',
            429, 'rate_limited')
    if status in (401, 403):
        raise JobsApiError(
            'The job provider rejected the API key. Check RAPIDAPI_KEY and the '
            'JSearch subscription.', 503, 'unauthorized')
    if status >= 500:
        raise JobsApiError('The job provider is unavailable right now. Try again shortly.',
                           502, 'upstream')
    if status >= 400:
        raise JobsApiError(_provider_message(body) or 'The job provider rejected the search.',
                           502, 'upstream')

    try:
        payload = json.loads(body.decode('utf-8', 'replace'))
    except (ValueError, UnicodeDecodeError):
        raise JobsApiError('The job provider returned an unreadable response.', 502, 'bad_response')

    if not isinstance(payload, dict):
        raise JobsApiError('The job provider returned an unexpected response.', 502, 'bad_response')
    if str(payload.get('status', 'OK')).upper() == 'ERROR':
        error = payload.get('error') or {}
        message = error.get('message') if isinstance(error, dict) else str(error)
        raise JobsApiError(message or 'The job provider reported an error.', 502, 'upstream')
    return payload


def _provider_message(body):
    try:
        payload = json.loads(body.decode('utf-8', 'replace'))
    except (ValueError, UnicodeDecodeError):
        return ''
    if isinstance(payload, dict):
        error = payload.get('error')
        if isinstance(error, dict):
            return str(error.get('message') or '')
        if isinstance(error, str):
            return error
        if payload.get('message'):
            return str(payload['message'])
    return ''


# --- normalisation ----------------------------------------------------------

def _text(value):
    return str(value).strip() if value is not None else ''


def _salary(raw):
    minimum, maximum = raw.get('job_min_salary'), raw.get('job_max_salary')
    currency = _text(raw.get('job_salary_currency')) or ''
    period = _text(raw.get('job_salary_period')).lower()
    if minimum is None and maximum is None:
        return ''
    def amount(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ''
        return f'{number:,.0f}'
    low, high = amount(minimum), amount(maximum)
    if low and high and low != high:
        body = f'{currency} {low} - {high}'.strip()
    else:
        body = f'{currency} {low or high}'.strip()
    return f'{body} per {period}' if period else body


def _location(raw):
    parts = [_text(raw.get('job_city')), _text(raw.get('job_state')),
             _text(raw.get('job_country'))]
    seen, ordered = set(), []
    for part in parts:
        if part and part.lower() not in seen:
            seen.add(part.lower())
            ordered.append(part)
    return ', '.join(ordered) or _text(raw.get('job_location'))


def _description(raw, limit=6000):
    text = _text(raw.get('job_description'))
    return text[:limit]


def normalize_job(raw):
    """Map one JSearch record to the shape the UI renders. Missing fields degrade."""
    if not isinstance(raw, dict):
        return None

    apply_link = _text(raw.get('job_apply_link')) or _text(raw.get('job_google_link'))
    employment_types = raw.get('job_employment_types')
    if not isinstance(employment_types, list):
        employment_types = []

    posted_iso = _text(raw.get('job_posted_at_datetime_utc'))
    posted_at = _text(raw.get('job_posted_at'))

    skills = [str(skill).strip() for skill in (raw.get('job_required_skills') or [])
              if str(skill).strip()]

    return {
        'id': _text(raw.get('job_id')),
        'title': _text(raw.get('job_title')) or 'Untitled job',
        'company': _text(raw.get('employer_name')) or 'Employer not listed',
        'company_logo': _text(raw.get('employer_logo')),
        'company_website': _text(raw.get('employer_website')),
        'publisher': _text(raw.get('job_publisher')),
        'location': _location(raw),
        'is_remote': raw.get('job_is_remote'),
        'employment_type': _text(raw.get('job_employment_type')),
        'employment_types': [str(item).strip() for item in employment_types if str(item).strip()],
        'posted_at': posted_iso,
        'posted_at_label': posted_at,
        'description': _description(raw),
        'salary': _salary(raw),
        'apply_link': apply_link,
        'apply_is_direct': bool(raw.get('job_apply_is_direct')),
        'source_link': _text(raw.get('job_google_link')),
        'required_skills': skills[:12],
        'is_sample': False,
        'source': 'JSearch',
    }


def _jobs_from_payload(payload):
    """Accept both response shapes: v1 returns data as a list, v2 nests data.jobs."""
    data = payload.get('data')
    if isinstance(data, list):
        return data, ''
    if isinstance(data, dict):
        jobs = data.get('jobs')
        cursor = _text(data.get('cursor'))
        if isinstance(jobs, list):
            return jobs, cursor
    return [], ''


# --- cache ------------------------------------------------------------------

def _cache_key(plan):
    return (plan['query'].lower(), plan['country'], plan['date_posted'],
            tuple(plan['employment_types']), tuple(plan['job_requirements']),
            plan['work_mode'], plan['cursor'], plan['page'], plan['num_pages'],
            plan['language'])


def _cache_get(key):
    now = time.time()
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        if entry['expires'] <= now:
            _CACHE.pop(key, None)
            return None
        return entry['payload']


def _cache_set(key, payload):
    ttl = cache_ttl()
    if ttl <= 0:
        return
    with _CACHE_LOCK:
        if len(_CACHE) >= CACHE_MAX_ENTRIES:
            now = time.time()
            for stale in [k for k, v in _CACHE.items() if v['expires'] <= now]:
                _CACHE.pop(stale, None)
            if len(_CACHE) >= CACHE_MAX_ENTRIES:
                oldest = min(_CACHE, key=lambda k: _CACHE[k]['expires'])
                _CACHE.pop(oldest, None)
        _CACHE[key] = {'expires': time.time() + ttl, 'payload': payload}


def clear_cache():
    with _CACHE_LOCK:
        _CACHE.clear()


def cache_size():
    with _CACHE_LOCK:
        return len(_CACHE)


# --- public entry point -----------------------------------------------------

def search_jobs(keyword='', location='', country=None, date_posted='all',
                employment_types=(), work_mode='all', page=1,
                num_pages=DEFAULT_NUM_PAGES, cursor='', job_requirements=()):
    """Search JSearch and return a UI-ready payload. Raises JobsApiError on failure.

    Pagination is cursor-based: pass the ``next_cursor`` of a previous result to
    fetch the following batch. ``page`` only applies to the legacy /search
    endpoint, which this deployment does not serve.
    """
    plan = validate_search(keyword=keyword, location=location, country=country,
                           date_posted=date_posted, employment_types=employment_types,
                           work_mode=work_mode, page=page, num_pages=num_pages,
                           cursor=cursor, job_requirements=job_requirements)

    key = _cache_key(plan)
    cached = _cache_get(key)
    if cached is not None:
        return dict(cached, cached=True)

    payload, endpoint = _request(plan)
    raw_jobs, next_cursor = _jobs_from_payload(payload)

    jobs = [job for job in (normalize_job(raw) for raw in raw_jobs) if job]

    # work_from_home=true covers the remote side upstream; the on-site side has
    # no API parameter, so it is filtered here on the documented job_is_remote flag.
    if plan['work_mode'] == 'remote':
        jobs = [job for job in jobs if job['is_remote'] is not False]
    elif plan['work_mode'] == 'onsite':
        jobs = [job for job in jobs if job['is_remote'] is False]

    # /search-v2 reports an explicit cursor; the legacy endpoint is page-based.
    if next_cursor:
        has_more = True
    elif endpoint == 'search':
        has_more = (bool(jobs) and len(jobs) >= RESULTS_PER_PAGE * plan['num_pages']
                    and plan['page'] + plan['num_pages'] - 1 < MAX_PAGE)
    else:
        has_more = False

    result = {
        'jobs': jobs,
        'count': len(jobs),
        'page': plan['page'],
        'num_pages': plan['num_pages'],
        'has_more': has_more,
        'next_cursor': next_cursor,
        'endpoint': endpoint,
        'filters': {key_name: plan[key_name] for key_name in
                    ('keyword', 'location', 'country', 'date_posted',
                     'employment_types', 'job_requirements', 'work_mode', 'page')},
        'source': 'JSearch (RapidAPI)',
        'cached': False,
    }
    _cache_set(key, result)
    return result
