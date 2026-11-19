from flask import Flask, render_template, request, jsonify
import logging
import os
import pandas as pd
import re

from ml_recommender import build_model, synonym_expansions
from env_loader import load_env
import auth
import experience_filter
import job_matching
import jobs_api

# Populate RAPIDAPI_KEY & friends from .env without overriding real env vars.
load_env()

app = Flask(__name__)
auth.init_app(app)

CSV_PATH = 'data/internship.csv'


def normalize_skill(skill):
    """Lower-case a skill token and collapse runs of whitespace to a single space.

    Keeps 'Basic  Computer', 'basic computer' and '  BASIC COMPUTER ' identical
    so user input matches the CSV regardless of spacing or case.
    """
    return re.sub(r'\s+', ' ', skill).strip().lower()


# --- Load and Prepare Internship Data ---
try:
    internships_df = pd.read_csv(CSV_PATH)
    
    # Clean the 'stipend' column
    internships_df['stipend'] = internships_df['stipend'].str.replace('₹', '').str.replace('/month', '').str.strip()
    internships_df['stipend'] = pd.to_numeric(internships_df['stipend'], errors='coerce').fillna(0).astype(int)
    
    # Create a master list of all unique skills from the dataset
    internships_df.dropna(subset=['required_skills'], inplace=True)
    all_skills = set()
    for skills_list in internships_df['required_skills'].str.lower().str.split(','):
        if isinstance(skills_list, list):
            for skill in skills_list:
                normalized = normalize_skill(skill)
                if normalized:
                    all_skills.add(normalized)
    
    # Fill any remaining NaN values
    internships_df.fillna('', inplace=True)

except FileNotFoundError as exc:
    raise FileNotFoundError(f"Internship data file not found: {CSV_PATH}") from exc

# --- TF-IDF relevance model -------------------------------------------------
# Fitted once per process at startup. Never refitted per request.
ml_model = build_model(internships_df)

# --- Scoring Weights for Accuracy ---
WEIGHT_SKILL = 15
WEIGHT_LOCATION = 10
WEIGHT_SECTOR = 5
WEIGHT_SKILL_PERCENTAGE = 10 # Bonus for matching a high percentage of skills
MAX_RESULTS = 5 # Upper end of the documented 3-5 recommendation range

# --- Recommendation modes ---
MODES = ('rule', 'ml', 'hybrid')
DEFAULT_MODE = 'rule'      # unchanged behaviour for API clients that do not send a mode
HYBRID_ML_WEIGHT = 0.6     # 60% TF-IDF relevance
HYBRID_RULE_WEIGHT = 0.4   # 40% rule-based relevance
RULE_SCORE_SCALE = 60.0    # reference maximum used to bring rule points into 0-1

# Every record in data/internship.csv is synthetic. The flag travels with every
# response so API consumers and the UI can label them instead of implying real openings.
DATA_IS_SYNTHETIC = True

# --- Flexible Education Hierarchy ---
EDUCATION_LEVELS = {
    '10th pass': 0, '12th pass': 1, 'diploma': 2,
    'ba': 3, 'b.sc': 3, 'b.com': 3, 'bba': 3, 'bca': 3,
    'b.tech': 3, 'b.design': 3, 'b.pharma': 3,
    'ma': 4, 'm.sc': 4, 'mba': 4, 'm.tech': 4, 'm.pharma': 4,
    'phd': 5, 'master\'s': 4, 'bachelor\'s': 3,
}

# Keywords to search for in freeform education text
# Each entry is (pattern, level). We use word-boundary-aware matching
# to prevent 'ba' matching inside 'mba', etc.
_EDUCATION_KEYWORDS = [
    (r'\b10th\b', 0),
    (r'\b12th\b', 1),
    (r'\bdiploma\b', 2),
    (r'\bb\.?tech\b', 3), (r'\bb\.?pharma\b', 3), (r'\bb\.?design\b', 3),
    (r'\bb\.?sc\b', 3), (r'\bb\.?com\b', 3), (r'\bbba\b', 3), (r'\bbca\b', 3),
    (r'\bba\b', 3), (r"\bbachelor'?s?\b", 3), (r'\bdegree\b', 3),
    (r'\bm\.?tech\b', 4), (r'\bm\.?pharma\b', 4), (r'\bm\.?sc\b', 4),
    (r'\bmba\b', 4), (r'\bma\b', 4), (r"\bmaster'?s?\b", 4), (r'\bms\b', 4),
    (r'\bphd\b', 5),
]

# Pre-compile the patterns for performance
_EDUCATION_PATTERNS = [(re.compile(pat, re.IGNORECASE), level) for pat, level in _EDUCATION_KEYWORDS]


def _parse_education_level(education_text):
    """Return the education level for recognised text, or None if nothing matches."""
    text = education_text.lower()

    # First try an exact match against the clean dictionary
    exact = EDUCATION_LEVELS.get(text)
    if exact is not None:
        return exact

    # Scan for keywords — find the lowest matching level (minimum requirement)
    matched_level = None
    for pattern, level in _EDUCATION_PATTERNS:
        if pattern.search(text):
            if matched_level is None or level < matched_level:
                matched_level = level

    return matched_level


def parse_education_level(education_text):
    """Parse a dataset requirement string into a level (never None).

    Used for internship rows, where unrecognised text falls back to 3 (bachelor's)
    because most listings target undergraduates. Candidate qualifications use
    resolve_candidate_level() instead, which rejects unrecognised values.
    """
    level = _parse_education_level(education_text)
    return 3 if level is None else level


# Accepted candidate qualifications, for the validation error message.
QUALIFICATION_EXAMPLES = (
    '10th Pass, 12th Pass, Diploma, B.Tech, B.Sc, B.Com, BBA, BCA, BA, B.Design, '
    'B.Pharma, M.Tech, M.Sc, MBA, MA, M.Pharma, PhD'
)


def resolve_candidate_level(qualification):
    """Resolve a candidate's qualification to a level, or raise ValueError.

    Unknown or empty values are rejected instead of silently defaulting to a
    bachelor's level: a default would hand the candidate eligibility they never
    claimed. Freeform text is still accepted when it contains a recognised degree
    keyword, e.g. "Pursuing B.Tech in Computer Science".
    """
    text = str(qualification or '').strip().lower()
    if not text:
        raise ValueError('qualification is required; send one of: ' + QUALIFICATION_EXAMPLES + '.')

    level = _parse_education_level(text)
    if level is None:
        raise ValueError(
            f"Unsupported qualification {qualification!r}. Send one of: {QUALIFICATION_EXAMPLES}.")
    return level

# --- Location Map ---
# The dropdown offers states, but the dataset stores cities.
STATE_TO_CITIES = {
    'karnataka': ['bangalore'],
    'maharashtra': ['mumbai', 'pune'],
    'telangana': ['hyderabad'],
    'haryana': ['gurgaon'],
    'delhi': ['delhi'],
    'tamil nadu': ['chennai'],
}

@app.route('/')
@auth.login_required
def home():
    """Serves your main HTML page and passes the list of skills."""
    return render_template('index.html', all_skills=sorted(list(all_skills)))

def _describe_match(mode, matched_skills, synonym_matches, required_skill_count, matched_sector,
                    matched_location, ml_similarity, rule_score, rule_norm, hybrid_score):
    """Plain-language explanation built only from values the algorithm actually used."""
    parts = []
    if matched_skills:
        parts.append(f"matched {len(matched_skills)} of {required_skill_count} required skills "
                     f"({', '.join(sorted(matched_skills))})")
    elif synonym_matches:
        pairs = ', '.join(f"'{phrase}' means '{canonical}'" for phrase, canonical in synonym_matches)
        parts.append(f"matched {len(synonym_matches)} required skill(s) through synonym mapping "
                     f"({pairs})")
    else:
        parts.append('no exact skill overlap')
    if matched_sector:
        parts.append('sector preference matches')
    if matched_location:
        parts.append('location preference matches')

    if mode == 'ml':
        lead = f"Text similarity {ml_similarity:.2f} (cosine over TF-IDF vectors, 0-1)."
    elif mode == 'hybrid':
        lead = (f"Hybrid score {hybrid_score:.2f} = {HYBRID_ML_WEIGHT:.0%} text similarity "
                f"({ml_similarity:.2f}) + {HYBRID_RULE_WEIGHT:.0%} rule score "
                f"({rule_score:.0f} points, scaled to {rule_norm:.2f}).")
    else:
        lead = f"Rule score {rule_score:.0f} points (skills + sector + location)."
    return f"{lead} {('; '.join(parts)).capitalize()}."


def rank_internships(qualification='', skills='', sector='', location='',
                     mode=DEFAULT_MODE, ml_weight=HYBRID_ML_WEIGHT):
    """Rank internships for one candidate profile.

    Pure function with no Flask objects, so the same code path serves the API and
    the evaluation script. Education eligibility is applied before anything else
    and cannot be overridden by text similarity.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown recommendation mode {mode!r}; choose from {', '.join(MODES)}.")

    qualification = str(qualification or '').lower()
    skills_text = str(skills or '').lower()
    sector = str(sector or '').lower()
    location = str(location or '').lower()

    # Filter user skills to only include skills that exist in our master list
    user_skills_list = [normalize_skill(skill) for skill in skills_text.split(',') if skill.strip()]
    valid_user_skills = set(skill for skill in user_skills_list if skill in all_skills)

    # Candidate qualification is required and must be recognised: an unknown value
    # raises ValueError (the API maps it to a 400) instead of defaulting to a level
    # the candidate never claimed.
    user_education_level = resolve_candidate_level(qualification)

    # One TF-IDF query per request, reused for every row.
    similarity_by_id = ml_model.similarity_by_id(skills_text)

    # Candidate skill phrases that only match the dataset through the synonym table.
    synonym_pairs = synonym_expansions(skills_text)

    scored_internships = []
    seen_ids = set()

    for index, internship in internships_df.iterrows():

        required_skills = set(normalize_skill(skill) for skill in internship['required_skills'].split(',') if skill.strip())

        # Education is a hard filter in every mode. Text similarity never overrides it.
        required_education_level = parse_education_level(internship['required_education'])
        if user_education_level < required_education_level:
            continue

        matching_skills = valid_user_skills.intersection(required_skills)

        # Which of the candidate's synonym phrases this row actually requires.
        row_canonical_skills = ml_model.canonical_skills_by_id.get(internship['internship_id'], set())
        synonym_matches = [(phrase, canonical) for phrase, canonical in synonym_pairs
                           if canonical in row_canonical_skills and canonical not in matching_skills]

        # The exact-skill gate is the historical rule-engine behaviour. In the ML modes
        # the measured similarity decides relevance instead, so near-matches can surface.
        if mode == 'rule' and skills_text and not matching_skills:
            continue

        # --- Rule-based score (unchanged weights) ---
        current_score = 0

        matched_location = False
        if location:
            accepted_locations = STATE_TO_CITIES.get(location, [location])
            matched_location = any(city in internship['location'].lower() for city in accepted_locations)
            if matched_location:
                current_score += WEIGHT_LOCATION

        matched_sector = bool(sector) and sector in internship['sector'].lower()
        if matched_sector:
            current_score += WEIGHT_SECTOR

        if required_skills and matching_skills:
            current_score += len(matching_skills) * WEIGHT_SKILL
            current_score += (len(matching_skills) / len(required_skills)) * WEIGHT_SKILL_PERCENTAGE

        # --- Normalised components ---
        ml_similarity = float(similarity_by_id.get(internship['internship_id'], 0.0))
        rule_norm = min(current_score / RULE_SCORE_SCALE, 1.0)
        ml_norm = min(max(ml_similarity, 0.0), 1.0)
        hybrid_score = ml_weight * ml_norm + (1.0 - ml_weight) * rule_norm

        if mode == 'rule':
            ranking_score = float(current_score)
        elif mode == 'ml':
            ranking_score = ml_norm
        else:
            ranking_score = hybrid_score

        # A record with no positive score is not a match; never pad the list with it.
        # Duplicate internship_ids in the CSV are counted once.
        if ranking_score <= 0 or internship['internship_id'] in seen_ids:
            continue
        seen_ids.add(internship['internship_id'])

        internship_data = internship.to_dict()
        internship_data.update({
            'score': ranking_score,
            'mode': mode,
            'rule_score': round(current_score, 1),
            'tfidf_similarity': round(ml_similarity, 3),
            'hybrid_score': round(hybrid_score, 3),
            'matched_skills': sorted(matching_skills),
            'synonym_matches': [{'input': phrase, 'dataset_skill': canonical}
                                for phrase, canonical in synonym_matches],
            'matched_sector': matched_sector,
            'matched_location': matched_location,
            'is_sample': DATA_IS_SYNTHETIC,
        })
        internship_data['explanation'] = _describe_match(
            mode, matching_skills, synonym_matches, len(required_skills), matched_sector,
            matched_location, ml_similarity, current_score, rule_norm, hybrid_score)
        scored_internships.append(internship_data)

    # Rank by score (raw, unrounded) and return the best MAX_RESULTS distinct internships
    sorted_internships = sorted(scored_internships, key=lambda x: x['score'], reverse=True)
    top_recommendations = sorted_internships[:MAX_RESULTS]
    for recommendation in top_recommendations:
        recommendation['score'] = (round(recommendation['score'], 1) if mode == 'rule'
                                   else round(recommendation['score'], 3))

    return top_recommendations


@app.route('/recommend', methods=['POST'])
@auth.login_required
def recommend():
    """
    API endpoint that receives candidate data and returns recommendations.

    Optional "mode": "rule" (default), "ml" or "hybrid".
    """
    data = request.get_json(silent=True)
    if data is None:
        if request.data:
            return jsonify({'error': 'Request body is not valid JSON.'}), 400
        return jsonify({'error': 'Request body is required.'}), 400
    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object.'}), 400

    # Extract candidate details
    user_qualification = str(data.get('qualification') or '').lower()
    user_skills_str = str(data.get('skills') or '').lower()
    user_sector = str(data.get('sector_interested') or '').lower()
    user_location = str(data.get('location_interested') or '').lower()
    mode = str(data.get('mode') or DEFAULT_MODE).strip().lower()

    if mode not in MODES:
        return jsonify({'error': f"Unknown mode {mode!r}. Choose one of: {', '.join(MODES)}."}), 400

    # Refuse a completely blank profile instead of returning an empty list
    if not (user_qualification.strip() or user_skills_str.strip() or user_sector.strip() or user_location.strip()):
        return jsonify({'error': 'Please provide at least one of: qualification, skills, sector_interested, location_interested.'}), 400

    try:
        recommendations = rank_internships(
            user_qualification, user_skills_str, user_sector, user_location, mode=mode)
    except ValueError as exc:
        # Unsupported or empty qualification -> JSON 400, never a silent default.
        return jsonify({'error': str(exc)}), 400

    return jsonify(recommendations)


# --- External job search (JSearch / RapidAPI) --------------------------------
# The API key stays server-side: the browser only ever talks to these routes.

JOB_SEARCH_COUNTRIES = (
    ('in', 'India'), ('us', 'United States'), ('gb', 'United Kingdom'),
    ('ca', 'Canada'), ('au', 'Australia'), ('ae', 'United Arab Emirates'),
    ('sg', 'Singapore'), ('de', 'Germany'), ('nz', 'New Zealand'),
)


@app.route('/jobs')
@auth.login_required
def jobs_page():
    """Live job search page backed by the JSearch API."""
    args = request.args
    profile = job_matching.build_profile(
        skills=args.get('skills', ''),
        qualification=args.get('qualification', ''),
        sector=args.get('sector', ''),
        location=args.get('location', ''),
    )
    # A profile with skills but no keyword still describes a searchable role.
    prefill_query = (args.get('q') or ' '.join(profile['skills'][:2])).strip()

    return render_template(
        'jobs.html',
        countries=JOB_SEARCH_COUNTRIES,
        default_country=jobs_api.default_country(),
        date_options=jobs_api.DATE_POSTED_VALUES,
        employment_options=jobs_api.EMPLOYMENT_TYPES,
        work_modes=jobs_api.WORK_MODES,
        api_configured=jobs_api.is_configured(),
        profile=profile,
        prefill_query=prefill_query,
        prefill_location=args.get('location', ''),
        entry_level=((args.get('entry_level') or '1').strip().lower()
                     not in ('0', 'false', 'no', 'off')),
    )


@app.route('/api/jobs')
@auth.login_required
def jobs_search():
    """Proxy one JSearch query. Returns JSON, never the API key."""
    args = request.args
    profile = job_matching.build_profile(
        skills=args.get('skills', ''),
        qualification=args.get('qualification', ''),
        sector=args.get('sector', ''),
        location=args.get('location', ''),
    )

    # With a profile but no keyword, search whatever the profile describes.
    keyword = (args.get('q') or '').strip()
    if not keyword and not (args.get('location') or '').strip() and profile['skills']:
        keyword = ' '.join(profile['skills'][:2])

    entry_level = ((args.get('entry_level') or '1').strip().lower()
                   not in ('0', 'false', 'no', 'off'))

    try:
        result = jobs_api.search_jobs(
            keyword=keyword,
            location=args.get('location', ''),
            country=args.get('country'),
            date_posted=args.get('date_posted', 'all'),
            employment_types=args.get('employment_type', ''),
            work_mode=args.get('work_mode', 'all'),
            page=args.get('page', 1),
            cursor=args.get('cursor', ''),
            # Documented /search-v2 parameter; only the no-experience value suits internships.
            job_requirements=('no_experience',) if entry_level else (),
            num_pages=args.get('num_pages', jobs_api.DEFAULT_NUM_PAGES),
        )
    except jobs_api.JobsApiError as exc:
        response = jsonify({'error': exc.message, 'code': exc.code})
        response.status_code = exc.status
        if exc.status == 429:
            response.headers['Retry-After'] = '60'
        return response

    # Belt-and-braces: Google's requirement matching is advisory, so the drop is
    # repeated locally on title seniority and stated years.
    if entry_level:
        kept, dropped = experience_filter.filter_internships(result['jobs'])
        result['jobs'] = kept
        result['filtered_out'] = len(dropped)
        result['filtered_samples'] = dropped[:5]
    else:
        result['filtered_out'] = 0
        result['filtered_samples'] = []
    result['entry_level'] = entry_level
    result['max_experience_years'] = experience_filter.max_experience_years()
    result['count'] = len(result['jobs'])   # count must describe the filtered list

    # Scoring happens after the cache lookup, so cached payloads stay profile-free.
    result['jobs'] = job_matching.rank_jobs(result['jobs'], profile)
    result['profile'] = {
        'active': profile['active'],
        'skills': profile['skills'],
        'sector': profile['sector'],
        'qualification': profile['qualification'],
        'location': profile['location'],
    }
    return jsonify(result)


@app.route('/health', methods=['GET'])
def health():
    """Liveness/readiness probe for App Runner and load balancers."""
    return jsonify({
        'status': 'ok',
        'model': {
            'type': 'tfidf-cosine',
            'documents': ml_model.document_count,
            'features': ml_model.vocabulary_size,
            'fitted': ml_model.fitted,
        },
        'dataset': {'records': int(len(internships_df)), 'synthetic': DATA_IS_SYNTHETIC},
        'jobs_api': {'provider': 'jsearch', 'configured': jobs_api.is_configured(),
                     'cache_entries': jobs_api.cache_size()},
    })

_PROBE_LOG_PATTERN = re.compile(r'"POST / HTTP/1\.[01]" 405')


def quiet_probe_noise(logger_name='werkzeug'):
    """Drop access-log lines caused by other software probing this port.
    
    Steam's client opens connections to localhost:5000 and posts to '/', which no
    route here accepts, so the dev log fills with 405s. Only that exact pattern is
    filtered; every real request is still logged. Returns the filter for testing.
    """
    class _DropProbePosts(logging.Filter):
        def filter(self, record):
            try:
                message = record.getMessage()
            except Exception:  # pragma: no cover - defensive
                return True
            return not _PROBE_LOG_PATTERN.search(message)

    probe_filter = _DropProbePosts()
    logging.getLogger(logger_name).addFilter(probe_filter)
    return probe_filter


if __name__ == '__main__':
    quiet_probe_noise()
    debug = os.environ.get('FLASK_DEBUG', '').lower() in ('1', 'true', 'yes', 'on')
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=debug, host=host, port=port)
