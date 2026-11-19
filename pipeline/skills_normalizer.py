"""Canonical skill names and Indian location normalisation for the data pipeline.

This module is used by the offline preprocessing/audit scripts only. The live Flask
app keeps its own matching rules so existing behaviour is untouched.
"""
import re

# canonical skill -> accepted spellings (all lower-case)
SKILL_CANONICAL = {
    'excel': ['excel', 'ms excel', 'microsoft excel', 'advanced excel', 'excel sheet', 'spreadsheet'],
    'basic computer': ['basic computer', 'computer basics', 'basic computer knowledge',
                       'basic computer skills', 'computer literacy', 'computer knowledge'],
    'data entry': ['data entry', 'data entry operation', 'data entry operator', 'data entry work'],
    'communication': ['communication', 'communications', 'verbal communication',
                      'communication skills', 'written communication'],
    'tally': ['tally', 'tally erp', 'tally erp 9', 'tally prime'],
    'python': ['python', 'python3', 'python 3'],
    'sql': ['sql', 'mysql', 'postgresql', 'sql server', 'pl/sql', 'plsql'],
    'java': ['java', 'core java', 'java 8'],
    'customer service': ['customer service', 'customer support', 'customer care',
                         'client service', 'customer success'],
    'troubleshooting': ['troubleshooting', 'technical support', 'hardware troubleshooting',
                        'problem solving and troubleshooting'],
    'typing': ['typing', 'typing speed', 'data typing'],
    'inventory': ['inventory', 'inventory management', 'stock management', 'stock keeping'],
    'javascript': ['javascript', 'js', 'es6'],
    'react': ['react', 'reactjs', 'react.js'],
    'node.js': ['node.js', 'node js', 'nodejs'],
    'machine learning': ['machine learning', 'ml'],
    'deep learning': ['deep learning', 'dl'],
    'aws': ['aws', 'amazon web services'],
    'docker': ['docker', 'containerization'],
    'linux': ['linux', 'unix'],
    'networking': ['networking', 'computer networking', 'tcp/ip'],
    'html': ['html', 'html5'],
    'css': ['css', 'css3'],
}

_SYNONYM_TO_CANONICAL = {}
for canonical, variants in SKILL_CANONICAL.items():
    for variant in variants:
        _SYNONYM_TO_CANONICAL[variant] = canonical

# Cities: lower-case raw -> canonical English spelling used in the UI.
CITY_CANONICAL = {
    'bangalore': 'Bengaluru',
    'bengaluru': 'Bengaluru',
    'bombay': 'Mumbai',
    'mumbai': 'Mumbai',
    'new delhi': 'Delhi',
    'delhi': 'Delhi',
    'delhi ncr': 'Delhi',
    'gurgaon': 'Gurugram',
    'gurugram': 'Gurugram',
    'noida': 'Noida',
    'pune': 'Pune',
    'hyderabad': 'Hyderabad',
    'chennai': 'Chennai',
    'kolkata': 'Kolkata',
    'calcutta': 'Kolkata',
    'ahmedabad': 'Ahmedabad',
    'jaipur': 'Jaipur',
    'lucknow': 'Lucknow',
    'indore': 'Indore',
    'bhopal': 'Bhopal',
    'patna': 'Patna',
    'coimbatore': 'Coimbatore',
    'kochi': 'Kochi',
    'cochin': 'Kochi',
    'chandigarh': 'Chandigarh',
    'nagpur': 'Nagpur',
    'surat': 'Surat',
    'visakhapatnam': 'Visakhapatnam',
    'vadodara': 'Vadodara',
}

STATE_CANONICAL = {
    'andhra pradesh': 'Andhra Pradesh',
    'arunachal pradesh': 'Arunachal Pradesh',
    'assam': 'Assam',
    'bihar': 'Bihar',
    'chhattisgarh': 'Chhattisgarh',
    'delhi': 'Delhi',
    'goa': 'Goa',
    'gujarat': 'Gujarat',
    'haryana': 'Haryana',
    'himachal pradesh': 'Himachal Pradesh',
    'jharkhand': 'Jharkhand',
    'karnataka': 'Karnataka',
    'kerala': 'Kerala',
    'madhya pradesh': 'Madhya Pradesh',
    'maharashtra': 'Maharashtra',
    'manipur': 'Manipur',
    'meghalaya': 'Meghalaya',
    'mizoram': 'Mizoram',
    'nagaland': 'Nagaland',
    'odisha': 'Odisha',
    'orissa': 'Odisha',
    'punjab': 'Punjab',
    'rajasthan': 'Rajasthan',
    'sikkim': 'Sikkim',
    'tamil nadu': 'Tamil Nadu',
    'telangana': 'Telangana',
    'tripura': 'Tripura',
    'uttar pradesh': 'Uttar Pradesh',
    'uttarakhand': 'Uttarakhand',
    'west bengal': 'West Bengal',
    'jammu and kashmir': 'Jammu and Kashmir',
    'ladakh': 'Ladakh',
    'puducherry': 'Puducherry',
    'chandigarh': 'Chandigarh',
}

_WHITESPACE = re.compile(r'\s+')


def _clean(text):
    value = str(text or '').lower()
    value = re.sub(r'[^a-z0-9+/.\- ]', ' ', value)
    return _WHITESPACE.sub(' ', value).strip()


def canonical_skill(raw_skill):
    """Map one skill string to its canonical name, or return the cleaned text."""
    cleaned = _clean(raw_skill)
    if not cleaned:
        return ''
    if cleaned in _SYNONYM_TO_CANONICAL:
        return _SYNONYM_TO_CANONICAL[cleaned]
    for variant, canonical in _SYNONYM_TO_CANONICAL.items():
        if variant in cleaned:
            return canonical
    return cleaned


def canonical_skill_list(raw_skills):
    """Split a comma-separated skill string into an ordered, de-duplicated canonical list."""
    seen = []
    for part in str(raw_skills or '').replace(';', ',').split(','):
        canonical = canonical_skill(part)
        if canonical and canonical not in seen:
            seen.append(canonical)
    return seen


def canonical_city(raw_city):
    cleaned = _clean(raw_city)
    if not cleaned:
        return ''
    if cleaned in CITY_CANONICAL:
        return CITY_CANONICAL[cleaned]
    # 'Bengaluru, Karnataka' style values
    head = cleaned.split(',')[0].strip()
    return CITY_CANONICAL.get(head, raw_city.strip())


def canonical_state(raw_state):
    cleaned = _clean(raw_state)
    if not cleaned:
        return ''
    if cleaned in STATE_CANONICAL:
        return STATE_CANONICAL[cleaned]
    head = cleaned.split(',')[0].strip()
    return STATE_CANONICAL.get(head, raw_state.strip())


def is_indian_location(city, state, country=''):
    """True when the record maps onto a known Indian city or state."""
    if str(country).strip().lower() in ('india', 'in', 'bharat'):
        return True
    cleaned_city = _clean(city)
    cleaned_state = _clean(state)
    return (cleaned_city in CITY_CANONICAL) or (cleaned_state in STATE_CANONICAL)
