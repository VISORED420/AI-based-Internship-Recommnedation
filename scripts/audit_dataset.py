"""Audit an externally downloaded job dataset before any of it is used.

Writes data/audit/<name>_audit.json and a human-readable Markdown report.
Nothing here modifies the raw download.

Usage:
    python scripts/audit_dataset.py --csv data/external/raw/<file>.csv --name <source_name>
"""
import argparse
import hashlib
import html
import json
import os
import re
from collections import Counter

import pandas as pd

TEMPLATE_RE = re.compile(r'we are hiring a passionate .* to join our .* cluster\. ', re.I)
SHINGLE = 5
NEAR_DUP_THRESHOLD = 0.9


def strip_html(text):
    text = html.unescape(str(text or ''))
    return re.sub(r'<[^>]+>', ' ', text)


def normalise(text):
    return re.sub(r'\s+', ' ', strip_html(text).lower()).strip()


def shingles(text, size=SHINGLE):
    tokens = text.split()
    if len(tokens) < size:
        return {' '.join(tokens)} if tokens else set()
    return {' '.join(tokens[i:i + size]) for i in range(len(tokens) - size + 1)}


def near_duplicate_rate(descriptions, threshold=NEAR_DUP_THRESHOLD, max_rows=5000):
    """Fraction of rows whose description is >= threshold Jaccard-similar to an earlier row.

    Greedy first-match against previously seen shingle sets, capped at max_rows so the
    audit stays linear on large files.
    """
    seen = []
    duplicates = 0
    evaluated = 0
    for text in descriptions[:max_rows]:
        sh = shingles(text)
        evaluated += 1
        for other in seen:
            union = len(sh | other)
            if union and len(sh & other) / union >= threshold:
                duplicates += 1
                break
        else:
            seen.append(sh)
    return duplicates, evaluated, len(seen)


def column_profile(frame):
    profile = {}
    total = len(frame)
    for column in frame.columns:
        missing = int(frame[column].isna().sum())
        blanks = int((frame[column].astype(str).str.strip() == '').sum())
        distinct = int(frame[column].nunique(dropna=True))
        profile[column] = {
            'dtype': str(frame[column].dtype),
            'missing_pct': round(100 * (missing + blanks) / total, 2) if total else 0,
            'distinct': distinct,
        }
    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--description-column', default='job_description')
    args = parser.parse_args()

    with open(args.csv, 'rb') as handle:
        sha256 = hashlib.sha256(handle.read()).hexdigest()

    frame = pd.read_csv(args.csv, dtype=str).fillna('')
    total = len(frame)
    descriptions = [normalise(value) for value in frame[args.description_column]]

    exact_dupe_rows = int(frame.duplicated().sum())
    desc_counter = Counter(descriptions)
    unique_descriptions = len(desc_counter)
    duplicate_description_rate = round(100 * (total - unique_descriptions) / total, 2)

    templated = sum(1 for value in descriptions if TEMPLATE_RE.search(value))
    dup_near, evaluated, unique_near = near_duplicate_rate(descriptions)

    def distribution(column, top=15):
        if column not in frame:
            return {}
        return dict(Counter(frame[column].replace('', '<missing>')).most_common(top))

    skill_counter = Counter()
    for value in frame.get('skills_extracted', pd.Series(dtype=str)):
        for skill in str(value).split(','):
            skill = skill.strip().lower()
            if skill:
                skill_counter[skill] += 1

    city_counter = Counter(v.strip() for v in frame.get('location_city', []) if str(v).strip())
    state_counter = Counter(v.strip() for v in frame.get('location_state', []) if str(v).strip())

    report = {
        'source': args.name,
        'file': os.path.basename(args.csv),
        'sha256': sha256,
        'downloaded_at_utc': os.environ.get('AUDIT_DATE', ''),
        'total_records': total,
        'total_columns': len(frame.columns),
        'exact_duplicate_rows': exact_dupe_rows,
        'exact_duplicate_pct': round(100 * exact_dupe_rows / total, 2),
        'unique_job_descriptions': unique_descriptions,
        'duplicate_description_pct': duplicate_description_rate,
        'templated_description_pct': round(100 * templated / total, 2),
        'near_duplicate_pct_first_%d' % evaluated: round(100 * dup_near / evaluated, 2) if evaluated else 0,
        'near_duplicate_threshold': NEAR_DUP_THRESHOLD,
        'unique_descriptions_after_near_dedup_first_%d' % evaluated: unique_near,
        'columns': column_profile(frame),
        'industry_distribution': distribution('industry'),
        'education_distribution': distribution('education_required'),
        'employment_type_distribution': distribution('employment_type'),
        'work_mode_distribution': distribution('work_mode'),
        'country_distribution': distribution('location_country'),
        'top_cities': dict(city_counter.most_common(20)),
        'top_states': dict(state_counter.most_common(15)),
        'skills_present_pct': round(100 * (frame['skills_extracted'].str.strip() != '').mean(), 2)
        if 'skills_extracted' in frame else 0,
        'unique_skills': len(skill_counter),
        'top_skills': dict(skill_counter.most_common(25)),
        'unique_companies': int(frame['company_name'].nunique()) if 'company_name' in frame else 0,
        'unique_titles': int(frame['job_title'].nunique()) if 'job_title' in frame else 0,
        'sample_descriptions': [value for value in descriptions[:5]],
    }

    os.makedirs('data/audit', exist_ok=True)
    json_path = f'data/audit/{args.name}_audit.json'
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(json.dumps({key: report[key] for key in (
        'total_records', 'exact_duplicate_pct', 'unique_job_descriptions',
        'duplicate_description_pct', 'templated_description_pct',
        'unique_companies', 'unique_titles', 'unique_skills', 'skills_present_pct',
    )}, indent=2))
    print('wrote', json_path)


if __name__ == '__main__':
    main()
