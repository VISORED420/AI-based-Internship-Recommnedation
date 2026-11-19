"""Preprocess an audited external file into the standard schema.

Never invents fields: anything the source does not provide is left empty and marked in
`data_status` / `education_known` instead of being guessed.

Usage:
    python scripts/preprocess.py \
        --csv data/external/raw/indian_tech_career_intelligence_2026.csv \
        --source "Uzaib52/indian-tech-career-intelligence-2026" \
        --source-url "https://huggingface.co/datasets/Uzaib52/indian-tech-career-intelligence-2026" \
        --status synthetic_templated \
        --sample 5000 \
        --out data/processed/jobs_sample.csv
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.qualifications import normalize_qualification, level_of, track_of  # noqa: E402
from pipeline.skills_normalizer import (  # noqa: E402
    canonical_city, canonical_skill_list, canonical_state, is_indian_location,
)

OUTPUT_COLUMNS = [
    'job_id', 'title', 'company_name', 'description', 'required_skills',
    'required_education', 'education_level', 'education_track', 'education_known',
    'sector', 'location', 'location_raw', 'state', 'stipend', 'stipend_period',
    'job_type', 'source', 'source_url', 'data_status', 'description_quality',
    'is_currently_open', 'raw_company_name', 'raw_required_education', 'raw_skills',
]


def build_frame(frame, source, source_url, status):
    rows = []
    for index, record in enumerate(frame.to_dict('records'), start=1):
        employment = str(record.get('employment_type', '')).strip()
        job_type = 'Internship' if employment.lower().startswith('intern') else (employment or 'Unknown')

        if job_type == 'Internship' and str(record.get('stipend_amount', '')).strip() not in ('', 'nan'):
            stipend, period = record.get('stipend_amount'), 'month'
        elif str(record.get('salary_minimum', '')).strip() not in ('', 'nan'):
            stipend, period = record.get('salary_minimum'), 'year'
        else:
            stipend, period = '', ''

        raw_education = record.get('education_required', '')
        qualification = normalize_qualification(raw_education)
        skills = canonical_skill_list(record.get('skills_extracted', ''))
        city = canonical_city(record.get('location_city', ''))
        state = canonical_state(record.get('location_state', ''))

        rows.append({
            'job_id': f'ITC26-{index:06d}',
            'title': str(record.get('job_title', '')).strip(),
            'company_name': str(record.get('company_name', '')).strip(),
            'description': str(record.get('job_description', '')).strip(),
            'required_skills': ', '.join(skills),
            'required_education': qualification or '',
            'education_level': level_of(qualification) if qualification else '',
            'education_track': track_of(qualification) if qualification else '',
            'education_known': bool(qualification),
            'sector': str(record.get('industry', '')).strip(),
            'location': city,
            'location_raw': f"{record.get('location_city', '')}, {record.get('location_state', '')}".strip(', '),
            'state': state,
            'stipend': stipend,
            'stipend_period': period,
            'job_type': job_type,
            'source': source,
            'source_url': source_url,
            'data_status': status,
            'description_quality': 'templated',
            'is_currently_open': False,
            'raw_company_name': str(record.get('company_name', '')).strip(),
            'raw_required_education': str(raw_education).strip(),
            'raw_skills': str(record.get('skills_extracted', '')).strip(),
        })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--source-url', required=True)
    parser.add_argument('--status', default='synthetic_templated')
    parser.add_argument('--sample', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    raw = pd.read_csv(args.csv, dtype=str).fillna('')
    sample = raw.sample(n=min(args.sample, len(raw)), random_state=args.seed).sort_index()
    processed = build_frame(sample, args.source, args.source_url, args.status)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    processed.to_csv(args.out, index=False, encoding='utf-8')

    stats = {
        'input_rows': int(len(raw)),
        'sampled_rows': int(len(processed)),
        'education_known_pct': round(100 * processed['education_known'].mean(), 2),
        'education_distribution': processed['required_education'].value_counts().to_dict(),
        'job_type_distribution': processed['job_type'].value_counts().to_dict(),
        'indian_location_pct': round(100 * processed.apply(
            lambda row: is_indian_location(row['location'], row['state'], 'India'), axis=1).mean(), 2),
        'unique_cities': int(processed['location'].nunique()),
        'unique_skills': len({skill for value in processed['required_skills']
                              for skill in str(value).split(',') if skill.strip()}),
    }
    stats_path = os.path.splitext(args.out)[0] + '_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2))
    print('wrote', args.out, 'and', stats_path)


if __name__ == '__main__':
    main()
