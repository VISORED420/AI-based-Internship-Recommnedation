"""Offline evaluation of the rule-based, ML and hybrid recommenders.

Usage (from the repository root):

    python evaluation/evaluate.py

Labels come from evaluation/profiles.json criteria, computed directly from the CSV.
They are never taken from a recommender's output.

Metrics
    Precision@5 = relevant records in the top 5 / 5
    Recall@5    = relevant records in the top 5 / all relevant records for that profile
    Eligibility violations = returned records whose required education level exceeds the
                             candidate's level (must be 0 in every mode)
"""
import json
import math
import os
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import app  # noqa: E402

TOP_K = 5
REPEATS = 20

WEIGHT_VARIANTS = [
    ('hybrid 0.6/0.4 (default)', 0.6),
    ('hybrid 0.5/0.5', 0.5),
    ('hybrid 0.7/0.3', 0.7),
    ('hybrid 0.4/0.6', 0.4),
]


def normalise(text):
    return ' '.join(str(text or '').lower().split())


def row_skill_tokens(required_skills):
    """Split the CSV skill string into normalised skill phrases."""
    return [normalise(part) for part in str(required_skills).split(',') if part.strip()]


def skill_matches(token, skill_phrase):
    """A criterion token matches when it equals the skill or appears as a whole word in it."""
    if token == skill_phrase:
        return True
    return token in skill_phrase.split()


def relevant_ids(profile, rows):
    criteria = profile['relevance']
    max_level = criteria['max_level']
    tokens = [normalise(skill) for skill in criteria['any_skills']]
    ids = []
    for row in rows:
        if app.parse_education_level(row['required_education']) > max_level:
            continue
        skills = row_skill_tokens(row['required_skills'])
        if any(skill_matches(token, skill) for token in tokens for skill in skills):
            ids.append(row['internship_id'])
    return ids


def evaluate_mode(profile, mode, ml_weight, rows):
    started = time.perf_counter()
    results = app.rank_internships(
        qualification=profile['qualification'],
        skills=profile['skills'],
        sector=profile['sector_interested'],
        location=profile['location_interested'],
        mode=mode,
        ml_weight=ml_weight,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    labels = set(relevant_ids(profile, rows))
    returned = [rec['internship_id'] for rec in results]
    hits = [internship_id for internship_id in returned if internship_id in labels]

    candidate_level = app.parse_education_level(profile['qualification'])
    violations = [rec for rec in results
                  if app.parse_education_level(rec['required_education']) > candidate_level]

    dcg = sum(1.0 / math.log2(position + 2)
              for position, internship_id in enumerate(returned) if internship_id in labels)
    ideal_hits = min(len(labels), TOP_K)
    idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_hits))
    sectors = {rec['sector'] for rec in results}

    return {
        'precision': len(hits) / TOP_K,
        'recall': len(hits) / len(labels) if labels else 0.0,
        'ndcg': (dcg / idcg) if idcg else 0.0,
        'diversity': (len(sectors) / len(returned)) if returned else 0.0,
        'hits': hits,
        'returned': returned,
        'labels': sorted(labels),
        'violations': len(violations),
        'latency_ms': elapsed_ms,
        'results': results,
    }


def latency(mode, ml_weight, profiles):
    timings = []
    for _ in range(REPEATS):
        for profile in profiles:
            started = time.perf_counter()
            app.rank_internships(
                qualification=profile['qualification'],
                skills=profile['skills'],
                sector=profile['sector_interested'],
                location=profile['location_interested'],
                mode=mode,
                ml_weight=ml_weight,
            )
            timings.append((time.perf_counter() - started) * 1000)
    return statistics.median(timings)


def main():
    with open('evaluation/profiles.json', encoding='utf-8') as handle:
        profiles = json.load(handle)['profiles']
    rows = app.internships_df.to_dict('records')

    print(f'Model: {app.ml_model.document_count} documents, '
          f'{app.ml_model.vocabulary_size} TF-IDF features (unigrams + bigrams)')
    print(f'{len(profiles)} profiles, Precision@{TOP_K} / Recall@{TOP_K}\n')

    runs = [('rule', 0.6), ('ml', 0.6)] + [(label, weight) for label, weight in WEIGHT_VARIANTS]

    summary = {}
    for label, weight in runs:
        mode = 'hybrid' if label.startswith('hybrid') else label
        per_profile = []
        print(f'=== {label} ===')
        print(f"{'profile':26s} {'n':>2s} {'P@5':>5s} {'R@5':>5s} {'viol':>5s}  hits")
        for profile in profiles:
            result = evaluate_mode(profile, mode, weight, rows)
            per_profile.append(result)
            print(f"{profile['id']:26s} {len(result['returned']):>2d} "
                  f"{result['precision']:>5.2f} {result['recall']:>5.2f} "
                  f"{result['violations']:>5d}  {result['hits']}")
        mean_p = statistics.mean(r['precision'] for r in per_profile)
        mean_r = statistics.mean(r['recall'] for r in per_profile)
        mean_ndcg = statistics.mean(r['ndcg'] for r in per_profile)
        coverage = sum(1 for r in per_profile if r['returned']) / len(per_profile)
        diversity = statistics.mean(r['diversity'] for r in per_profile)
        violations = sum(r['violations'] for r in per_profile)
        summary[label] = (mean_p, mean_r, mean_ndcg, coverage, diversity, violations)
        print(f"  mean P@5={mean_p:.3f}  mean R@5={mean_r:.3f}  mean NDCG@5={mean_ndcg:.3f}  "
              f"coverage={coverage:.2f}  diversity={diversity:.2f}  "
              f"eligibility violations={violations}\n")

    print('=== summary ===')
    print(f"{'approach':26s} {'P@5':>6s} {'R@5':>6s} {'NDCG@5':>7s} {'cover':>6s} "
          f"{'divers':>7s} {'viol':>5s}")
    for label, (mean_p, mean_r, mean_ndcg, coverage, diversity, violations) in summary.items():
        print(f'{label:26s} {mean_p:>6.3f} {mean_r:>6.3f} {mean_ndcg:>7.3f} '
              f'{coverage:>6.2f} {diversity:>7.2f} {violations:>5d}')

    print('\n=== latency (median over '
          f'{REPEATS} x {len(profiles)} calls, in-process) ===')
    for label, weight in [('rule', 0.6), ('ml', 0.6), ('hybrid 0.6/0.4 (default)', 0.6)]:
        mode = 'hybrid' if label.startswith('hybrid') else label
        print(f'  {label:26s} {latency(mode, weight, profiles):7.2f} ms')


if __name__ == '__main__':
    main()
