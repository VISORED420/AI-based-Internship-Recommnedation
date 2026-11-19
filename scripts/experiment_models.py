"""Compare the production recommender against experimental models.

Usage (from the repository root):
    python scripts/experiment_models.py

Writes data/evaluation/model_comparison.json and .md.
Every model gets the same treatment: mandatory education eligibility filter first,
then ranking, then the same metric code.
"""
import json
import math
import os
import statistics
import sys
import time

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import app  # noqa: E402
from evaluation.evaluate import relevant_ids  # same criteria labels, one implementation
from pipeline.models import (  # noqa: E402
    SentenceEmbeddingVariant, TfidfOnDescriptions, TfidfVariant,
)

TOP_K = 5
REPEATS = 20
with open('evaluation/profiles.json', encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)['profiles']
ROWS = app.internships_df.to_dict('records')


def ndcg_at_k(ranked_ids, relevant, k=TOP_K):
    dcg = 0.0
    for position, internship_id in enumerate(ranked_ids[:k]):
        if internship_id in relevant:
            dcg += 1.0 / math.log2(position + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def candidate_level(profile):
    return app.parse_education_level(profile['qualification'])


def eligible_positions(profile):
    level = candidate_level(profile)
    return [position for position, row in enumerate(ROWS)
            if app.parse_education_level(row['required_education']) <= level]


def rank_with_scores(profile, scores):
    positions = eligible_positions(profile)
    ordered = sorted(positions, key=lambda position: scores[position], reverse=True)
    picked = [position for position in ordered[:TOP_K] if scores[position] > 0]
    return [ROWS[position]['internship_id'] for position in picked]


def metrics_for_ranker(ranker):
    per_profile = {}
    latencies = []
    violations = 0
    results = {}

    for profile in PROFILES:
        started = time.perf_counter()
        ranked = ranker(profile)
        latencies.append((time.perf_counter() - started) * 1000)
        results[profile['id']] = ranked

        labels = set(relevant_ids(profile, ROWS))
        hits = [internship_id for internship_id in ranked if internship_id in labels]
        level = candidate_level(profile)
        violations += sum(1 for internship_id in ranked
                          if app.parse_education_level(
                              next(row['required_education'] for row in ROWS
                                   if row['internship_id'] == internship_id)) > level)

        sectors = {next(row['sector'] for row in ROWS if row['internship_id'] == internship_id)
                   for internship_id in ranked}
        per_profile[profile['id']] = {
            'precision': len(hits) / TOP_K,
            'recall': len(hits) / len(labels) if labels else 0.0,
            'ndcg': ndcg_at_k(ranked, labels),
            'returned': len(ranked),
            'diversity': (len(sectors) / len(ranked)) if ranked else 0.0,
            'qualification': profile['qualification'],
        }

    for _ in range(REPEATS):
        for profile in PROFILES:
            started = time.perf_counter()
            ranker(profile)
            latencies.append((time.perf_counter() - started) * 1000)

    summary = {
        'precision_at_5': round(statistics.mean(v['precision'] for v in per_profile.values()), 4),
        'recall_at_5': round(statistics.mean(v['recall'] for v in per_profile.values()), 4),
        'ndcg_at_5': round(statistics.mean(v['ndcg'] for v in per_profile.values()), 4),
        'eligibility_violations': violations,
        'coverage_profiles_with_results': round(
            sum(1 for v in per_profile.values() if v['returned'] > 0) / len(PROFILES), 4),
        'mean_diversity': round(statistics.mean(v['diversity'] for v in per_profile.values()), 4),
        'median_latency_ms': round(statistics.median(latencies), 2),
        'per_profile': per_profile,
        'results': results,
    }
    return summary


def coverage_by_qualification(all_summaries):
    buckets = {}
    for name, summary in all_summaries.items():
        for profile_id, values in summary['per_profile'].items():
            bucket = buckets.setdefault(values['qualification'],
                                        {'profiles': 0, 'with_results': 0, 'precision_sum': 0.0})
            bucket['profiles'] += 1
            bucket['with_results'] += 1 if values['returned'] else 0
            bucket['precision_sum'] += values['precision']
    return {qualification: {
        'profiles': data['profiles'],
        'profiles_with_results': data['with_results'],
        'mean_precision_at_5': round(data['precision_sum'] / data['profiles'], 4),
    } for qualification, data in buckets.items()}


def main():
    try:
        import psutil
        process = psutil.Process()
        rss = lambda: process.memory_info().rss / (1024 ** 2)
    except ImportError:  # pragma: no cover - optional dependency
        rss = lambda: 0.0

    summaries = {}
    memory = {}

    def add(name, ranker, before, after):
        summaries[name] = metrics_for_ranker(ranker)
        memory[name] = round(after - before, 1)
        print(f"{name:28s} P@5={summaries[name]['precision_at_5']:.3f} "
              f"R@5={summaries[name]['recall_at_5']:.3f} "
              f"NDCG@5={summaries[name]['ndcg_at_5']:.3f} "
              f"lat={summaries[name]['median_latency_ms']:.1f}ms +{memory[name]}MB")

    add('baseline_rule', lambda profile: [row['internship_id'] for row in app.rank_internships(
        profile['qualification'], profile['skills'], profile['sector_interested'],
        profile['location_interested'], mode='rule')], rss(), rss())

    before = rss()
    add('baseline_hybrid', lambda profile: [row['internship_id'] for row in app.rank_internships(
        profile['qualification'], profile['skills'], profile['sector_interested'],
        profile['location_interested'], mode='hybrid')], before, before)

    variants = [
        TfidfVariant('A0 title+sector+skills 1-2', ('title', 'sector', 'required_skills')),
        TfidfVariant('A1 skills only 1-1', ('required_skills',), ngram_range=(1, 1)),
        TfidfVariant('A2 title+skills 1-2', ('title', 'required_skills')),
        TfidfVariant('A3 title+sector+skills 1-1', ('title', 'sector', 'required_skills'),
                     ngram_range=(1, 1)),
    ]
    for variant in variants:
        before = rss()
        variant.fit(app.internships_df)
        add(variant.name, (lambda v: (lambda profile: rank_with_scores(
            profile, v.scores(profile['skills']))))(variant), before, rss())

    try:
        before = rss()
        embedding_model = SentenceEmbeddingVariant()
        embedding_model.fit(app.internships_df)
        add(embedding_model.name, (lambda m: (lambda profile: rank_with_scores(
            profile, m.scores(profile['skills']))))(embedding_model), before, rss())
    except Exception as exc:  # dependency or download failure
        print('sentence-transformer unavailable:', type(exc).__name__, str(exc)[:120])
        summaries[embedding_model_name := 'embeddings:unavailable'] = None

    external_path = 'data/processed/jobs_sample.csv'
    description_diagnostics = None
    if os.path.exists(external_path):
        external = pd.read_csv(external_path, dtype=str).fillna('')
        try:
            model = TfidfOnDescriptions().fit(external)
            description_diagnostics = model.diagnostics()
        except ValueError as exc:
            description_diagnostics = {'error': str(exc)}

    payload = {
        'evaluated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'benchmark': 'evaluation/profiles.json (9 profiles, criteria-derived labels, '
                     'synthetic catalog of 115 records)',
        'label_type': 'synthetic (criteria computed from dataset fields; no human relevance judgments)',
        'records_in_catalog': len(ROWS),
        'summaries': {name: value for name, value in summaries.items() if value},
        'memory_delta_mb': memory,
        'coverage_by_qualification': coverage_by_qualification(
            {name: value for name, value in summaries.items() if value}),
        'external_description_diagnostics': description_diagnostics,
    }

    os.makedirs('data/evaluation', exist_ok=True)
    with open('data/evaluation/model_comparison.json', 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    lines = [
        '# Model comparison (synthetic benchmark)',
        '',
        f"Catalog: {len(ROWS)} records · Benchmark: 9 criteria-defined profiles · "
        f"Labels: synthetic (no human relevance judgments)",
        '',
        '| Model | P@5 | R@5 | NDCG@5 | Violations | Coverage | Diversity | Latency (ms) | ΔRAM (MB) |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for name, summary in payload['summaries'].items():
        lines.append(
            f"| {name} | {summary['precision_at_5']:.3f} | {summary['recall_at_5']:.3f} | "
            f"{summary['ndcg_at_5']:.3f} | {summary['eligibility_violations']} | "
            f"{summary['coverage_profiles_with_results']:.2f} | {summary['mean_diversity']:.2f} | "
            f"{summary['median_latency_ms']:.2f} | {memory.get(name, 0)} |")
    if description_diagnostics:
        lines += ['', '## External description-column diagnostics', '',
                  json.dumps(description_diagnostics, indent=2)]
    with open('data/evaluation/model_comparison.md', 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')

    print('\nwrote data/evaluation/model_comparison.{json,md}')
    print(json.dumps(description_diagnostics, indent=2))


if __name__ == '__main__':
    main()
