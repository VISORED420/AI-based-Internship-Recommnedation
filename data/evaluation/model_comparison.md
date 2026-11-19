# Model comparison (synthetic benchmark)

Catalog: 115 records · Benchmark: 9 criteria-defined profiles · Labels: synthetic (no human relevance judgments)

| Model | P@5 | R@5 | NDCG@5 | Violations | Coverage | Diversity | Latency (ms) | ΔRAM (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_rule | 0.778 | 0.645 | 0.889 | 0 | 0.89 | 0.61 | 4.95 | 0.0 |
| baseline_hybrid | 0.889 | 0.715 | 1.000 | 0 | 1.00 | 0.65 | 5.14 | 0.0 |
| A0 title+sector+skills 1-2 | 0.889 | 0.715 | 1.000 | 0 | 1.00 | 0.70 | 1.51 | 0.2 |
| A1 skills only 1-1 | 0.889 | 0.715 | 1.000 | 0 | 1.00 | 0.67 | 1.48 | 0.0 |
| A2 title+skills 1-2 | 0.889 | 0.715 | 1.000 | 0 | 1.00 | 0.72 | 1.49 | 0.1 |
| A3 title+sector+skills 1-1 | 0.889 | 0.715 | 1.000 | 0 | 1.00 | 0.70 | 1.47 | 0.1 |
| embeddings:all-MiniLM-L6-v2 | 0.867 | 0.703 | 0.980 | 0 | 1.00 | 0.64 | 7.73 | 381.3 |

## External description-column diagnostics

{
  "documents": 5000,
  "unique_description_vectors": 24,
  "mean_cosine_between_different_titles": 0.234,
  "max_cosine_between_different_titles": 0.5025,
  "titles_compared": 24
}
