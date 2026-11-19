"""Parameter tuning for reranker weights and answerability-gate thresholds.

Phases: cache (retrieval), matrices (numpy features), search (Optuna),
landscape (analytic gate sweep), finalize (recommendation export).
"""