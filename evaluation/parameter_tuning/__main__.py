"""Entry point: python -m parameter_tuning --trials 2000"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from app.config.content_settings import content_config
from evaluation.evaluate_rag import GOLDEN_CSV, load_cases

from .calibration import calibrate
from .cache import build_cache
from .finalize import finalize
from .matrices import build_matrices
from .search import run_search
from .service import create_rag_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Reranker + gate parameter tuning")
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--output", "-o", type=Path, default=Path("recommended_settings.json"))
    parser.add_argument("--study-db", type=str, default=None,
                        help="Optuna storage URL, e.g. sqlite:///tuning.db (omit: in-memory)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rows = load_cases(GOLDEN_CSV)
    prompt_names = list(content_config.decision_model_prompts)

    service = create_rag_service()
    cases = asyncio.run(build_cache(rows, service))
    matrices = build_matrices(cases)

    cal = calibrate(cases, matrices, prompt_names)
    if not cal["passed"]:
        print("\nCALIBRATION FAILED - matrix scores do not reproduce the "
              "live reranker. Fix before tuning:")
        for check in cal["checks"]:
            if not (check["scores_match"] and check["top_match"]):
                print(f"  [{check['question'][:50]}] max_diff="
                      f"{check['max_abs_diff']:.6f} "
                      f"predicted={check['predicted']} live={check['live_top']}")
        return
    print("Calibration passed: matrix scores match the live reranker.")

    study_name = f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = run_search(cases, matrices, prompt_names, n_trials=args.trials,
                       storage=args.study_db, study_name=study_name, seed=args.seed)

    report = finalize(cases, matrices, prompt_names, study)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nBest F0.5: {report['best_f05']:.3f} (trial {report['best_trial']})")
    for k, v in report["recommendation"].items():
        print(f"  {k}: {v}")
    if report["boundary_warnings"]:
        print("\nBoundary warnings:")
        for w in report["boundary_warnings"]:
            print(f"  - {w}")
    if not report["gate_validation"]["matches_analytic"]:
        print("\nWARNING: real gate disagrees with analytic counts — investigate!")
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()