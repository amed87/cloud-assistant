"""Entry point: python -m parameter_tuning --trials 2000"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from app.config.content_settings import content_config
from evaluation.evaluate_rag import GOLDEN_CSV, load_cases

from .calibration import calibrate, resolve_base_weights
from .cache import build_cache, save_cache, load_cache
from .finalize import finalize
from .matrices import build_matrices
from .search import run_search
from .service import create_rag_service
from .tuning_config import load_tuning_config


# Artefakte liegen unter <repo>/eval/output/Tuning (unabhängig vom cwd).
# parents[3], weil dieses Modul eine Ebene tiefer liegt als evaluate_rag.py.
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "eval" / "output" / "Tuning"
REPO_DIR = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reranker + gate parameter tuning")
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--output", "-o", type=Path,
                        default=OUTPUT_DIR / "recommended_settings.json")
    parser.add_argument("--config", type=Path,
                        default=REPO_DIR / "tuning_config.yaml",
                        help="Tuning-Gruppen-Konfiguration (yaml); fehlt sie, "
                             "werden alle Gruppen getunt.")
    parser.add_argument("--study-db", type=str, default=None,
                        help="Optuna storage URL, e.g. sqlite:///tuning.db (omit: in-memory)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optuna sampler seed. Omit (default) to derive: "
                             "reused from the loaded cache for reproducibility, "
                             "or drawn fresh for a newly built cache so the "
                             "search never overfits one DM sample. An explicit "
                             "value always wins.")
    parser.add_argument("--cache", type=Path,
                        default=OUTPUT_DIR / "tuning_case_cache.json",
                        help="Persisted retrieval cache (json).")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Discard the persisted cache and re-retrieve all "
                             "golden cases live. Required after changes to the "
                             "FAQ database, features, or reranker code.")
    args = parser.parse_args()

    tune_config = load_tuning_config(args.config)
    print(f"Tuning groups: intent={tune_config.intent} keyword={tune_config.keyword} "
          f"decision_model={tune_config.decision_model} centroid={tune_config.centroid}")

    rows = load_cases(GOLDEN_CSV)
    prompt_names = list(content_config.decision_model_prompts)

    cases = None
    run_seed = args.seed
    if not args.refresh_cache:
        cases, cached_seed = load_cache(args.cache, rows, prompt_names)
        if cases is not None:
            print(f"Loaded persisted cache for {len(cases)} cases from "
                  f"{args.cache} (--refresh-cache to rebuild).")
            # Reproduzierbar auf derselben Matrix: Seed aus dem Cache —
            # außer der Aufrufer erzwingt explizit einen anderen.
            run_seed = args.seed if args.seed is not None else cached_seed
    if cases is None:
        # Frische Matrix = frischer Würfelwurf der DM-Werte: mit zufälligem
        # Seed suchen, damit kein einzelnes Würfelergebnis überoptimiert wird.
        if run_seed is None:
            import random
            run_seed = random.SystemRandom().randrange(2**31)
        service = create_rag_service()
        cases = asyncio.run(build_cache(rows, service))
        save_cache(cases, args.cache, sampler_seed=run_seed)
        print(f"Retrieval cache saved to {args.cache} — later runs reuse it "
              "until you pass --refresh-cache.")

    print(f"Sampler seed: {run_seed}" if run_seed is not None else "Sampler seed: None")
    matrices = build_matrices(cases)
    base = resolve_base_weights(cases, prompt_names)

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
                       storage=args.study_db, study_name=study_name, seed=run_seed,
                       config=tune_config, base=base)

    report = finalize(cases, matrices, prompt_names, study,
                      config=tune_config, base=base)
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
