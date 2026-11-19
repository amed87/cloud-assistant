"""
RAG Chatbot Evaluation Runner

MODES:
    full      : Vollständige Pipeline (Quality + Latency + KPIs)
    quality   : Nur Quality Evaluation + KPIs
    latency   : Nur Latency Tests + KPIs
    kpis      : Nur KPI-Berechnung aus vorhandenen Dateien

USAGE:
    python -m evaluation.evaluate --mode full
    python -m evaluation.evaluate --mode quality
    python -m evaluation.evaluate --mode latency
    python -m evaluation.evaluate --mode kpis
"""

import argparse
import asyncio
import sys
import subprocess
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPT_DIR.parent
EVAL_OUTPUT = PROJECT_ROOT / "eval" / "output"
QUALITY_OUTPUT = EVAL_OUTPUT / "Qualitätsevaluation"
LOADTEST_OUTPUT = EVAL_OUTPUT / "Loadtest"
KPI_OUTPUT = EVAL_OUTPUT / "KPIs"

# Module names for python -m invocation
QUALITY_MODULE = "evaluation.evaluate_rag"
LATENCY_MODULE = "evaluation.request_looper"
KPI_MODULE = "evaluation.calculate_kpis"

def ensure_output_dir():
    """Create the split output directories for evaluation artifacts."""
    for directory in (QUALITY_OUTPUT, LOADTEST_OUTPUT, KPI_OUTPUT):
        directory.mkdir(parents=True, exist_ok=True)
    print(f"Output directories: {EVAL_OUTPUT}/{{Qualitätsevaluation,Loadtest,KPIs}}")

def run_module(module: str, args: list[str] | None = None) -> bool:
    """Run an evaluation module in a child Python process and report success."""
    cmd = [sys.executable, "-m", module]
    if args:
        cmd.extend(args)
    
    print(f"\n{'='*60}")
    print(f"RUNNING: {' '.join(cmd)}")
    print('='*60 + '\n')
    
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    
    if result.returncode == 0:
        print(f"{module} completed successfully\n")
        return True
    else:
        print(f"{module} failed with exit code {result.returncode}\n")
        return False

async def run_quality_only():
    """Run golden-dataset evaluation and calculate KPIs for its output."""
    print("\nMODE: QUALITY EVALUATION")
    print("-" * 40)
    
    ensure_output_dir()
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = QUALITY_OUTPUT / f"results_{timestamp}.csv"

    success = run_module(QUALITY_MODULE, ["--output", str(output_file)])

    if success:
        print(f"Results saved to: {output_file.name}")

        # KPIs berechnen — Scopes: Qualität aus Qualitätsevaluation, Rest aus Loadtest
        print("\nCalculating KPIs...")
        kpis_output = KPI_OUTPUT / f"kpis_{timestamp}.json"
        run_module(KPI_MODULE, ["--dir", str(QUALITY_OUTPUT), str(LOADTEST_OUTPUT),
                                "--output", str(kpis_output)])

    return success

async def run_latency_only():
    """Run the load scenario and calculate KPIs from generated measurements."""
    print("\nMODE: LATENCY & RESOURCES")
    print("-" * 40)
    
    ensure_output_dir()
    
    success = run_module(LATENCY_MODULE)
    
    if success:
        load_files = sorted(LOADTEST_OUTPUT.glob('load-test-*.csv'))
        res_files = sorted(LOADTEST_OUTPUT.glob('*-resources.csv'))
        if load_files:
            print(f"Load test: {load_files[-1].name}")
        if res_files:
            print(f"Resources: {res_files[-1].name}")

        # KPIs berechnen — Scopes: Qualität aus Qualitätsevaluation, Rest aus Loadtest
        print("\nCalculating KPIs...")
        kpis_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        kpis_output = KPI_OUTPUT / f"kpis_{kpis_timestamp}.json"
        run_module(KPI_MODULE, ["--dir", str(QUALITY_OUTPUT), str(LOADTEST_OUTPUT),
                                "--output", str(kpis_output)])
    
    return success

async def run_kpis_only():
    """Calculate KPIs from existing evaluation output without rerunning tests."""
    print("\nMODE: KPI CALCULATION")
    print("-" * 40)
    
    if not EVAL_OUTPUT.exists():
        print(f"Directory not found: {EVAL_OUTPUT}")
        return False
    
    results_files = list(QUALITY_OUTPUT.glob('results_*.csv'))
    load_files = list(LOADTEST_OUTPUT.glob('load-test-*.csv'))

    print(f"Found: {len(results_files)} results file(s), {len(load_files)} load file(s)")

    if not results_files and not load_files:
        print("No evaluation files found!")
        return False

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = KPI_OUTPUT / f"kpis_{timestamp}.json"

    success = run_module(KPI_MODULE, ["--dir", str(QUALITY_OUTPUT), str(LOADTEST_OUTPUT),
                                      "--output", str(output_file)])
    
    if success:
        print(f"KPIs saved to: {output_file.name}")
    
    return success

async def run_full_pipeline():
    """Run quality evaluation, load testing, and KPI calculation in sequence."""
    print("\nMODE: FULL PIPELINE")
    print("=" * 60)
    
    ensure_output_dir()
    
    all_success = True
    
    print("\n[1/3] Quality evaluation...")
    success1 = run_module(
        QUALITY_MODULE,
        ["--output", str(QUALITY_OUTPUT / f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")]
    )
    all_success = all_success and success1

    print("\n[2/3] Latency & resources...")
    success2 = run_module(LATENCY_MODULE)
    all_success = all_success and success2

    print("\n[3/3] Calculating KPIs...")
    kpis_output = KPI_OUTPUT / f"kpis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    success3 = run_module(KPI_MODULE, ["--dir", str(QUALITY_OUTPUT), str(LOADTEST_OUTPUT),
                                       "--output", str(kpis_output)])
    all_success = all_success and success3

    print("\n" + "=" * 60)
    if all_success:
        print("PIPELINE COMPLETED")
        print(f"\nGenerated files:")
        for f in sorted(QUALITY_OUTPUT.glob('*')) + sorted(LOADTEST_OUTPUT.glob('*')) + sorted(KPI_OUTPUT.glob('*')):
            print(f"   - {f.relative_to(EVAL_OUTPUT)}")
    else:
        print("PIPELINE COMPLETED WITH ERRORS")
    
    return all_success

def main():
    """Parse the selected evaluation mode and return an appropriate exit code."""
    parser = argparse.ArgumentParser(
        description='RAG Chatbot Evaluation Runner'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['full', 'quality', 'latency', 'kpis'],
        default='full',
        help='Evaluation mode (default: full)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        success = asyncio.run(run_full_pipeline())
    elif args.mode == 'quality':
        success = asyncio.run(run_quality_only())
    elif args.mode == 'latency':
        success = asyncio.run(run_latency_only())
    elif args.mode == 'kpis':
        success = asyncio.run(run_kpis_only())
    else:
        parser.print_help()
        success = False
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()