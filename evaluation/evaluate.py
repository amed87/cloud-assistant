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

# Module names for python -m invocation
QUALITY_MODULE = "evaluation.evaluate_rag"
LATENCY_MODULE = "evaluation.request_looper"
KPI_MODULE = "evaluation.calculate_kpis"

def ensure_output_dir():
    EVAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {EVAL_OUTPUT}")

def run_module(module: str, args: list[str] | None = None) -> bool:
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
    print("\nMODE: QUALITY EVALUATION")
    print("-" * 40)
    
    ensure_output_dir()
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = EVAL_OUTPUT / f"results_{timestamp}.csv"
    
    success = run_module(QUALITY_MODULE, ["--output", str(output_file)])
    
    if success:
        print(f"Results saved to: {output_file.name}")
        
        # KPIs berechnen
        print("\nCalculating KPIs...")
        kpis_output = EVAL_OUTPUT / f"kpis_{timestamp}.json"
        run_module(KPI_MODULE, ["--dir", str(EVAL_OUTPUT), "--output", str(kpis_output)])
    
    return success

async def run_latency_only():
    print("\nMODE: LATENCY & RESOURCES")
    print("-" * 40)
    
    ensure_output_dir()
    
    success = run_module(LATENCY_MODULE)
    
    if success:
        load_files = sorted(EVAL_OUTPUT.glob('load-test-*.csv'))
        res_files = sorted(EVAL_OUTPUT.glob('*-resources.csv'))
        if load_files:
            print(f"Load test: {load_files[-1].name}")
        if res_files:
            print(f"Resources: {res_files[-1].name}")
        
        # KPIs berechnen
        print("\nCalculating KPIs...")
        kpis_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        kpis_output = EVAL_OUTPUT / f"kpis_{kpis_timestamp}.json"
        run_module(KPI_MODULE, ["--dir", str(EVAL_OUTPUT), "--output", str(kpis_output)])
    
    return success

async def run_kpis_only():
    print("\nMODE: KPI CALCULATION")
    print("-" * 40)
    
    if not EVAL_OUTPUT.exists():
        print(f"Directory not found: {EVAL_OUTPUT}")
        return False
    
    results_files = list(EVAL_OUTPUT.glob('results_*.csv'))
    load_files = list(EVAL_OUTPUT.glob('load-test-*.csv'))
    
    print(f"Found: {len(results_files)} results file(s), {len(load_files)} load file(s)")
    
    if not results_files and not load_files:
        print("No evaluation files found!")
        return False
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = EVAL_OUTPUT / f"kpis_{timestamp}.json"
    
    success = run_module(KPI_MODULE, ["--dir", str(EVAL_OUTPUT), "--output", str(output_file)])
    
    if success:
        print(f"KPIs saved to: {output_file.name}")
    
    return success

async def run_full_pipeline():
    print("\nMODE: FULL PIPELINE")
    print("=" * 60)
    
    ensure_output_dir()
    
    all_success = True
    
    print("\n[1/3] Quality evaluation...")
    success1 = run_module(
        QUALITY_MODULE,
        ["--output", str(EVAL_OUTPUT / f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")]
    )
    all_success = all_success and success1
    
    print("\n[2/3] Latency & resources...")
    success2 = run_module(LATENCY_MODULE)
    all_success = all_success and success2
    
    print("\n[3/3] Calculating KPIs...")
    kpis_output = EVAL_OUTPUT / f"kpis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    success3 = run_module(KPI_MODULE, ["--dir", str(EVAL_OUTPUT), "--output", str(kpis_output)])
    all_success = all_success and success3
    
    print("\n" + "=" * 60)
    if all_success:
        print("PIPELINE COMPLETED")
        print(f"\nGenerated files:")
        for f in sorted(EVAL_OUTPUT.glob('*.csv')):
            print(f"   - {f.name}")
        for f in sorted(EVAL_OUTPUT.glob('*.json')):
            print(f"   - {f.name}")
    else:
        print("PIPELINE COMPLETED WITH ERRORS")
    
    return all_success

def main():
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