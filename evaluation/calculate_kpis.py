"""
RAG Chatbot Evaluation KPI Generator

Reads evaluation output files and generates comprehensive KPIs for:
- Latency & Load (from load-test-*.csv)
- Answer Quality (from results_*.csv)
"""

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev


def load_csv(filepath: Path) -> list[dict]:
    """Load CSV file as list of dictionaries."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


# ============================================
# LATENCY & LOAD KPIS
# ============================================

def calculate_latency_kpis(load_file: Path) -> dict:
    """Calculate latency/load KPIs from load test results."""
    data = load_csv(load_file)
    
    if not data:
        return {}
    
    # Parse durations
    durations = [float(row['Duration (ms)']) for row in data 
                 if row.get('Duration (ms)')]
    
    # Filter successful requests (exclude errors/negative values)
    successful = [d for d in durations if d > 0]
    
    # Calculate percentiles
    sorted_durations = sorted(successful) if successful else []
    
    def percentile(p):
        idx = min(int(len(sorted_durations) * p / 100), len(sorted_durations) - 1)
        return sorted_durations[idx] if sorted_durations else 0
    
    # System-level info from elapsed time (if available)
    elapsed_times = [float(row['Elapsed Time (s)']) for row in data 
                     if row.get('Elapsed Time (s)')]
    total_elapsed = max(elapsed_times) if elapsed_times else 1.0
    
    return {
        'total_requests': len(durations),
        'successful_requests': len(successful),
        'avg_response_time_ms': mean(successful) if successful else 0,
        'median_response_time_ms': percentile(50),
        'p95_response_time_ms': percentile(95),
        'p99_response_time_ms': percentile(99),
        'min_response_time_ms': min(successful) if successful else 0,
        'max_response_time_ms': max(successful) if successful else 0,
        'throughput_requests_per_sec': len(data) / max(0.001, total_elapsed),
        # Resource stats moved to calculate_resource_kpis() - no 'Resource' column here
    }

def calculate_resource_kpis(resources_file: Path) -> dict:
    """Calculate per-resource KPIs from resources monitoring."""
    data = load_csv(resources_file)
    
    if not data:
        return {}
    
    resource_stats = defaultdict(lambda: {'cpu': [], 'memory': [], 'mem_pct': []})
    
    for row in data:
        resource = row.get('Resource', '')
        if not resource:
            continue
        if row.get('CPU (%)'):
            try:
                resource_stats[resource]['cpu'].append(float(row['CPU (%)']))
            except ValueError:
                pass
        if row.get('Memory (MB)'):
            try:
                resource_stats[resource]['memory'].append(float(row['Memory (MB)']))
            except ValueError:
                pass
        if row.get('Memory (%)'):
            try:
                resource_stats[resource]['mem_pct'].append(float(row['Memory (%)']))
            except ValueError:
                pass
    
    kpis = {}
    for resource, values in resource_stats.items():
        safe_name = resource.replace(':', '_').replace('.', '_').replace(' ', '_')
        kpis[f'{safe_name}_avg_cpu'] = mean(values['cpu']) if values['cpu'] else 0
        kpis[f'{safe_name}_peak_cpu'] = max(values['cpu']) if values['cpu'] else 0
        kpis[f'{safe_name}_avg_mem_mb'] = mean(values['memory']) if values['memory'] else 0
        kpis[f'{safe_name}_peak_mem_mb'] = max(values['memory']) if values['memory'] else 0
        kpis[f'{safe_name}_avg_mem_pct'] = mean(values['mem_pct']) if values['mem_pct'] else 0
    
    return kpis

# ============================================
# ANSWER QUALITY KPIS (KORRIGIERT)
# ============================================

def calculate_quality_kpis(results_file: Path) -> dict:
    """Calculate answer quality KPIs from evaluation results."""
    data = load_csv(results_file)
    
    if not data:
        return {}
    
    total = len(data)
    
    # Verdict counts
    verdicts = defaultdict(int)
    for row in data:
        verdicts[row['verdict']] += 1
    
    # Score statistics
    scores = [float(row['score']) for row in data if row.get('score')]
    margins = [float(row['margin']) for row in data 
               if row.get('margin') and row['margin'] != '-']
    
    # Category breakdown
    categories = defaultdict(lambda: {'total': 0, 'ok': 0})
    for row in data:
        cat = row['category']
        categories[cat]['total'] += 1
        if row['verdict'] == 'ok':
            categories[cat]['ok'] += 1
    
    # ========================================
    # GATE PERFORMANCE METRIKEN (korrigiert)
    # ========================================
    
    # Kategorisieren welche Frage-Arten "beantwortbar" vs "nicht-beantwortbar" sind
    answerable_categories = {
        'positiv', 'paraphrase', 'grenzfall_tippfehler', 'grenzfall_fragment',
        'grenzfall_kleinschreibung', 'grenzfall_stichwoerter', 'grenzfall_mehrdeutig',
        'grenzfall_lange_frage', 'fast_treffer'
    }
    
    unanswerable_categories = {
        'negativ', 'negativ_grenzfall_fremdsprache'
    }
    
    # Gate True Positives: Beantwortbare Fragen → Retrieval ausgelöst (ok)
    gate_tp = sum(1 for row in data if row['verdict'] == 'ok')
    
    # Gate False Positives: Nicht-beantwortbare Fragen → Retrieval ausgelöst
    gate_fp = sum(1 for row in data if 'false_positive' in row['verdict'])
    
    # Gate False Negatives: Beantwortbare Fragen → Retrieval BLOCKIERT (fallback)
    gate_fn = sum(1 for row in data 
                  if 'fallback' in row['verdict'] 
                  and row['category'] in answerable_categories)
    
    # Gate True Negatives: Nicht-beantwortbare Fragen → Retrieval korrekt blockiert
    gate_tn = sum(1 for row in data 
                  if 'fallback' in row['verdict'] 
                  and row['category'] in unanswerable_categories)
    
    # Gateway Metriken berechnen
    gate_precision = gate_tp / (gate_tp + gate_fp) if (gate_tp + gate_fp) > 0 else 0
    gate_recall = gate_tp / (gate_tp + gate_fn) if (gate_tp + gate_fn) > 0 else 0
    gate_specificity = gate_tn / (gate_tn + gate_fp) if (gate_tn + gate_fp) > 0 else 0
    gate_f1 = 2 * gate_tp / (2 * gate_tp + gate_fp + gate_fn) if (2 * gate_tp + gate_fp + gate_fn) > 0 else 0
    
    # ========================================
    # CONFIDENCE ANALYSE
    # ========================================
    
    # Confident retrievals (high score AND high margin)
    confident = sum(1 for row in data 
                    if float(row['score']) >= 0.8 
                    and row.get('margin') 
                    and float(row['margin']) >= 0.3)
    
    # Ambiguous retrievals (low margin - near threshold)
    ambiguous = sum(1 for row in data 
                    if row.get('margin') 
                    and float(row['margin']) < 0.1)
    
    # ========================================
    # MARGIN STATISTIKEN
    # ========================================
    
    def percentile(values, p):
        if not values:
            return 0
        sorted_vals = sorted(values)
        idx = min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)
        return sorted_vals[idx]
    
    margin_mean = mean(margins) if margins else 0
    margin_std = stdev(margins) if len(margins) > 1 else 0
    margin_p50 = percentile(margins, 50)
    margin_p90 = percentile(margins, 90)
    margin_p10 = percentile(margins, 10)
    
    # ========================================
    # FINAL KPIS DICHTIONAR
    # ========================================
    
    return {
        # Gesamtstatistik
        'total_questions': total,
        
        # Verdict Verteilung
        'verdict_distribution': dict(verdicts),
        
        # Answerability Success
        'overall_success_rate': gate_tp / total if total else 0,
        'true_positives': gate_tp,
        'false_positives': gate_fp,
        'false_negatives': gate_fn,
        'true_negatives': gate_tn,
        
        # 🎯 GATE PERFORMANCE (neu/korrigiert)
        'gate_precision': gate_precision,
        'gate_recall': gate_recall,
        'gate_specificity': gate_specificity,
        'gate_f1_score': gate_f1,
        
        # 📊 MARGIN ANALYSE (neu)
        'margin_mean': margin_mean,
        'margin_std': margin_std,
        'margin_median': margin_p50,
        'margin_p10': margin_p10,
        'margin_p90': margin_p90,
        'ambiguity_rate': ambiguous / total if total else 0,
        
        # Score Statistics
        'avg_score': mean(scores) if scores else 0,
        'median_score': percentile(scores, 50),
        'min_score': min(scores) if scores else 0,
        'max_score': max(scores) if scores else 0,
        
        # Confidence Metrics
        'confident_retrievals_pct': confident / total if total else 0,
        'ambiguous_retrievals_pct': ambiguous / total if total else 0,
        'confidence_gap_mean': margin_mean,
        
        # Categories breakdown
        'categories': {
            cat: {
                'total': stats['total'],
                'success_rate': stats['ok'] / stats['total'] if stats['total'] else 0,
                'ok_count': stats['ok'],
            }
            for cat, stats in categories.items()
        },
    }


# ============================================
# COMBINED REPORT
# ============================================

def generate_kpi_report(eval_dir: Path, run_id: str | None = None) -> dict:
    """Generate comprehensive KPI report from evaluation directory."""
    
    load_files = sorted(eval_dir.glob('load-test-quick-*.csv'))
    results_files = sorted(eval_dir.glob('results_*.csv'))
    resources_files = sorted(eval_dir.glob('*-resources.csv'))
    
    if not load_files or not results_files:
        raise ValueError(f"No evaluation files found in {eval_dir}")
    
    latest_load = load_files[-1]
    latest_results = results_files[-1]
    latest_resources = resources_files[-1] if resources_files else None
    
    if not run_id and load_files:
        run_id = latest_load.stem.replace('load-test-quick-', '')
    
    kpis = {
        'run_id': run_id,
        'generated_at': datetime.now().isoformat(),
        'files_analyzed': {
            'load_test': str(latest_load.name),
            'results': str(latest_results.name),
            'resources': str(latest_resources.name) if latest_resources else None,
        },
        'latency': calculate_latency_kpis(latest_load),
        'quality': calculate_quality_kpis(latest_results),
    }
    
    # Add resource stats separately
    if latest_resources:
        kpis['resources'] = calculate_resource_kpis(latest_resources)
        # Also add convenience copy to latency for backward compatibility
        res = kpis['resources']
    
    # Composite health score (0-100)
    quality_score = kpis['quality']['overall_success_rate'] * 50
    latency_score = max(0, (1 - kpis['latency']['p95_response_time_ms'] / 500)) * 30
    confidence_score = kpis['quality']['confident_retrievals_pct'] * 20
    kpis['composite_health_score'] = round(quality_score + latency_score + confidence_score, 1)
    
    return kpis


def save_kpis(kpis: dict, output_path: Path) -> None:
    """Save KPIs to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(kpis, f, indent=2, ensure_ascii=False)


def print_summary(kpis: dict) -> None:
    """Print human-readable summary."""
    q = kpis['quality']
    l = kpis['latency']
    
    print("\n" + "="*60)
    print("RAG CHATBOT EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\n RUN INFO")
    print(f"   Run ID: {kpis['run_id']}")
    print(f"   Generated: {kpis['generated_at']}")
    
    print(f"\n  LATENCY KPIS")
    print(f"   Total Requests: {l['total_requests']}")
    print(f"   Avg Response Time: {l['avg_response_time_ms']:.1f} ms")
    print(f"   P95 Response Time: {l['p95_response_time_ms']:.1f} ms")
    print(f"   Throughput: {l['throughput_requests_per_sec']:.1f} req/sec")
    
    # Resource KPIs
    if 'resources' in kpis:
        r = kpis['resources']
        
        print(f"\n RESOURCE UTILIZATION")
        
        # Chatbot Metriken
        chatbot_cpu = r.get('chatbot_avg_cpu_pct', 0)
        chatbot_mem = r.get('chatbot_avg_mem_mb', 0)
        print(f"   Chatbot Avg CPU: {chatbot_cpu:.1f}%")
        print(f"   Chatbot Peak CPU: {r.get('chatbot_peak_cpu_pct', 0):.1f}%")
        print(f"   Chatbot Avg Memory: {chatbot_mem:.1f} MB")
        print(f"   System Avg CPU: {r.get('system_avg_cpu_pct', 0):.1f}%")
        
        # Ollama/llama-server Metriken (hochgeladene Last)
        ollama_cpu = r.get('ollama_llama-server_exe_avg_cpu', 0)
        ollama_peak_cpu = r.get('ollama_llama-server_exe_peak_cpu', 0)
        ollama_mem = r.get('ollama_llama-server_exe_avg_mem_mb', 0)
        ollama_peak_mem = r.get('ollama_llama-server_exe_peak_mem_mb', 0)
        
        # Nur anzeigen wenn Werte vorhanden sind
        if ollama_cpu or ollama_mem:
            print(f"\n OLLAMA / LLAMA-SERVER")
            print(f"   Avg CPU: {ollama_cpu:.1f}%")
            print(f"   Peak CPU: {ollama_peak_cpu:.1f}%")
            print(f"   Avg Memory: {ollama_mem:.1f} MB")
            print(f"   Peak Memory: {ollama_peak_mem:.1f} MB")
    
    print(f"\n ANSWER QUALITY")
    print(f"   Total Questions: {q['total_questions']}")
    print(f"   Overall Success Rate: {q['overall_success_rate']*100:.1f}%")
    print(f"   True Positives: {q['true_positives']}")
    
    print(f"\n GATE PERFORMANCE")
    print(f"   Precision: {q['gate_precision']*100:.1f}%")
    print(f"   Recall: {q['gate_recall']*100:.1f}%")
    print(f"   Specificity: {q['gate_specificity']*100:.1f}%")
    print(f"   F1 Score: {q['gate_f1_score']:.3f}")
    print(f"   └─ False Positives (zu liberal): {q['false_positives']}")
    print(f"   └─ False Negatives (zu streng): {q['false_negatives']}")
    print(f"   └─ True Negatives (korrekt blockiert): {q['true_negatives']}")
    
    print(f"\n RETRIEVAL QUALITY")
    print(f"   Avg Score: {q['avg_score']:.3f}")
    print(f"   Median Score: {q['median_score']:.3f}")
    print(f"   Avg Margin: {q['margin_mean']:.3f}")
    print(f"   Margin StdDev: {q['margin_std']:.3f}")
    print(f"   Margins (P10/P90): {q['margin_p10']:.3f} / {q['margin_p90']:.3f}")
    print(f"   Confident Retrievals: {q['confident_retrievals_pct']*100:.1f}%")
    print(f"   Ambiguous (margin<0.1): {q['ambiguous_retrievals_pct']*100:.1f}%")
    
    # Kategorien (top 5 nach total)
    cats = sorted(q['categories'].items(), key=lambda x: x[1]['total'], reverse=True)[:5]
    if cats:
        print(f"\n TOP CATEGORIES")
        for cat, stats in cats:
            print(f"   {cat}: {stats['success_rate']*100:.1f}% ({stats['ok_count']}/{stats['total']})")
    
    print(f"\n COMPOSITE HEALTH SCORE: {kpis['composite_health_score']}/100")
    print("="*60 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate KPIs from RAG evaluation files')
    parser.add_argument('--dir', '-d', type=Path, default=Path('.'),
                       help='Directory containing evaluation files')
    parser.add_argument('--output', '-o', type=Path, default=Path('kpis.json'),
                       help='Output JSON file')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress console output')
    args = parser.parse_args()
    
    try:
        kpis = generate_kpi_report(args.dir)
        save_kpis(kpis, args.output)
        
        if not args.quiet:
            print_summary(kpis)
        
        print(f"✓ KPIs saved to {args.output}")
    except Exception as e:
        print(f"✗ Error: {e}")
        exit(1)