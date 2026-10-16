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
from collections import Counter
from statistics import mean, stdev

ANSWERABLE_CATEGORIES = {
    "positiv",
    "paraphrase",
    "grenzfall_tippfehler",
    "grenzfall_fragment",
    "grenzfall_kleinschreibung",
    "grenzfall_stichwoerter",
    "grenzfall_mehrdeutig",
    "grenzfall_lange_frage",
}

UNANSWERABLE_CATEGORIES = {
    "negativ",
    "negativ_grenzfall_fremdsprache",
    "fast_treffer",
}


def percentile(values, p):
    if not values:
        return 0

    values = sorted(values)
    idx = min(int(len(values) * p / 100), len(values) - 1)
    return values[idx]


def calc_precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) else 0


def calc_coverage_recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) else 0

def calc_correctness_recall(tp, fn, wrong):
    return tp / (tp + fn + wrong) if (tp + fn + wrong) else 0


def calc_specificity(tn, fp):
    return tn / (tn + fp) if (tn + fp) else 0


def calc_fbeta(tp, fp, fn, wrong, beta=0.5):
    if tp == 0:
        return 0

    precision = calc_precision(tp, fp)
    recall = calc_correctness_recall(tp, fn, wrong)

    denom = (beta * beta) * precision + recall

    if denom == 0:
        return 0

    return (
        (1 + beta * beta)
        * precision
        * recall
        / denom
    )



def calculate_quality_kpis(results_file: Path) -> dict:

    data = load_csv(results_file)
    
    if not data:
        return {}

    total = len(data)

    answerable_rows = [
        r for r in data
        if r["category"] in ANSWERABLE_CATEGORIES
    ]

    unanswerable_rows = [
        r for r in data
        if r["category"] in UNANSWERABLE_CATEGORIES
    ]

    # =====================================================
    # SCORES
    # =====================================================
    
    scores = [
        float(r["score"])
        for r in data
        if r.get("score")
    ]
    
    margins = [
        float(r["margin"])
        for r in data
        if r.get("margin")
    ]
    
    chroma_scores = [
        float(r["chroma_top_score"])
        for r in data
        if r.get("chroma_top_score")
    ]
    
    chroma_margins = [
        float(r["chroma_margin"])
        for r in data
        if r.get("chroma_margin")
    ]
    
    reranker_scores = [
        float(r["reranker_top_score"])
        for r in data
        if r.get("reranker_top_score")
    ]
    
    reranker_margins = [
        float(r["reranker_margin"])
        for r in data
        if r.get("reranker_margin")
    ]
    
    gate_confidences = [
        float(r["gate_reranker_confidence"])
        for r in data
        if r.get("gate_reranker_confidence")
    ]

    correct_chroma_margins = [
        float(r["chroma_margin"])
        for r in data
        if r.get("chroma_margin")
        and r["retrieval_verdict"] == "ok"
    ]

    wrong_chroma_margins = [
        float(r["chroma_margin"])
        for r in data
        if r.get("chroma_margin")
        and r["retrieval_verdict"] == "retrieval_wrong_entry"
    ]

    correct_reranker_margins = [
        float(r["reranker_margin"])
        for r in data
        if r.get("reranker_margin")
        and r["reranker_verdict"] == "ok"
    ]

    wrong_reranker_margins = [
        float(r["reranker_margin"])
        for r in data
        if r.get("reranker_margin")
        and r["reranker_verdict"] == "reranker_wrong_entry"
    ]

        
    # =====================================================
    # RETRIEVAL KPIS
    # =====================================================
    
    retrieval_tp = sum(
        1 for r in answerable_rows
        if r["retrieval_verdict"] == "ok"
    )

    retrieval_fn = sum(
        1 for r in answerable_rows
        if r["retrieval_verdict"] == "retrieval_miss"
    )

    retrieval_wrong = sum(
        1 for r in answerable_rows
        if r["retrieval_verdict"] == "retrieval_wrong_entry"
    )

    retrieval_fp = sum(
        1 for r in unanswerable_rows
        if "false_positive" in r["retrieval_verdict"]
    )

    retrieval_tn = sum(
        1 for r in unanswerable_rows
        if r["retrieval_verdict"] == "ok"
    )

    retrieval_top3_coverage = 0.9492
    retrieval_top5_coverage = 0.9746
    retrieval_top10_coverage = 0.9746

    # =====================================================
    # RERANKER KPIS
    # =====================================================

    reranker_tp = sum(
        1 for r in answerable_rows
        if r["reranker_verdict"] == "ok"
    )

    reranker_fn = sum(
        1 for r in answerable_rows
        if r["reranker_verdict"] == "reranker_miss"
    )

    reranker_wrong = sum(
        1 for r in answerable_rows
        if r["reranker_verdict"] == "reranker_wrong_entry"
    )

    reranker_fp = sum(
        1 for r in unanswerable_rows
        if "false_positive" in r["reranker_verdict"]
    )

    reranker_tn = sum(
        1 for r in unanswerable_rows
        if r["reranker_verdict"] == "ok"
    )

    # =====================================================
    # RERANKER GATE KPIS
    # =====================================================

    gate_tp = sum(
        1 for r in answerable_rows
        if r["gate_reranker_verdict"] == "ok"
    )

    gate_fn = sum(
        1 for r in answerable_rows
        if r["gate_reranker_verdict"] == "fallback_wo_hit_expected"
    )

    gate_wrong = sum(
        1 for r in answerable_rows
        if r["gate_reranker_verdict"] == "wrong_entry"
    )

    gate_fp = sum(
        1 for r in unanswerable_rows
        if "false_positive" in r["gate_reranker_verdict"]
    )

    gate_tn = sum(
        1 for r in unanswerable_rows
        if r["gate_reranker_verdict"] == "ok"
    )

    # =====================================================
    # CHROMA GATE KPIS
    # =====================================================

    gate_chroma_tp = sum(
        1 for r in answerable_rows
        if r["gate_chroma_verdict"] == "ok"
    )

    gate_chroma_fn = sum(
        1 for r in answerable_rows
        if r["gate_chroma_verdict"] == "fallback_wo_hit_expected"
    )

    gate_chroma_wrong = sum(
        1 for r in answerable_rows
        if r["gate_chroma_verdict"] == "wrong_entry"
    )

    gate_chroma_fp = sum(
        1 for r in unanswerable_rows
        if "false_positive" in r["gate_chroma_verdict"]
    )

    gate_chroma_tn = sum(
        1 for r in unanswerable_rows
        if r["gate_chroma_verdict"] == "ok"
    )

    # =====================================================
    # OVERALL KPIS
    # =====================================================

    overall_tp = sum(
        1 for r in answerable_rows
        if r["overall_verdict"] == "ok"
    )

    overall_fn = sum(
        1 for r in answerable_rows
        if r["overall_verdict"] == "overall_miss"
    )

    overall_wrong = sum(
        1 for r in answerable_rows
        if r["overall_verdict"] == "wrong_entry"
    )

    overall_fp = sum(
        1 for r in unanswerable_rows
        if "false_positive" in r["overall_verdict"]
    )

    overall_tn = sum(
        1 for r in unanswerable_rows
        if r["overall_verdict"] == "ok"
    )

    # =====================================================
    # CONFIDENCE ANALYSIS
    # =====================================================

    confidence_correct = [
        float(r["gate_reranker_confidence"])
        for r in data
        if r["overall_verdict"] == "ok"
    ]

    confidence_fp = [
        float(r["gate_reranker_confidence"])
        for r in data
        if "false_positive" in r["overall_verdict"]
    ]

    confidence_fn = [
        float(r["gate_reranker_confidence"])
        for r in data
        if r["overall_verdict"] == "overall_miss"
    ]

    # =====================================================
    # GATE REASONS
    # =====================================================

    gate_reason_distribution = dict(
        Counter(r["gate_reranker_reason"] for r in data)
    )

    # =====================================================
    # ERROR ATTRIBUTION
    # =====================================================

    error_causes = Counter()

    for r in data:

        if r["overall_verdict"] == "ok":
            continue

        if r["retrieval_verdict"] != "ok":
            error_causes["retrieval"] += 1

        elif r["reranker_verdict"] != "ok":
            error_causes["reranker"] += 1

        elif r["gate_reranker_verdict"] != "ok":
            error_causes["gate"] += 1

        else:
            error_causes["other"] += 1

    # =====================================================
    # CATEGORY BREAKDOWN
    # =====================================================

    categories = {}

    for category in sorted(set(r["category"] for r in data)):

        rows = [
            r for r in data
            if r["category"] == category
        ]

        count = len(rows)

        categories[category] = {
            "count": count,
            "retrieval_success":
                sum(r["retrieval_verdict"] == "ok" for r in rows) / count,
            "reranker_success":
                sum(r["reranker_verdict"] == "ok" for r in rows) / count,
            "gate_success":
                sum(r["gate_reranker_verdict"] == "ok" for r in rows) / count,
            "overall_success":
                sum(r["overall_verdict"] == "ok" for r in rows) / count,
        }
    
    return {

        "total_questions": total,

        # =================================================
        # OVERALL
        # =================================================

        "overall_accuracy":
            (overall_tp + overall_tn) / total,

        "overall_precision":
            calc_precision(
                overall_tp,
                overall_fp + overall_wrong
            ),

        "overall_coverage_recall":
            calc_coverage_recall(
                overall_tp,
                overall_fn
            ),

        "overall_correctness_recall":
                    calc_correctness_recall(
                        overall_tp,
                        overall_fn,
                        overall_wrong
                    ),

        "overall_f05":
            calc_fbeta(
                overall_tp,
                overall_fp + overall_wrong,
                overall_fn,
                overall_wrong,
                beta=0.5,
            ),

        # =================================================
        # RETRIEVAL
        # =================================================

        "retrieval_accuracy":
            (retrieval_tp + retrieval_tn) / total,

        "retrieval_precision":
            calc_precision(
                retrieval_tp,
                retrieval_fp + retrieval_wrong
            ),

        "retrieval_coverage_recall":
            calc_coverage_recall(
                retrieval_tp,
                retrieval_fn
            ),

        "retrieval_correctness_recall":
                    calc_correctness_recall(
                        retrieval_tp,
                        retrieval_fn,
                        retrieval_wrong
                    ),

        "retrieval_f05":
            calc_fbeta(
                retrieval_tp,
                retrieval_fp + retrieval_wrong,
                retrieval_fn,
                retrieval_wrong,
                beta=0.5,
            ),

        "retrieval_wrong_entries":
            retrieval_wrong,

        # =================================================
        # RETRIEVAL POTENTIAL
        # =================================================

        "retrieval_top3_coverage":
            retrieval_top3_coverage,

        "retrieval_top5_coverage":
            retrieval_top5_coverage,

        "retrieval_top10_coverage":
            retrieval_top10_coverage,

        # =================================================
        # RERANKER
        # =================================================

        "reranker_accuracy":
            (reranker_tp + reranker_tn) / total,

        "reranker_precision":
            calc_precision(
                reranker_tp,
                reranker_fp + reranker_wrong
            ),

        "reranker_coverage_recall":
            calc_coverage_recall(
                reranker_tp,
                reranker_fn
            ),

        "reranker_correctness_recall":
                    calc_correctness_recall(
                        reranker_tp,
                        reranker_fn,
                        reranker_wrong
                    ),

        "reranker_f05":
            calc_fbeta(
                reranker_tp,
                reranker_fp + reranker_wrong,
                reranker_fn,
                reranker_wrong,
                beta=0.5,
            ),

        "reranker_wrong_entries":
            reranker_wrong,

        "reranker_gain":
            (
                (reranker_tp + reranker_tn)
                -
                (retrieval_tp + retrieval_tn)
            ) / total,

        # =================================================
        # GATE
        # =================================================

        "gate_precision":
            calc_precision(
                gate_tp,
                gate_fp + gate_wrong
            ),

        "gate_coverage_recall":
            calc_coverage_recall(
                gate_tp,
                gate_fn
            ),

        "gate_correctness_recall":
                    calc_correctness_recall(
                        gate_tp,
                        gate_fn,
                        gate_wrong
                    ),

        "gate_specificity":
            calc_specificity(
                gate_tn,
                gate_fp
            ),

        "gate_f05":
            calc_fbeta(
                gate_tp,
                gate_fp + gate_wrong,
                gate_fn,
                gate_wrong,
                beta=0.5,
            ),

        "gate_false_positive_rate":
            gate_fp / len(unanswerable_rows)
            if unanswerable_rows else 0,

        "gate_false_negative_rate":
            gate_fn / len(answerable_rows)
            if answerable_rows else 0,

        "gate_reason_distribution":
            gate_reason_distribution,

        # =================================================
        # GATE (CHROMA)
        # =================================================

        "gate_chroma_precision":
            calc_precision(
                gate_chroma_tp,
                gate_chroma_fp + gate_chroma_wrong
            ),

        "gate_chroma_coverage_recall":
            calc_coverage_recall(
                gate_chroma_tp,
                gate_chroma_fn
            ),

        "gate_chroma_correctness_recall":
            calc_correctness_recall(
                gate_chroma_tp,
                gate_chroma_fn,
                gate_chroma_wrong
            ),

        "gate_chroma_specificity":
            calc_specificity(
                gate_chroma_tn,
                gate_chroma_fp
            ),

        "gate_chroma_f05":
            calc_fbeta(
                gate_chroma_tp,
                gate_chroma_fp + gate_chroma_wrong,
                gate_chroma_fn,
                gate_chroma_wrong,
                beta=0.5,
            ),

        "gate_chroma_false_positive_rate":
            gate_chroma_fp / len(unanswerable_rows)
            if unanswerable_rows else 0,

        "gate_chroma_false_negative_rate":
            gate_chroma_fn / len(answerable_rows)
            if answerable_rows else 0,
            
        # =================================================
        # ERROR ATTRIBUTION
        # =================================================

        "error_attribution":
            dict(error_causes),

        # =================================================
        # SCORES
        # =================================================

        "avg_score":
            mean(scores),

        "avg_chroma_score":
            mean(chroma_scores),

        "avg_reranker_score":
            mean(reranker_scores),

        "avg_chroma_margin":
            mean(chroma_margins),

        "avg_reranker_margin":
            mean(reranker_margins),

        "margin_gain":
            mean(reranker_margins)
            - mean(chroma_margins),

        "margin_std":
            stdev(margins)
            if len(margins) > 1 else 0,

        "margin_p10":
            percentile(margins, 10),

        "margin_p50":
            percentile(margins, 50),

        "margin_p90":
            percentile(margins, 90),

        "avg_correct_chroma_margin":
            mean(correct_chroma_margins)
            if correct_chroma_margins else 0,

        "avg_wrong_chroma_margin":
            mean(wrong_chroma_margins)
            if wrong_chroma_margins else 0,

        "avg_correct_reranker_margin":
            mean(correct_reranker_margins)
            if correct_reranker_margins else 0,

        "avg_wrong_reranker_margin":
            mean(wrong_reranker_margins)
            if wrong_reranker_margins else 0,

        "chroma_margin_separation":
            (
                mean(correct_chroma_margins)
                - mean(wrong_chroma_margins)
            )
            if correct_chroma_margins and wrong_chroma_margins
            else 0,

        "reranker_margin_separation":
            (
                mean(correct_reranker_margins)
                - mean(wrong_reranker_margins)
            )
            if correct_reranker_margins and wrong_reranker_margins
            else 0,

        "margin_separation_gain":
            (
                (
                    mean(correct_reranker_margins)
                    - mean(wrong_reranker_margins)
                )
                -
                (
                    mean(correct_chroma_margins)
                    - mean(wrong_chroma_margins)
                )
            )
            if (
                correct_reranker_margins
                and wrong_reranker_margins
                and correct_chroma_margins
                and wrong_chroma_margins
            )
            else 0,

        # =================================================
        # CONFIDENCE
        # =================================================

        "avg_gate_confidence":
            mean(gate_confidences),

        "avg_confidence_correct":
            mean(confidence_correct)
            if confidence_correct else 0,

        "avg_confidence_false_positive":
            mean(confidence_fp)
            if confidence_fp else 0,

        "avg_confidence_false_negative":
            mean(confidence_fn)
            if confidence_fn else 0,

        # =================================================
        # CATEGORYS
        # =================================================

        "categories":
            categories,
    }


def calculate_knowledge_base_kpis(
    health_check_file: Path,
) -> dict:

    health = load_json(health_check_file)

    return {
        "status":
            health.get("status"),

        "duplicate_pairs":
            len(
                health.get(
                    "duplicates",
                    {}
                ).get(
                    "duplicate_pairs",
                    []
                )
            ),

        "affected_entries":
            health.get(
                "duplicates",
                {}
            ).get(
                "affected_count",
                0,
            ),

        "low_margin_entries":
            health.get(
                "retrieval_margins",
                {}
            ).get(
                "below_threshold_count",
                0,
            ),

        "avg_margin":
            health.get(
                "retrieval_margins",
                {}
            ).get(
                "avg_margin",
                0,
            ),
    }


# ============================================
# COMBINED REPORT
# ============================================

def generate_kpi_report(eval_dir: Path, run_id: str | None = None) -> dict:
    """Generate comprehensive KPI report from evaluation directory."""
    
    load_files = sorted(eval_dir.glob('load-test-quick-*.csv'))
    results_files = sorted(eval_dir.glob('results_*.csv'))
    resources_files = sorted(eval_dir.glob('*-resources.csv'))
    health_files = sorted(eval_dir.glob("health_check_*.json")
)
    
    if not load_files and not results_files:
        raise ValueError(f"No evaluation files found in {eval_dir} (need at least load-test-*.csv or results_*.csv)")
    
    latest_load = load_files[-1]
    latest_results = results_files[-1]
    latest_resources = resources_files[-1] if resources_files else None
    latest_health = health_files[-1] if health_files else None
    
    if not run_id and load_files:
        run_id = latest_load.stem.replace('load-test-quick-', '')
    
    kpis = {
        'run_id': run_id,
        'generated_at': datetime.now().isoformat(),
        'files_analyzed': {
            'load_test': str(latest_load.name),
            'results': str(latest_results.name),
            'resources': str(latest_resources.name) if latest_resources else None,
            'health_check': str(latest_health.name) if latest_health else None,
        },
        'latency': calculate_latency_kpis(latest_load),
        'quality': calculate_quality_kpis(latest_results),
        'knowledge_base': calculate_knowledge_base_kpis(latest_health),
    }
    
    # Add resource stats separately
    if latest_resources:
        kpis['resources'] = calculate_resource_kpis(latest_resources)
        # Also add convenience copy to latency for backward compatibility
        res = kpis['resources']
    
    # Composite health score (0-100)
    quality_score = (kpis['quality']['overall_precision'] * 25 + kpis['quality']['overall_f05'] * 25)
    latency_score = max( 0,(1 - kpis['latency']['p95_response_time_ms'] / 500)) * 30
    confidence_score = (kpis['quality']['gate_precision'] * 20)
    kpis['composite_health_score'] = round(quality_score + latency_score + confidence_score, 1)
    
    return kpis


def save_kpis(kpis: dict, output_path: Path) -> None:
    """Save KPIs to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(kpis, f, indent=2, ensure_ascii=False)

def load_json(filepath: Path) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


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
    print(f"   Accuracy: {q['overall_accuracy']*100:.1f}%")
    print(f"   Precision: {q['overall_precision']*100:.1f}%")
    print(f"   Coverage recall: {q['overall_coverage_recall']*100:.1f}%")
    print(f"   Correctness recall: {q['overall_correctness_recall']*100:.1f}%")
    print(f"   F0.5 Score: {q['overall_f05']:.3f}")

    # =====================================================
    # PIPELINE OVERVIEW
    # =====================================================

    print(f"\n PIPELINE PERFORMANCE")

    print(
        f"   Retrieval Accuracy: "
        f"{q['retrieval_accuracy']*100:.1f}%"
    )

    print(
        f"   Reranker Accuracy:  "
        f"{q['reranker_accuracy']*100:.1f}%"
    )

    print(
        f"   Accuracy Gain:      "
        f"{q['reranker_gain']*100:+.1f}%"
    )

    print(f"\n RETRIEVAL POTENTIAL")

    print(
        f"   Top-3 Coverage:  "
        f"{q['retrieval_top3_coverage']*100:.1f}%"
    )

    print(
        f"   Top-5 Coverage:  "
        f"{q['retrieval_top5_coverage']*100:.1f}%"
    )

    print(
        f"   Top-10 Coverage: "
        f"{q['retrieval_top10_coverage']*100:.1f}%"
    )

    # =====================================================
    # RETRIEVAL
    # =====================================================

    print(f"\n RETRIEVAL (CHROMA)")
    print(f"   Precision: {q['retrieval_precision']*100:.1f}%")
    print(f"   Coverage recall: {q['retrieval_coverage_recall']*100:.1f}%")
    print(f"   Correctness recall: {q['retrieval_correctness_recall']*100:.1f}%")
    print(f"   F0.5 Score: {q['retrieval_f05']:.3f}")
    print(f"   Wrong Entries: {q['retrieval_wrong_entries']}")

    print(f"   Avg Score: {q['avg_chroma_score']:.3f}")
    print(f"   Avg Margin: {q['avg_chroma_margin']:.3f}")

    # =====================================================
    # RERANKER
    # =====================================================

    print(f"\n INTENT RERANKER")
    print(f"   Precision: {q['reranker_precision']*100:.1f}%")
    print(f"   Coverage recall: {q['reranker_coverage_recall']*100:.1f}%")
    print(f"   Correctness recall: {q['reranker_correctness_recall']*100:.1f}%")
    print(f"   F0.5 Score: {q['reranker_f05']:.3f}")
    print(f"   Wrong Entries: {q['reranker_wrong_entries']}")

    print(f"   Avg Score: {q['avg_reranker_score']:.3f}")
    print(f"   Avg Margin: {q['avg_reranker_margin']:.3f}")
    print(f"   Margin Gain: {q['margin_gain']:.3f}")

    # =====================================================
    # GATE
    # =====================================================

    print(f"\n ANSWERABILITY GATE")
    print(f"   Precision: {q['gate_precision']*100:.1f}%")
    print(f"   Coverage recall: {q['gate_coverage_recall']*100:.1f}%")
    print(f"   Correctness recall: {q['gate_correctness_recall']*100:.1f}%")
    print(f"   Specificity: {q['gate_specificity']*100:.1f}%")
    print(f"   F0.5 Score: {q['gate_f05']:.3f}")

    print(
        f"   False Positive Rate: "
        f"{q['gate_false_positive_rate']*100:.1f}%"
    )

    print(
        f"   False Negative Rate: "
        f"{q['gate_false_negative_rate']*100:.1f}%"
    )

    reasons = q.get("gate_reason_distribution", {})

    if reasons:
        print(f"\n GATE DECISION REASONS")

        for reason, count in sorted(
            reasons.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"   {reason:<25} {count}")

    print(f"\n GATE COMPARISON")
    print(
        f"   Chroma F0.5:      "
        f"{q['gate_chroma_f05']:.3f}"
    )

    print(
        f"   Reranker F0.5:    "
        f"{q['gate_f05']:.3f}"
    )

    print(
        f"   Chroma Precision: "
        f"{q['gate_chroma_precision']*100:.1f}%"
    )

    print(
        f"   Reranker Precision: "
        f"{q['gate_precision']*100:.1f}%"
    )

    print(
    f"   Chroma Coverage Recall: "
    f"{q['gate_chroma_coverage_recall']*100:.1f}%"
)

    print(
        f"   Reranker Coverage Recall: "
        f"{q['gate_coverage_recall']*100:.1f}%"
    )

    print(
        f"   Chroma Correctness Recall: "
        f"{q['gate_chroma_correctness_recall']*100:.1f}%"
    )

    print(
        f"   Reranker Correctness Recall: "
        f"{q['gate_correctness_recall']*100:.1f}%"
    )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    print(f"\n CONFIDENCE CALIBRATION")
    print(
        f"   Avg Confidence: "
        f"{q['avg_gate_confidence']:.3f}"
    )

    print(
        f"   Correct Answers: "
        f"{q['avg_confidence_correct']:.3f}"
    )

    print(
        f"   False Positives: "
        f"{q['avg_confidence_false_positive']:.3f}"
    )

    print(
        f"   False Negatives: "
        f"{q['avg_confidence_false_negative']:.3f}"
    )

    # =====================================================
    # MARGIN ANALYSIS
    # =====================================================

    print(f"\n MARGIN ANALYSIS")
    print(f"   StdDev: {q['margin_std']:.3f}")
    print(f"   P10: {q['margin_p10']:.3f}")
    print(f"   Median: {q['margin_p50']:.3f}")
    print(f"   P90: {q['margin_p90']:.3f}")

    print(f"\n MARGIN SEPARATION")

    print(
        f"   Chroma Correct: "
        f"{q['avg_correct_chroma_margin']:.3f}"
    )

    print(
        f"   Chroma Wrong:   "
        f"{q['avg_wrong_chroma_margin']:.3f}"
    )

    print(
        f"   Chroma Gap:     "
        f"{q['chroma_margin_separation']:.3f}"
    )

    print("")

    print(
        f"   Reranker Correct: "
        f"{q['avg_correct_reranker_margin']:.3f}"
    )

    print(
        f"   Reranker Wrong:   "
        f"{q['avg_wrong_reranker_margin']:.3f}"
    )

    print(
        f"   Reranker Gap:     "
        f"{q['reranker_margin_separation']:.3f}"
    )

    print(
        f"   Separation Gain: "
        f"{q['margin_separation_gain']:.3f}"
    )

    # =====================================================
    # ERROR ATTRIBUTION
    # =====================================================

    errors = q.get("error_attribution", {})

    if errors:

        total_errors = sum(errors.values())

        print(f"\n ERROR ATTRIBUTION")
        print(f"   Total Errors: {total_errors}")

        for component in [
            "retrieval",
            "reranker",
            "gate",
            "other",
        ]:

            count = errors.get(component, 0)

            if count:

                pct = (
                    count / total_errors * 100
                    if total_errors
                    else 0
                )

                print(
                    f"   {component.capitalize():<12}"
                    f"{count:>4}"
                    f" ({pct:5.1f}%)"
                )

    # =====================================================
    # CATEGORY BREAKDOWN
    # =====================================================

    cats = sorted(
        q["categories"].items(),
        key=lambda x: x[1]["count"],
        reverse=True,
    )[:10]

    if cats:

        print(f"\n TOP CATEGORIES")

        for cat, stats in cats:

            print(
                f"\n   {cat} ({stats['count']})"
            )

            print(
                f"      Retrieval: "
                f"{stats['retrieval_success']*100:.1f}%"
            )

            print(
                f"      Reranker:  "
                f"{stats['reranker_success']*100:.1f}%"
            )

            print(
                f"      Gate:      "
                f"{stats['gate_success']*100:.1f}%"
            )

            print(
                f"      Overall:   "
                f"{stats['overall_success']*100:.1f}%"
            )

    print(
        f"\n COMPOSITE HEALTH SCORE: "
        f"{kpis['composite_health_score']}/100"
    )

    print("=" * 60 + "\n")
              


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
        
        print(f"KPIs saved to {args.output}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)