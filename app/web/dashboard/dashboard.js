/* ============================================================
   FAQ-Chatbot · Operations Dashboard
   ============================================================ */

(function () {
    'use strict';

    const CONFIG = {
        primaryPath: './dashboard-assets/metrics.json',
        fallbackPath: './dashboard-assets/kpis_latest.json',
        locale: 'de-DE'
    };

    const THRESHOLDS = {
        precision:          { good: 0.95, warn: 0.85, dir: 'up' },
        correctnessRecall:  { good: 0.85, warn: 0.70, dir: 'up' },
        f05:                { good: 0.90, warn: 0.80, dir: 'up' },
        coverage:           { good: 0.95, warn: 0.85, dir: 'up' },
        latencyAvg:         { good: 120,  warn: 240,  dir: 'down' },
        latencyP95:         { good: 265,  warn: 460,  dir: 'down' },
        procCpu:            { good: 60,   warn: 85,   dir: 'down' }
    };

    /* ============================================================
       UTILS
       ============================================================ */
    function getStatus(value, thresholds) {
        if (!thresholds || typeof value !== 'number' || Number.isNaN(value)) {
            return { class: 'neutral', label: '-' };
        }

        if (thresholds.dir === 'down') {
            if (value <= thresholds.good) return { class: 'good', label: 'OK' };
            if (value <= thresholds.warn) return { class: 'warn', label: 'ATTENTION' };
            return { class: 'bad', label: 'CRITICAL' };
        }

        if (value >= thresholds.good) return { class: 'good', label: 'OK' };
        if (value >= thresholds.warn) return { class: 'warn', label: 'ATTENTION' };
        return { class: 'bad', label: 'CRITICAL' };
    }

    function fmtPercent(v, d = 1) { return (v * 100).toFixed(d) + '%'; }
    function fmtMs(v) { return Math.round(v) + ' ms'; }
    function fmtNum(v) { return Math.round(v).toLocaleString(CONFIG.locale); }
    function fmtDate(iso) {
        try {
            return new Date(iso).toLocaleString(CONFIG.locale, {
                year: 'numeric', month: '2-digit', day: '2-digit',
                hour: '2-digit', minute: '2-digit', second: '2-digit'
            });
        } catch { return iso; }
    }

    /* ============================================================
       DOM HELPERS
       ============================================================ */
    function el(id) { return document.getElementById(id); }

    // Akzeptiert sowohl eine ID (String) als auch ein Element
    function resolve(target) {
        return typeof target === 'string' ? el(target) : target;
    }

    function hide(target) { const e = resolve(target); if (e) e.hidden = true; }
    function show(target) { const e = resolve(target); if (e) e.hidden = false; }
    function text(id, txt) { const e = el(id); if (e) e.textContent = txt; }

    /* ------------------------------------------------------------
       BARS & METRICS
       ------------------------------------------------------------ */
    function barLabel(label, value, max, status) {
        const w = Math.min(100, (value / max) * 100);
        return `
            <div class="bar-row">
                <span class="bar-label">${label}</span>
                <div class="bar-track">
                    <div class="bar-fill ${status.class}" style="width:${w}%"></div>
                </div>
                <span class="bar-val">${value.toLocaleString(CONFIG.locale)}</span>
            </div>
        `;
    }

    function metricBox(title, value, status, help = '') {
        return `
            <div class="metric-box ${status.class}">
                <div class="metric-title">${title}</div>
                <div class="metric-value">${value}</div>
                <div class="metric-status">${status.label}</div>
                ${help ? `<div class="metric-help">${help}</div>` : ''}
            </div>
        `;
    }

    function sectionHeading(title, subtitle) {
        return `<h3 class="section-heading">${title}<span class="sub">${subtitle}</span></h3>`;
    }

    function verdictHtml(rating, headline, details = []) {
        const ratingClass = rating.toLowerCase();
        const ratingLabels = { excellent: 'EXCELLENT', good: 'GOOD', acceptable: 'ACCEPTABLE', poor: 'POOR', bad: 'BAD' };
        
        return `
            <div class="verdict verdict-${ratingClass}">
                <div class="verdict-rating">${ratingLabels[rating] || 'UNKNOWN'}</div>
                <div class="verdict-headline">${headline}</div>
                ${details.length ? '<div class="verdict-details">' + details.map(d => `<div>• ${d}</div>`).join('') + '</div>' : ''}
            </div>
        `;
    }

    /* ============================================================
       RENDER FUNCTIONS
       ============================================================ */
    function renderVerdict(data) {
        const q = data.quality || {};
        const kb = data.knowledge_base || {};
        
        let issues = [];
        let rating = 'good';

        // Precision check
        if (q.overall_precision < 0.85) { issues.push('Precision unter 85%'); rating = 'poor'; }
        else if (q.overall_precision < 0.95) { rating = 'acceptable'; }

        // F0.5 check
        if (q.overall_f05 < 0.80) { issues.push('F0.5 unter 80%'); rating = 'poor'; }
        else if (q.overall_f05 < 0.90) { rating = rating === 'good' ? 'acceptable' : rating; }

        // KB Status
        if (kb.status !== 'OK') { issues.push(`Knowledge Base: ${kb.status}`); if (rating === 'good') rating = 'acceptable'; }
        
        // Latency check
        if ((data.latency?.avg_response_time_ms || 0) > 240) { issues.push('Hohe Latenz (>240ms)'); }

        const headlines = {
            excellent: 'System läuft optimal',
            good: 'Solide Qualität ohne kritische Probleme',
            acceptable: 'Bedienbar, aber Verbesserungen empfohlen',
            poor: 'Kritische Qualitätsprobleme erkannt',
            bad: 'Dringender Handlungsbedarf'
        };

        el('verdict').innerHTML = verdictHtml(rating, headlines[rating] || '', issues);
    }

    function renderQuality(data) {
        const q = data.quality || {};
        const body = el('qualityBody');
        
        const p = q.overall_precision || 0;
        const r = q.overall_correctness_recall || 0;
        const f = q.overall_f05 || 0;

        body.innerHTML = `
            ${sectionHeading('Service Quality', 'Nach dem Gate')}
            <div class="metrics-grid">
                ${metricBox('Precision', fmtPercent(p, 2), getStatus(p, THRESHOLDS.precision), 'Anteil korrekter Antworten')}
                ${metricBox('Correctness Recall', fmtPercent(r, 2), getStatus(r, THRESHOLDS.correctnessRecall), 'Fähigkeit, richtige Antworten zu finden')}
                ${metricBox('F0.5 Score', fmtPercent(f, 2), getStatus(f, THRESHOLDS.f05), 'Gewichtete Kombination (Precision wichtiger)')}
            </div>
        `;
    }

    function renderOperations(data) {
        const lat = data.latency || {};
        const res = data.resources || {};
        const chatbot = {
            cpu_avg_pct: res.chatbot_avg_cpu,
            mem_avg_mb: res.chatbot_avg_mem_mb
        };
        const ollama = {
            cpu_avg_pct: res['ollama_llama-server_exe_avg_cpu'],
            mem_avg_mb: res['ollama_llama-server_exe_avg_mem_mb']
        };

        const body = el('opsBody');

        body.innerHTML = `
            ${sectionHeading('Latenz', 'Antwortzeiten')}
            <div class="metrics-grid" style="grid-template-columns: repeat(3, 1fr);">
                ${metricBox('Avg', fmtMs(lat.avg_response_time_ms || 0), getStatus(lat.avg_response_time_ms, THRESHOLDS.latencyAvg))}
                ${metricBox('P95', fmtMs(lat.p95_response_time_ms || 0), getStatus(lat.p95_response_time_ms, THRESHOLDS.latencyP95))}
                ${metricBox('Request Throughput', fmtNum(lat.throughput_requests_per_sec || 0) + '/s', { class: 'neutral', label: '-' })}
            </div>

            ${sectionHeading('Ressourcen', 'CPU & Memory')}
            <div class="resource-grid">
                <div class="resource-block">
                    <div class="resource-title">Chatbot</div>
                    ${barLabel('CPU (%)', (chatbot.cpu_avg_pct || 0).toFixed(1), 100, getStatus(chatbot.cpu_avg_pct, THRESHOLDS.procCpu))}
                    ${barLabel('Memory (MB)', (chatbot.mem_avg_mb || 0).toFixed(1), 8192, { class: 'neutral', label: '-' })}
                </div>
                <div class="resource-block">
                    <div class="resource-title">Ollama</div>
                    ${barLabel('CPU (%)', (ollama.cpu_avg_pct || 0).toFixed(1), 100, getStatus(ollama.cpu_avg_pct, THRESHOLDS.procCpu))}
                    ${barLabel('Memory (MB)', (ollama.mem_avg_mb || 0).toFixed(1), 8192, { class: 'neutral', label: '-' })}
                </div>
            </div>
        `;
    }

    function renderLocalization(data) {
        const ea = data.quality?.error_attribution || {};
        const total = (ea.retrieval || 0) + (ea.reranker || 0) + (ea.gate || 0) || 1;

        const body = el('locBody');
        
        const pctRetrieval = ((ea.retrieval || 0) / total * 100).toFixed(1);
        const pctReranker = ((ea.reranker || 0) / total * 100).toFixed(1);
        const pctGate = ((ea.gate || 0) / total * 100).toFixed(1);

        body.innerHTML = `
            ${sectionHeading('Error Attribution', 'Wo werden Antworten falsch aussortiert?')}
            <div class="attribution-bars">
                <div class="attribution-bar">
                    <div class="ab-fill retrieval" style="height:${pctRetrieval}%"></div>
                    <div class="ab-label">${pctRetrieval}%<br><small>Retrieval</small></div>
                </div>
                <div class="attribution-bar">
                    <div class="ab-fill reranker" style="height:${pctReranker}%"></div>
                    <div class="ab-label">${pctReranker}%<br><small>Reranker</small></div>
                </div>
                <div class="attribution-bar">
                    <div class="ab-fill gate" style="height:${pctGate}%"></div>
                    <div class="ab-label">${pctGate}%<br><small>Gate</small></div>
                </div>
            </div>
        `;
    }

    function renderCoverage(data) {
        const q = data.quality || {};
        const body = el('covBody');

        body.innerHTML = `
            ${sectionHeading('Top-K Coverage', 'Wie oft ist die richtige FAQ in den Chroma-Kandidaten?')}
            <div class="coverage-grid">
                ${metricBox('Top-3', fmtPercent(q.retrieval_top3_coverage || 0, 2), getStatus(q.retrieval_top3_coverage, THRESHOLDS.coverage))}
                ${metricBox('Top-5', fmtPercent(q.retrieval_top5_coverage || 0, 2), getStatus(q.retrieval_top5_coverage, THRESHOLDS.coverage))}
                ${metricBox('Top-10', fmtPercent(q.retrieval_top10_coverage || 0, 2), getStatus(q.retrieval_top10_coverage, THRESHOLDS.coverage))}
            </div>
            <p class="coverage-note">
                <strong>Hinweis:</strong> Wenn Top-5 hoch aber Precision niedrig ist, ist Ranking das Problem.
            </p>
        `;
    }

    function renderPipeline(data) {
        const q = data.quality || {};
        const body = el('pipeBody');

        body.innerHTML = `
            ${sectionHeading('Stage Performance', 'Genauigkeit jeder Pipeline-Stage')}
            <div class="metrics-grid" style="grid-template-columns: repeat(2, 1fr);">
                ${metricBox('Retrieval Accuracy', fmtPercent(q.retrieval_accuracy || 0, 2), getStatus(q.retrieval_accuracy, { good: 0.90, warn: 0.80, dir: 'up' }))}
                ${metricBox('Reranker Accuracy', fmtPercent(q.reranker_accuracy || 0, 2), getStatus(q.reranker_accuracy, { good: 0.90, warn: 0.80, dir: 'up' }))}
            </div>

            ${sectionHeading('Gate Comparison', 'Nutzen des Rerankers')}
            <div class="comparison-table">
                <div class="comp-row">
                    <span class="comp-label">Precision</span>
                    <span class="comp-val chroma">${fmtPercent(q.gate_chroma_precision || 0, 2)}</span>
                    <span class="comp-arrow">${q.gate_precision > q.gate_chroma_precision ? '↑' : '→'}</span>
                    <span class="comp-val reranker">${fmtPercent(q.gate_precision || 0, 2)}</span>
                </div>
                <div class="comp-row">
                    <span class="comp-label">Recall</span>
                    <span class="comp-val chroma">${fmtPercent(q.gate_chroma_correctness_recall || 0, 2)}</span>
                    <span class="comp-arrow">${q.gate_correctness_recall > q.gate_chroma_correctness_recall ? '↑' : '→'}</span>
                    <span class="comp-val reranker">${fmtPercent(q.gate_correctness_recall || 0, 2)}</span>
                </div>
                <div class="comp-row">
                    <span class="comp-label">F0.5</span>
                    <span class="comp-val chroma">${fmtPercent(q.gate_chroma_f05 || 0, 2)}</span>
                    <span class="comp-arrow">${q.gate_f05 > q.gate_chroma_f05 ? '↑' : '→'}</span>
                    <span class="comp-val reranker">${fmtPercent(q.gate_f05 || 0, 2)}</span>
                </div>
            </div>
        `;
    }

    function renderProblemAreas(data) {
        const catsObj = data.quality?.categories || {};
        const cats = Object.entries(catsObj).map(([name, c]) => ({
            name,
            success_rate: c.overall_success || 0,
            test_count: c.count || 0
        }));

        if (!cats.length) {
            el('areasBody').innerHTML = '<p class="empty-state">Keine Kategoriedaten verfügbar</p>';
            return;
        }

        // Sortieren nach schlechtester Erfolgsquote
        const worst = [...cats].sort((a, b) => (a.success_rate || 1) - (b.success_rate || 1)).slice(0, 5);
        const body = el('areasBody');

        body.innerHTML = `
            ${sectionHeading('Bottom 5 Kategorien', 'Schlechteste Erfolgsraten')}
            <div class="problem-list">
                ${worst.map(cat => {
                    const sr = cat.success_rate || 0;
                    const status = sr >= 0.85 ? 'good' : sr >= 0.60 ? 'warn' : 'bad';
                    const barColor = sr >= 0.85 ? '#10b981' : sr >= 0.60 ? '#f59e0b' : '#ef4444';
                    return `
                        <div class="problem-item ${status}">
                            <div class="pi-name">${cat.name}</div>
                            <div class="pi-stat">Erfolgsrate ${fmtPercent(sr, 1)}</div>
                            <div class="pi-progress">
                                <div class="pp-bar" style="width:${sr * 100}%; background: ${barColor}"></div>
                            </div>
                            <div class="pi-count">${cat.test_count || '-'} Tests</div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    }

    function renderKBHealth(data) {
        const kb = data.knowledge_base || {};
        const body = el('kbBody');

        const statusMap = {
            'OK': { class: 'good', msg: 'Keine Probleme erkannt' },
            'WARNING': { class: 'warn', msg: 'Einige Probleme gefunden' },
            'FAIL': { class: 'bad', msg: 'Kritische Probleme im Knowledge Base' }
        };
        const s = statusMap[kb.status || 'WARNING'] || statusMap['WARNING'];

        body.innerHTML = `
            <div class="kb-header ${s.class}">
                <div class="kb-status-badge">${s.msg}</div>
            </div>
            <div class="metrics-grid" style="grid-template-columns: repeat(4, 1fr);">
                ${metricBox('Duplicate Pairs', kb.duplicate_pairs || 0, { class: 'neutral', label: '-' })}
                ${metricBox('Affected Entries', kb.affected_entries || 0, { class: 'neutral', label: '-' })}
                ${metricBox('Low Margin Entries', kb.low_margin_entries || 0, { class: 'neutral', label: '-' })}
                ${metricBox('Avg Margin', fmtPercent(kb.avg_margin || 0, 3), getStatus(kb.avg_margin, { good: 0.25, warn: 0.15, dir: 'up' }))}
            </div>
        `;
    }

    function renderTechnical(data) {
        const q = data.quality || {};
        const body = el('techBody');

        body.innerHTML = `
            ${sectionHeading('Margin Analysis', 'Trennschärfe der Scores')}
            <div class="metrics-grid" style="grid-template-columns: repeat(3, 1fr);">
                ${metricBox('Chroma Margin Separation', fmtPercent(q.chroma_margin_separation || 0, 3), { class: 'neutral', label: '-' })}
                ${metricBox('Reranker Margin Separation', fmtPercent(q.reranker_margin_separation || 0, 3), { class: 'neutral', label: '-' })}
                ${metricBox('Margin Separation Gain', fmtPercent(q.margin_separation_gain || 0, 3), { class: 'neutral', label: '-' })}
            </div>

            ${sectionHeading('Gate Details', 'Gate Metriken')}
            <div class="metrics-grid" style="grid-template-columns: repeat(3, 1fr);">
                ${metricBox('Gate Precision', fmtPercent(q.gate_precision || 0, 3), { class: 'neutral', label: '-' })}
                ${metricBox('Gate Recall', fmtPercent(q.gate_correctness_recall || 0, 3), { class: 'neutral', label: '-' })}
                ${metricBox('Gate F0.5', fmtPercent(q.gate_f05 || 0, 3), { class: 'neutral', label: '-' })}
            </div>
        `;
    }

    /* ============================================================
       MAIN LOAD
       ============================================================ */
    async function loadDashboard() {
        const stateLoading = el('stateLoading');
        const stateError = el('stateError');
        const app = el('app');
        const refreshBtn = el('refreshBtn');

        show(stateLoading);
        hide(stateError);
        if (app) app.hidden = true;

        const urlParams = new URLSearchParams(window.location.search);
        const dataPath = urlParams.get('data') || CONFIG.primaryPath;

        try {
            const response = await fetch(dataPath);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            
            renderVerdict(data);
            renderQuality(data);
            renderOperations(data);
            renderLocalization(data);
            renderCoverage(data);
            renderPipeline(data);
            renderProblemAreas(data);
            renderKBHealth(data);
            renderTechnical(data);

            // Metadata
            const runId = data.run_id || '-';
            const genAt = fmtDate(data.generated_at || new Date().toISOString());
            text('meta', `Run ID: ${runId} • Generated: ${genAt}`);
            text('footMeta', `Generated: ${genAt} • Run ID: ${runId}`);

            hide(stateLoading);
            show(app);

        } catch (error) {
            console.error('Dashboard error:', error);
            
            // Fallback versuchen
            if (dataPath !== CONFIG.fallbackPath) {
                try {
                    const retry = await fetch(CONFIG.fallbackPath);
                    if (retry.ok) {
                        const data = await retry.json();

                        renderVerdict(data);
                        renderQuality(data);
                        renderOperations(data);
                        renderLocalization(data);
                        renderCoverage(data);
                        renderPipeline(data);
                        renderProblemAreas(data);
                        renderKBHealth(data);
                        renderTechnical(data);

                        const runId = data.run_id || '-';
                        const genAt = fmtDate(data.generated_at || new Date().toISOString());
                        text('meta', `Run ID: ${runId} • Generated: ${genAt} (Fallback)`);
                        text('footMeta', `Generated: ${genAt} • Run ID: ${runId} (Fallback)`);

                        hide(stateLoading);
                        show(app);
                        return;
                    }
                } catch (fallbackError) {
                    console.error('Fallback error:', fallbackError);
                }
            }

            hide(stateLoading);
            show(stateError);
            el('stateError').textContent = `❌ Fehler beim Laden: ${error.message}`;
        }
    }

    // Init
    document.addEventListener('DOMContentLoaded', loadDashboard);
    el('refreshBtn')?.addEventListener('click', () => window.location.reload());

})();
