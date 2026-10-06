// ============================================
// KONFIGURATION
// ============================================
const METRICS_PATH = './metrics.json';

// ============================================
// CHART INSTANZEN
// ============================================
let categoryChart = null;
let verdictChart = null;
let latencyChart = null;

// ============================================
// DATA PARSING & TRANSFORMATION
// ============================================
function transformData(metrics) {
  return {
    meta: {
      runId: metrics.run_id,
      generatedAt: new Date(metrics.generated_at).toLocaleString('de-DE'),
      filesAnalyzed: metrics.files_analyzed
    },
    kpis: {
      successRate: (metrics.quality.overall_success_rate * 100).toFixed(1),
      avgResponseTime: metrics.latency.avg_response_time_ms.toFixed(1),
      medianResponseTime: metrics.latency.median_response_time_ms.toFixed(1),
      p95ResponseTime: metrics.latency.p95_response_time_ms.toFixed(1),
      f1Score: metrics.quality.gate_f1_score.toFixed(3),
      precision: metrics.quality.gate_precision.toFixed(3),
      recall: metrics.quality.gate_recall.toFixed(3),
      healthScore: metrics.composite_health_score.toFixed(1)
    },
    categories: Object.entries(metrics.quality.categories || {}).map(([name, data]) => ({
      name,
      total: data.total,
      successRate: (data.success_rate * 100).toFixed(1)
    })),
    verdicts: Object.entries(metrics.quality.verdict_distribution || {}).map(([name, count]) => ({
      name,
      count
    })),
    latency: {
      min: metrics.latency.min_response_time_ms.toFixed(1),
      median: metrics.latency.median_response_time_ms.toFixed(1),
      p95: metrics.latency.p95_response_time_ms.toFixed(1),
      p99: metrics.latency.p99_response_time_ms.toFixed(1),
      max: metrics.latency.max_response_time_ms.toFixed(1)
    },
    resources: parseResources(metrics.resources || {})
  };
}

function parseResources(resourcesRaw) {
    const result = {};
    
    for (const [key, value] of Object.entries(resourcesRaw)) {
      // Keys sind im Format: processName_metricType (z.B. "chatbot_avg_cpu")
      if (key.endsWith('_avg_cpu')) {
        const processName = key.replace('_avg_cpu', '');
        if (!result[processName]) result[processName] = {};
        result[processName].avgCpu = value;
      }
      else if (key.endsWith('_peak_cpu')) {
        const processName = key.replace('_peak_cpu', '');
        if (!result[processName]) result[processName] = {};
        result[processName].peakCpu = value;
      }
      else if (key.endsWith('_avg_mem_mb')) {
        const processName = key.replace('_avg_mem_mb', '');
        if (!result[processName]) result[processName] = {};
        result[processName].avgMem = value;
      }
      else if (key.endsWith('_peak_mem_mb')) {
        const processName = key.replace('_peak_mem_mb', '');
        if (!result[processName]) result[processName] = {};
        result[processName].peakMem = value;
      }
    }
    
    return result;
  }

// ============================================
// UI UPDATES
// ============================================
function updateMetadata(meta) {
  document.getElementById('metadata').textContent = 
    `Run ID: ${meta.runId} • Generiert: ${meta.generatedAt}`;
  document.getElementById('generated-at').textContent = meta.generatedAt;
  document.getElementById('run-id').textContent = meta.runId;
  document.getElementById('current-date').textContent = 
    new Date().toLocaleString('de-DE');
}

function updateKPIS(kpis) {
  // Success Rate
  const srEl = document.getElementById('success-rate');
  srEl.textContent = `${kpis.successRate}%`;
  srEl.className = 'kpi-value ' + (kpis.successRate >= 80 ? 'success' : 'warning');

  // Latency
  document.getElementById('response-time').textContent = `${kpis.avgResponseTime} ms`;
  document.getElementById('p95-latency').textContent = `${kpis.p95ResponseTime} ms`;

  // F1 Score
  document.getElementById('f1-score').textContent = kpis.f1Score;
  document.getElementById('precision').textContent = kpis.precision;

  // Health Score
  document.getElementById('health-score').textContent = kpis.healthScore;
  document.getElementById('health-status').className = 
    'kpi-delta ' + (kpis.healthScore >= 70 ? 'positive' : 'warning');
}

function updateResourceTable(resources) {
    console.log('updateResourceTable called with keys:', Object.keys(resources));
    
    const tbody = document.querySelector('#resourceTable tbody');
    if (!tbody) {
      console.error('Resource table tbody not found!');
      return;
    }
    
    tbody.innerHTML = '';  // ALLE alten Zeilen löschen
    
    for (const [process, data] of Object.entries(resources)) {
      console.log(`Processing ${process}:`, data);
      
      const row = document.createElement('tr');
      
      // Safe-Zugriff mit default
      const avgCpu = data.avgCpu !== undefined ? data.avgCpu.toFixed(1) : '--';
      const peakCpu = data.peakCpu !== undefined ? data.peakCpu.toFixed(1) : '--';
      const avgMem = data.avgMem !== undefined ? data.avgMem.toFixed(1) : '--';
      const peakMem = data.peakMem !== undefined ? data.peakMem.toFixed(1) : '--';
      
      row.innerHTML = `
        <td>${process}</td>
        <td>${avgCpu}</td>
        <td>${peakCpu}</td>
        <td>${avgMem}</td>
        <td>${peakMem}</td>
      `;
      tbody.appendChild(row);
      
      console.log(`✓ Added row for ${process}`);
    }
    
    console.log(`Total rows added: ${tbody.querySelectorAll('tr').length}`);
  }

// ============================================
// CHART INITIALIZATION
// ============================================
function initCategoryChart(categories) {
  const ctx = document.getElementById('categoryChart').getContext('2d');
  
  const sortedCategories = [...categories].sort((a, b) => b.successRate - a.successRate);
  const labels = sortedCategories.map(c => c.name);
  const data = sortedCategories.map(c => c.successRate);

  categoryChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Erfolgsrate (%)',
        data: data,
        backgroundColor: data.map(val => val >= 80 ? '#4caf50' : val >= 50 ? '#ff9800' : '#f44336'),
        borderColor: data.map(val => val >= 80 ? '#43a047' : val >= 50 ? '#fb8c00' : '#e53935'),
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (context) => context[0].label,
            label: (context) => `Erfolgsrate: ${context.raw}% (${sortedCategories[context.dataIndex].total} Tests)`
          }
        }
      },
      scales: {
        x: {
          ticks: {
            autoSkip: false,
            maxRotation: 45,
            minRotation: 45,
            font: { size: 10 }
          }
        },
        y: {
          beginAtZero: true,
          max: 100,
          ticks: { callback: (val) => val + '%' }
        }
      }
    }
  });
}

function initVerdictChart(verdicts) {
  const ctx = document.getElementById('verdictChart').getContext('2d');
  
  const labels = verdicts.map(v => v.name.replace(/_/g, ' '));
  const data = verdicts.map(v => v.count);

  verdictChart = new Chart(ctx, {
    type: 'pie',
    data: {
      labels: labels,
      datasets: [{
        data: data,
        backgroundColor: ['#4caf50', '#2196f3', '#ff9800', '#f44336', '#9c27b0', '#795548'],
        hoverOffset: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            padding: 10,
            font: { size: 11 }
          }
        },
        tooltip: {
          callbacks: {
            label: (context) => {
              const total = data.reduce((a, b) => a + b, 0);
              const percentage = ((context.raw / total) * 100).toFixed(1);
              return `${context.label}: ${context.raw} (${percentage}%)`;
            }
          }
        }
      }
    }
  });
}

function initLatencyChart(latency) {
  const ctx = document.getElementById('latencyChart').getContext('2d');
  
  const labels = ['Min', 'Median', 'P95', 'P99', 'Max'];
  const data = [latency.min, latency.median, latency.p95, latency.p99, latency.max];

  latencyChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Antwortzeit (ms)',
        data: data,
        backgroundColor: ['#90caf9', '#64b5f6', '#42a5f5', '#2196f3', '#1976d2'],
        borderColor: '#1565c0',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context) => `${context.raw} ms`
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: { callback: (val) => val + ' ms' }
        }
      }
    }
  });
}

// ============================================
// MAIN LOAD FUNCTION
// ============================================
async function loadDashboard() {
  try {
    const response = await fetch(METRICS_PATH);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const metrics = await response.json();
    const transformed = transformData(metrics);
    
    // DEBUG: Resources ausgeben
    console.log('=== PARSE DEBUG ===');
    console.log('Original JSON resources:', metrics.resources);
    console.log('Parsed resources:', transformed.resources);
    console.log('===================');

    // UI Update
    updateMetadata(transformed.meta);
    updateKPIS(transformed.kpis);
    updateResourceTable(transformed.resources);
    
    initCategoryChart(transformed.categories);
    initVerdictChart(transformed.verdicts);
    initLatencyChart(transformed.latency);
    
    // States umschalten
    document.getElementById('loading').style.display = 'none';
    document.getElementById('error').style.display = 'none';
    document.getElementById('dashboard-content').style.display = 'block';
    
  } catch (error) {
    console.error('Dashboard Fehler:', error);
    document.getElementById('loading').style.display = 'none';
    document.getElementById('error').style.display = 'block';
    document.getElementById('error').textContent = 
      `Fehler beim Laden von metrics.json: ${error.message}`;
  }
}

// ============================================
// INITIALIZATION
// ============================================
document.addEventListener('DOMContentLoaded', loadDashboard);

// Optional: Manuelles Reload (Entwickler-Console)
window.reloadDashboard = loadDashboard;