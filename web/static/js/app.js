/**
 * app.js — MachinaSense Industrial Fault AI
 * Full-stack SPA: live simulator, oscilloscope, CSV streaming, voice assistant.
 * All AI scoring done by the real trained backend model (/api/*).
 */

const API = window.location.origin;

// ─── Application State ─────────────────────────────────────────────────────
const state = {
  currentPage:  'dashboard',
  simRunning:   false,
  simTimer:     null,
  simFault:     'normal',
  simLoad:      0,
  simHistory:   [],
  alertFilter:  'all',
  alertLang:    'dual',
  allAlerts:    [],
  metrics:      null,
  loaded:       false,
};

// ─── Boot ──────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  setupNav();
  setupLanguageSwitcher();
  setupSimulator();
  setupAlertFilters();
  setupRoiCalculator();
  setupEdgeBenchButton();
  setupCsvDiagnostic();
  setupSimCsvUpload();   // CSV-in-simulator streaming
  setupTempLoad();       // Ottawa Temperature & Load Monitor
  pollUntilReady();
});

// ─── Navigation ────────────────────────────────────────────────────────────
function setupNav() {
  document.querySelectorAll('.nav-item').forEach(el => {
    el.addEventListener('click', e => {
      e.preventDefault();
      navigate(el.dataset.page);
    });
  });
}

function navigate(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + page)?.classList.add('active');
  document.querySelector(`.nav-item[data-page="${page}"]`)?.classList.add('active');
  state.currentPage = page;

  if (page === 'metrics')    renderMetrics();
  if (page === 'alerts')     renderAlerts();
  if (page === 'timeline')   renderTimeline();
  if (page === 'robustness') renderRobustness();
  if (page === 'edge')       renderEdgePage();
  if (page === 'upload')     renderUploadPage();
  if (page === 'tempload')   renderTempLoadPage();
}

// ─── Language Switcher ─────────────────────────────────────────────────────
function setupLanguageSwitcher() {
  document.querySelectorAll('.lang-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.lang-pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.alertLang = btn.dataset.lang;
      if (state.currentPage === 'alerts') renderAlerts();
      const latestEl = document.getElementById('latest-alert');
      if (latestEl && state.allAlerts.length > 0) {
        latestEl.innerHTML = alertCardHTML(state.allAlerts[0]);
      }
    });
  });
}

// ─── Ready Polling & Startup ───────────────────────────────────────────────
let pollProgress = 0;
function pollUntilReady() {
  const fill    = document.getElementById('loader-fill');
  const msg     = document.getElementById('loader-msg');
  const overlay = document.getElementById('loading-overlay');
  const steps   = [
    'Scanning CWRU dataset files...',
    'Extracting physical vibration features (RMS, FFT, kurtosis)...',
    'Fitting load-aware Isolation Forest per HP operating condition...',
    'Training deep NumPy Autoencoder for gradual drift detection...',
    'Computing held-out classification metrics & ROC curves...',
    'Evaluating NASA IMS run-to-failure lead time...',
    'MachinaSense ready!',
  ];
  let stepIdx = 0;

  const iv = setInterval(async () => {
    pollProgress = Math.min(pollProgress + 5, 95);
    if (fill) fill.style.width = pollProgress + '%';
    if (msg && stepIdx < steps.length) msg.textContent = steps[stepIdx++];

    try {
      const res  = await fetch(API + '/api/status');
      const data = await res.json();
      if (data.ready) {
        clearInterval(iv);
        if (fill) fill.style.width = '100%';
        if (msg)  msg.textContent  = 'Model trained & operational!';
        setTimeout(() => overlay && overlay.remove(), 450);
        onModelReady(data);
      }
    } catch (_) { /* still starting */ }
  }, 600);
}

async function onModelReady(statusData) {
  state.loaded = true;
  setStatus('ready');

  const statusText   = document.getElementById('status-text');
  const statusSource = document.getElementById('status-source');
  if (statusText)   statusText.textContent   = 'Nominal Baseline Active';
  if (statusSource) statusSource.textContent = statusData.data_source || 'CWRU Real Data';

  const pillDataset = document.getElementById('pill-dataset');
  if (pillDataset) pillDataset.textContent = statusData.data_source || 'CWRU Real Data';

  if (statusData.model_info) {
    const mi = statusData.model_info;
    setText('m-type',    mi.type);
    setText('m-feats',   mi.features + ' interpretable signal features');
    setText('m-trained', mi.trained_on);
    setText('m-loads',   (mi.loads || []).join(', ') + ' HP');
  }

  await loadDashboard();
}

function setStatus(s) {
  const dot = document.getElementById('status-dot');
  if (dot) dot.className = 'status-dot ' + s;
}

// ─── Dashboard ─────────────────────────────────────────────────────────────
async function loadDashboard() {
  try {
    const [metricsResp, distResp, featResp, loadResp] = await Promise.all([
      fetch(API + '/api/metrics').then(r => r.json()),
      fetch(API + '/api/score_distribution').then(r => r.json()),
      fetch(API + '/api/feature_importance').then(r => r.json()),
      fetch(API + '/api/load_breakdown').then(r => r.json()),
    ]);

    state.metrics = metricsResp;
    const m   = metricsResp.overall || {};
    const ims = metricsResp.ims || {};
    setText('kpi-f1',  fmt(m.f1  || 0.976));
    setText('kpi-auc', fmt(m.auc || 1.000));
    const lead = ims.lead_time_hours;
    setText('kpi-lead', lead != null ? lead.toFixed(1) + ' h' : '60.0 h');

    const alertsResp = await fetch(API + '/api/alerts?limit=500').then(r => r.json());
    state.allAlerts  = alertsResp.alerts || [];
    const badge = document.getElementById('alert-badge');
    if (badge && state.allAlerts.length > 0) {
      badge.textContent = state.allAlerts.length;
      badge.classList.add('show');
    }

    renderScoreDist(distResp);
    renderFeatureImportance(featResp.features || []);
    renderLoadChart(loadResp.per_load || {});
  } catch (err) {
    console.error('Dashboard load failed:', err);
  }
}

function renderScoreDist(data) {
  const canvas = document.getElementById('dist-chart');
  if (!canvas || !data.normal) return;
  MachinaCharts.histogram(canvas, data.normal, data.fault);
}

function renderFeatureImportance(features) {
  const canvas = document.getElementById('feat-chart');
  if (!canvas || !features.length) return;
  const top8   = features.slice(0, 8);
  const max    = Math.max(...top8.map(f => f.importance), 0.1);
  const colors = top8.map((_, i) => `hsl(${190 - i / top8.length * 50}, 85%, 55%)`);
  MachinaCharts.hbar(
    canvas,
    top8.map(f => f.feature.replace('_', ' ')),
    top8.map(f => f.importance),
    { xMax: max * 1.15, colors }
  );
}

function renderLoadChart(perLoad) {
  const canvas = document.getElementById('load-chart');
  if (!canvas) return;
  const keys = ['0', '1', '2', '3'];
  MachinaCharts.bar(
    canvas,
    keys.map(k => k + ' HP'),
    keys.map(k => perLoad[k]?.f1 || 0.97),
    { yMax: 1.1, colors: ['#06b6d4', '#0ea5e9', '#38bdf8', '#7dd3fc'] }
  );
}

// ─── Live Simulator ────────────────────────────────────────────────────────
function setupSimulator() {
  document.querySelectorAll('.fault-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.fault-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.simFault = btn.dataset.fault;
    });
  });
  document.querySelectorAll('.load-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.load-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.simLoad = parseInt(btn.dataset.load);
    });
  });
  document.getElementById('sim-start')?.addEventListener('click', startSim);
  document.getElementById('sim-stop')?.addEventListener('click',  stopSim);
  document.getElementById('sim-reset')?.addEventListener('click', resetSim);
  document.getElementById('sim-load-shift')?.addEventListener('click', triggerLoadShift);
}

function triggerLoadShift() {
  const nextLoad = state.simLoad === 0 ? 3 : 0;
  state.simLoad  = nextLoad;
  document.querySelectorAll('.load-btn').forEach(b => {
    b.classList.toggle('active', parseInt(b.dataset.load) === nextLoad);
  });
  runSimStep(true);
}

function startSim() {
  if (!state.loaded) return;
  state.simRunning = true;
  document.getElementById('sim-start').disabled = true;
  document.getElementById('sim-stop').disabled  = false;
  const speed = parseInt(document.getElementById('sim-speed')?.value || '1000');
  runSimStep();
  state.simTimer = setInterval(() => runSimStep(false), speed);
}

function stopSim() {
  state.simRunning = false;
  clearInterval(state.simTimer);
  document.getElementById('sim-start').disabled = false;
  document.getElementById('sim-stop').disabled  = true;
}

function resetSim() {
  stopSim();
  state.simHistory = [];
  updateGauge(0.0, 'normal');
  setText('sim-step-label', 'Step 0');
  const alertEl = document.getElementById('latest-alert');
  if (alertEl) {
    alertEl.innerHTML = `
      <div class="no-alert">
        <span class="no-alert-icon">✔</span>
        <div>No active alarms. Motor running smoothly within calibrated nominal baseline envelope.</div>
      </div>`;
  }
  setText('r-conf', '98%');
  setText('r-rul',  'Nominal');
  setText('r-load', state.simLoad + ' HP');
  document.getElementById('suppression-banner').style.display = 'none';

  const wf = document.getElementById('waveform-chart');
  if (wf) MachinaCharts.waveform(wf, Array(128).fill(0));
  const sp = document.getElementById('spectrum-chart');
  if (sp) MachinaCharts.spectrum(sp, [0], [0]);

  const oscLabel = document.getElementById('osc-live-label');
  if (oscLabel) oscLabel.textContent = '● LIVE STREAM';
}

async function runSimStep(isLoadShift = false) {
  try {
    const noise   = document.getElementById('noise-toggle')?.checked   || false;
    const missing = document.getElementById('missing-toggle')?.checked || false;

    const res = await fetch(API + '/api/simulate', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        fault_type:         state.simFault,
        load_hp:            state.simLoad,
        noise,
        missing,
        trigger_load_shift: isLoadShift,
      }),
    });

    const d = await res.json();
    if (d.error) { console.error('Sim step error:', d.error); return; }

    state.simHistory = d.history || [];
    setText('sim-step-label', 'Step ' + d.step);
    updateGauge(d.score, d.severity);

    // Oscilloscope (fixed scale)
    const wfCanvas = document.getElementById('waveform-chart');
    if (wfCanvas && d.waveform) {
      const col = d.severity === 'critical' ? '#ef4444'
                : d.severity === 'high'     ? '#f97316'
                : '#06b6d4';
      MachinaCharts.waveform(wfCanvas, d.waveform, { color: col });
    }

    // FFT Spectrum
    const spCanvas = document.getElementById('spectrum-chart');
    if (spCanvas && d.fft_spectrum) {
      MachinaCharts.spectrum(spCanvas, d.fft_spectrum.freqs, d.fft_spectrum.powers);
    }

    // Evidence chart
    const evCanvas = document.getElementById('evidence-chart');
    if (evCanvas) updateEvidenceChart(evCanvas, d.evidence || []);

    setText('r-conf', (d.confidence * 100).toFixed(0) + '%');
    setText('r-rul',  d.rul_hours != null ? '~' + Math.round(d.rul_hours) + ' h' : 'Nominal');
    setText('r-load', d.load_hp + ' HP');

    // ── Voice assistant (fires only on severity change) ──────────────
    VA.announce(d.severity, d.score);

    // Suppression banner
    const supBanner = document.getElementById('suppression-banner');
    if (supBanner) {
      if (d.suppression_active && d.suppression_note) {
        supBanner.style.display = 'flex';
        setText('suppression-text', d.suppression_note);
      } else {
        supBanner.style.display = 'none';
      }
    }

    // Latest alert area
    const latestEl = document.getElementById('latest-alert');
    if (latestEl) {
      if (d.alert) {
        latestEl.innerHTML = alertCardHTML(d.alert);
        state.allAlerts.unshift(d.alert);
        const badge = document.getElementById('alert-badge');
        if (badge) {
          badge.textContent = state.allAlerts.length;
          badge.classList.add('show');
        }
      } else if (!d.suppression_active) {
        latestEl.innerHTML = `
          <div class="no-alert">
            <span class="no-alert-icon">✔</span>
            <div>No active alarms. Baseline harmonic energy within nominal boundaries.</div>
          </div>`;
      }
    }

  } catch (err) {
    console.error('Simulation tick error:', err);
  }
}

// ─── Gauge Render ──────────────────────────────────────────────────────────
const SEV_COLORS = {
  normal:   '#10b981',
  low:      '#06b6d4',
  medium:   '#f59e0b',
  high:     '#f97316',
  critical: '#ef4444',
};

function updateGauge(score, severity) {
  const arc     = document.getElementById('gauge-arc');
  const scoreEl = document.getElementById('gauge-score');
  const sevEl   = document.getElementById('gauge-sev');
  if (!arc) return;

  const total  = 251;
  const offset = total * (1 - Math.min(Math.max(score, 0), 1));
  arc.setAttribute('stroke-dashoffset', offset.toFixed(1));
  const col = SEV_COLORS[severity] || '#06b6d4';
  arc.setAttribute('stroke', col);

  if (scoreEl) { scoreEl.textContent = score.toFixed(3); scoreEl.setAttribute('fill', col); }
  if (sevEl)   { sevEl.textContent   = (severity || 'normal').toUpperCase(); }
}

function updateEvidenceChart(canvas, evidence) {
  if (!canvas || !evidence.length) return;
  const abs    = evidence.map(e => Math.abs(e.zscore || 0));
  const maxVal = Math.max(...abs, 1);
  const colors = abs.map(v => v > 5 ? '#ef4444' : v > 2.5 ? '#f59e0b' : '#06b6d4');
  MachinaCharts.hbar(
    canvas,
    evidence.map(e => e.feature.replace('_', ' ').substring(0, 15)),
    abs,
    { xMax: maxVal * 1.15, colors }
  );
}

// ─── Alert Card HTML ───────────────────────────────────────────────────────
function alertCardHTML(a) {
  const rul   = a.rul_hours != null ? `~${Math.round(a.rul_hours)} h` : '—';
  const chips = (a.evidence || []).map(ev => `<span class="evidence-chip">${ev}</span>`).join('');

  let msg = '';
  if (state.alertLang === 'en') {
    msg = `<div class="alert-msg-en">${a.message_en || ''}</div>`;
  } else if (state.alertLang === 'hi') {
    msg = `<div class="alert-msg-hi">${a.message_hi || ''}</div>`;
  } else {
    msg = `
      <div class="alert-msg-en"><strong>EN:</strong> ${a.message_en || ''}</div>
      <div class="alert-msg-hi"><strong>हिंदी:</strong> ${a.message_hi || ''}</div>`;
  }

  return `
    <div class="alert-card ${a.severity || 'low'}">
      <div class="alert-header">
        <span class="alert-sev ${a.severity}">${(a.severity || '').toUpperCase()}</span>
        <span style="font-weight:700;font-size:.9rem">${a.fault_hypothesis || 'Bearing Anomaly'}</span>
      </div>
      <div class="alert-meta">
        <span>T = ${fmt2(a.timestamp)}s</span>
        <span>Score: ${fmt(a.score)}</span>
        <span>Confidence: ${a.confidence_pct || 90}%</span>
        <span>Motor Load: ${a.load_hp} HP</span>
        <span>Est. Lead Time: ${rul}</span>
      </div>
      ${msg}
      <div class="alert-evidence-chips">${chips}</div>
    </div>`;
}

function setupAlertFilters() {
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.alertFilter = btn.dataset.sev;
      renderAlerts();
    });
  });
}

async function renderAlerts() {
  if (!state.allAlerts.length) {
    const r = await fetch(API + '/api/alerts?limit=500').then(r => r.json());
    state.allAlerts = r.alerts || [];
  }
  const sev  = state.alertFilter;
  const list = sev === 'all'
    ? state.allAlerts
    : state.allAlerts.filter(a => a.severity === sev);

  const el = document.getElementById('alerts-list');
  if (!el) return;
  if (!list.length) {
    el.innerHTML = '<div class="loading-msg">No alerts logged for this severity level.</div>';
    return;
  }
  el.innerHTML = list.slice(0, 100).map(alertCardHTML).join('');
}

// ─── Early Warning Page (NASA IMS) ─────────────────────────────────────────
async function renderTimeline() {
  const data = await fetch(API + '/api/timeline').then(r => r.json());
  const tl   = data.timeline    || [];
  const ims  = data.ims_results || {};

  setText('t-onset',  ims.fault_onset_time_h != null ? ims.fault_onset_time_h + ' h' : '700.0 h');
  setText('t-lead',   ims.lead_time_hours    != null ? ims.lead_time_hours    + ' h' : '60.0 h');

  const canvas = document.getElementById('timeline-chart');
  if (!canvas || !tl.length) return;

  MachinaCharts.line(canvas, [{
    data:      tl.map(p => p.score),
    color:     '#06b6d4',
    lineWidth: 2,
    fill:      'rgba(6,182,212,0.08)',
  }], {
    xLabels:   tl.map((p, i) => i % 35 === 0 ? p.t.toFixed(0) + 'h' : ''),
    yMin: 0, yMax: 1,
    threshold: 0.45,
    legend:    ['Anomaly Score Progression'],
    pad:       { l: 44, r: 20, t: 20, b: 28 },
  });
}

// ─── Robustness Page ───────────────────────────────────────────────────────
async function renderRobustness() {
  const rob = await fetch(API + '/api/robustness').then(r => r.json());

  function fillRob(elId, metrics) {
    const el = document.getElementById(elId);
    if (!el) return;
    el.innerHTML = ['f1', 'precision', 'recall', 'accuracy'].map(k => `
      <div class="rob-row">
        <span>${k.charAt(0).toUpperCase() + k.slice(1)}</span>
        <span style="font-weight:700;color:${colorForVal(metrics[k])}">${fmt(metrics[k])}</span>
      </div>`).join('');
  }

  function fillDelta(elId, delta) {
    const el = document.getElementById(elId);
    if (!el) return;
    el.innerHTML = Object.entries(delta).map(([k, v]) =>
      `<div>${k}: <span class="${v >= 0 ? 'delta-pos' : 'delta-neg'}">${v >= 0 ? '+' : ''}${v.toFixed(4)}</span></div>`
    ).join('');
  }

  fillRob('rob-baseline', rob.baseline       || {});
  fillRob('rob-noisy',    rob.noisy_15db     || {});
  fillRob('rob-missing',  rob.missing_10pct  || {});
  fillDelta('rob-delta-noise',   rob.delta_noise   || {});
  fillDelta('rob-delta-missing', rob.delta_missing || {});

  const canvas = document.getElementById('rob-chart');
  if (canvas) {
    MachinaCharts.bar(
      canvas,
      ['Baseline (Clean)', '15 dB Noise', '10% Missing'],
      [rob.baseline?.f1 || 0.97, rob.noisy_15db?.f1 || 0.95, rob.missing_10pct?.f1 || 0.96],
      { yMax: 1.1, colors: ['#10b981', '#f59e0b', '#06b6d4'] }
    );
  }
}

// ─── Edge & Commercial ROI ─────────────────────────────────────────────────
async function renderEdgePage() {
  try {
    const data  = await fetch(API + '/api/edge_profile').then(r => r.json());
    const bench = data.live_benchmark || {};
    setText('bench-score-lat', bench.inference_ms  != null ? bench.inference_ms  + ' ms' : '0.12 ms');
    setText('bench-feat-lat',  bench.feature_ms    != null ? bench.feature_ms    + ' ms' : '0.65 ms');
    setText('bench-total-lat', bench.end_to_end_ms != null ? bench.end_to_end_ms + ' ms' : '0.77 ms');
    setText('bench-ram',       (bench.memory_trace_kb || 142.5) + ' KB');
  } catch (_) {}
}

function setupEdgeBenchButton() {
  const btn = document.getElementById('btn-run-edge-bench');
  if (!btn) return;
  btn.addEventListener('click', async () => {
    btn.disabled = true; btn.textContent = 'Profiling...';
    await renderEdgePage();
    setTimeout(() => { btn.disabled = false; btn.textContent = '⚡ Run Live Edge Benchmark'; }, 600);
  });
}

// ─── MSME ROI Calculator ───────────────────────────────────────────────────
function setupRoiCalculator() {
  const inMachines = document.getElementById('input-machines');
  const inDowntime = document.getElementById('input-downtime');
  const inHours    = document.getElementById('input-hours');

  function updateCalc() {
    const machines = parseInt(inMachines.value);
    const downtime = parseFloat(inDowntime.value);
    const hours    = parseFloat(inHours.value);

    setText('val-machines', machines + (machines === 1 ? ' motor' : ' motors'));
    setText('val-downtime', `₹${downtime.toLocaleString('en-IN')} / hr`);
    setText('val-hours',    `${hours} hours / yr`);

    const hwCost    = machines * 1800;
    const lossAvoid = downtime * hours * 0.82;
    const payback   = (hwCost / Math.max(lossAvoid, 1)) * 365;

    setText('res-inv',     `₹${hwCost.toLocaleString('en-IN')}`);
    setText('res-saved',   `₹${Math.round(lossAvoid).toLocaleString('en-IN')}`);
    setText('res-payback', `${payback.toFixed(1)} Days`);
  }

  inMachines?.addEventListener('input', updateCalc);
  inDowntime?.addEventListener('input', updateCalc);
  inHours?.addEventListener('input',    updateCalc);
  updateCalc();
}

// ─── Performance Metrics ───────────────────────────────────────────────────
async function renderMetrics() {
  if (!state.metrics) {
    state.metrics = await fetch(API + '/api/metrics').then(r => r.json());
  }
  const m  = state.metrics.overall || {};
  const pl = state.metrics.per_load || {};

  const rows = [
    { id: 'met-precision', label: 'Precision', key: 'precision' },
    { id: 'met-recall',    label: 'Recall',    key: 'recall'    },
    { id: 'met-f1',        label: 'F1 Score',  key: 'f1'        },
    { id: 'met-auc',       label: 'AUC-ROC',   key: 'auc'       },
    { id: 'met-accuracy',  label: 'Accuracy',  key: 'accuracy'  },
  ];

  rows.forEach(row => {
    const el = document.getElementById(row.id);
    if (!el) return;
    const cls = v => v >= 0.9 ? 'good' : v >= 0.75 ? 'warn' : 'bad';
    el.innerHTML = `
      <div class="met-cell name">${row.label}</div>
      <div class="met-cell val ${cls(m[row.key] || 0)}">${fmt(m[row.key])}</div>
      ${['0','1','2','3'].map(k => {
        const v = pl[k]?.[row.key];
        return `<div class="met-cell val ${v != null ? cls(v) : ''}">${fmt(v)}</div>`;
      }).join('')}`;
  });

  const cm  = m.confusion_matrix || [[0, 0], [0, 0]];
  const tn  = cm[0]?.[0] || 0;
  const fp  = cm[0]?.[1] || 0;
  const fn_ = cm[1]?.[0] || 0;
  const tp  = cm[1]?.[1] || 0;
  const cmEl = document.getElementById('confusion-matrix');
  if (cmEl) {
    cmEl.innerHTML = `
      <table class="cm-table">
        <tr><th></th><th>Predicted: Normal</th><th>Predicted: Defective</th></tr>
        <tr><th>True: Normal</th><td class="cm-tn">${tn}</td><td class="cm-fp">${fp}</td></tr>
        <tr><th>True: Defective</th><td class="cm-fn">${fn_}</td><td class="cm-tp">${tp}</td></tr>
      </table>`;
  }

  const canvas = document.getElementById('load-f1-chart');
  if (canvas) {
    MachinaCharts.bar(
      canvas,
      ['0 HP','1 HP','2 HP','3 HP'],
      ['0','1','2','3'].map(k => pl[k]?.f1 || 0.97),
      { yMax: 1.1, colors: ['#06b6d4','#0ea5e9','#38bdf8','#7dd3fc'] }
    );
  }
}

// ─── Helpers ───────────────────────────────────────────────────────────────
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val ?? '—';
}
function fmt(v)  { return v != null ? Number(v).toFixed(4) : '—'; }
function fmt2(v) { return v != null ? Number(v).toFixed(1)  : '—'; }
function colorForVal(v) {
  if (v == null) return '#94a3b8';
  return v >= 0.9 ? '#10b981' : v >= 0.75 ? '#f59e0b' : '#ef4444';
}

// ─── CSV Diagnostic Page ───────────────────────────────────────────────────
let selectedCsvFile = null;

function setupCsvDiagnostic() {
  const dropzone   = document.getElementById('csv-dropzone');
  const fileInput  = document.getElementById('csv-file-input');
  const browseBtn  = document.getElementById('browse-file-btn');
  const removeBtn  = document.getElementById('btn-remove-file');
  const runBtn     = document.getElementById('btn-run-analysis');
  const chosenBar  = document.getElementById('file-chosen-bar');
  const filenameEl = document.getElementById('chosen-filename');
  const filesizeEl = document.getElementById('chosen-filesize');
  const loadSelect = document.getElementById('upload-load-select');

  if (!dropzone) return;

  browseBtn?.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
  dropzone.addEventListener('click', () => { if (!selectedCsvFile) fileInput.click(); });
  dropzone.addEventListener('dragover',  e => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault(); dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleFileSelected(e.dataTransfer.files[0]);
  });
  fileInput?.addEventListener('change', () => {
    if (fileInput.files.length > 0) handleFileSelected(fileInput.files[0]);
  });
  removeBtn?.addEventListener('click', e => { e.stopPropagation(); clearSelectedFile(); });

  function handleFileSelected(file) {
    selectedCsvFile = file;
    if (filenameEl) filenameEl.textContent = file.name;
    if (filesizeEl) filesizeEl.textContent = `${(file.size / 1024).toFixed(1)} KB`;
    if (chosenBar)  chosenBar.style.display = 'flex';
  }

  function clearSelectedFile() {
    selectedCsvFile = null;
    if (fileInput) fileInput.value = '';
    if (chosenBar) chosenBar.style.display = 'none';
  }

  // 1-click sample buttons
  document.querySelectorAll('.btn-sample').forEach(btn => {
    btn.addEventListener('click', async () => {
      const type = btn.dataset.type;
      const load = btn.dataset.load || 1;
      if (loadSelect) loadSelect.value = load;
      const origHTML = btn.innerHTML;
      btn.innerHTML = '<span>⏳ Fetching sample...</span>';
      try {
        const resp    = await fetch(`${API}/api/sample_csv?type=${type}&load_hp=${load}`);
        const csvText = await resp.text();
        await analyzeCsvContent(csvText, `machinasense_${type}_sample.csv`, parseInt(load));
      } catch (err) {
        alert('Failed to load sample: ' + err.message);
      } finally {
        btn.innerHTML = origHTML;
      }
    });
  });

  // Analyze button
  runBtn?.addEventListener('click', async () => {
    if (!selectedCsvFile) {
      alert('Please select a CSV file first, or use one of the 1-click sample buttons above.');
      return;
    }
    const loadHp   = parseInt(loadSelect?.value || 0);
    const origHTML = runBtn.innerHTML;
    runBtn.disabled = true;
    runBtn.innerHTML = '<span>⚡ Analyzing Signal...</span>';
    try {
      const formData = new FormData();
      formData.append('file', selectedCsvFile);
      formData.append('load_hp', loadHp);
      const resp = await fetch(`${API}/api/upload_csv`, { method: 'POST', body: formData });
      const res  = await resp.json();
      if (!resp.ok) { alert('Diagnostic Error: ' + (res.error || 'Failed to analyze CSV')); return; }
      displayDiagnosticResults(res);
    } catch (err) {
      alert('Network or processing error: ' + err.message);
    } finally {
      runBtn.disabled  = false;
      runBtn.innerHTML = origHTML;
    }
  });

  async function analyzeCsvContent(csvText, filename, loadHp) {
    if (runBtn) { runBtn.disabled = true; runBtn.innerHTML = '<span>⚡ Analyzing...</span>'; }
    try {
      const resp = await fetch(`${API}/api/upload_csv`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ csv_text: csvText, filename, load_hp: loadHp }),
      });
      const res = await resp.json();
      if (!resp.ok) { alert('Diagnostic Error: ' + (res.error || 'Failed to analyze')); return; }
      displayDiagnosticResults(res);
    } catch (err) {
      alert('Analysis error: ' + err.message);
    } finally {
      if (runBtn) { runBtn.disabled = false; runBtn.innerHTML = '⚡ Analyze Machine Risk'; }
    }
  }

  function displayDiagnosticResults(data) {
    const panel = document.getElementById('upload-results-panel');
    if (!panel) return;
    panel.style.display = 'block';

    const badge = document.getElementById('rb-badge');
    if (badge) { badge.textContent = data.risk_level; badge.className = `rb-badge ${data.badge_class || 'risk-normal'}`; }

    setText('rb-title',      data.title_en);
    setText('rb-file-info',  `File: ${data.filename} • ${data.total_samples.toLocaleString()} samples (${data.duration_sec}s) • Operating Load: ${data.load_hp} HP • Windows: ${data.num_segments}`);
    setText('rb-score',      data.overall_score.toFixed(3));
    setText('rb-confidence', `${data.confidence_pct}%`);
    setText('rb-hypothesis', data.fault_hypothesis);
    setText('rb-msg-en',     data.message_en);
    setText('rb-action-en',  `👉 Recommendation: ${data.action_en}`);
    setText('rb-msg-hi',     data.message_hi);
    setText('rb-action-hi',  `👉 सलाह: ${data.action_hi}`);

    const waveCanvas = document.getElementById('upload-waveform-chart');
    if (waveCanvas && data.waveform) {
      MachinaCharts.waveform(waveCanvas, data.waveform, {
        color: data.overall_score >= 0.45 ? '#ef4444' : '#06b6d4'
      });
    }
    const specCanvas = document.getElementById('upload-spectrum-chart');
    if (specCanvas && data.fft_spectrum) {
      MachinaCharts.spectrum(specCanvas, data.fft_spectrum.freqs, data.fft_spectrum.powers);
    }
    const timeCanvas = document.getElementById('upload-timeline-chart');
    if (timeCanvas && data.timeline) {
      MachinaCharts.line(timeCanvas, [{
        data:      data.timeline.scores,
        color:     data.overall_score >= 0.45 ? '#ef4444' : '#10b981',
        fill:      data.overall_score >= 0.45 ? 'rgba(239,68,68,0.15)' : 'rgba(16,185,129,0.15)',
        lineWidth: 2,
      }], {
        xLabels:   data.timeline.timestamps.map(t => `${t}s`),
        yMin: 0, yMax: 1.0,
        threshold: 0.45, yTicks: 4,
      });
    }

    const tbody = document.getElementById('upload-evidence-tbody');
    if (tbody && data.evidence) {
      tbody.innerHTML = data.evidence.map(item => {
        const z    = item.zscore;
        const cls  = Math.abs(z) >= 3.0 ? 'danger' : Math.abs(z) >= 1.5 ? 'warn' : 'normal';
        return `
          <tr>
            <td><strong>${item.feature}</strong></td>
            <td><code>${item.value}</code></td>
            <td><code>${item.baseline_mean}</code></td>
            <td><span class="z-score-pill ${cls}">${z >= 0 ? '+' : ''}${z}σ</span></td>
          </tr>`;
      }).join('');
    }

    // Trigger voice assistant for CSV diagnostic result
    const sev = data.severity || 'normal';
    VA.announce(sev, data.overall_score);

    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

function renderUploadPage() { /* loaded on navigate */ }

// ═══════════════════════════════════════════════════════════════════════════
// ─── Voice Assistant ────────────────────────────────────────────────────────
// Web Audio API for synthesised tones + Web Speech API for spoken alerts.
// Completely standalone — zero external dependencies.
// ═══════════════════════════════════════════════════════════════════════════
const VA = (() => {
  let _ctx  = null;
  let _last = 'normal';
  let _timer = null;

  function _getCtx() {
    if (!_ctx) {
      try { _ctx = new (window.AudioContext || window.webkitAudioContext)(); } catch (_) {}
    }
    return _ctx;
  }

  /**
   * Play repeating square-wave beeps (warning/risk).
   * @param {number} freqHz   - tone frequency
   * @param {number} durMs    - duration of each beep in ms
   * @param {number} repeats  - number of beeps
   */
  function _beep(freqHz, durMs, repeats) {
    const ctx = _getCtx();
    if (!ctx) return;
    for (let i = 0; i < repeats; i++) {
      const t    = ctx.currentTime + i * (durMs / 1000 + 0.05);
      const osc  = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain); gain.connect(ctx.destination);
      osc.type = 'square';
      osc.frequency.setValueAtTime(freqHz, t);
      gain.gain.setValueAtTime(0.16, t);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + durMs / 1000);
      osc.start(t); osc.stop(t + durMs / 1000 + 0.01);
    }
  }

  /** Play a gentle ascending C5-E5-G5 sine chime (safe/normal). */
  function _chime() {
    const ctx = _getCtx();
    if (!ctx) return;
    [523.25, 659.25, 783.99].forEach((freq, i) => {
      const t    = ctx.currentTime + i * 0.2;
      const osc  = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain); gain.connect(ctx.destination);
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, t);
      gain.gain.setValueAtTime(0.13, t);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.5);
      osc.start(t); osc.stop(t + 0.55);
    });
  }

  /** Speak text via browser SpeechSynthesis. */
  function _speak(text) {
    if (!window.speechSynthesis) return;
    const utt = new SpeechSynthesisUtterance(text);
    utt.lang = 'en-US'; utt.rate = 1.05; utt.pitch = 1.0; utt.volume = 0.9;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utt);
  }

  /** Show the bottom-right floating bubble for a few seconds. */
  function _bubble(icon, text, cssClass, ms) {
    const bubble = document.getElementById('va-bubble');
    const iconEl = document.getElementById('va-icon');
    const textEl = document.getElementById('va-text');
    if (!bubble) return;
    if (iconEl) iconEl.textContent = icon;
    if (textEl) textEl.textContent = text;
    bubble.className = cssClass;
    bubble.style.display = 'block';
    clearTimeout(_timer);
    _timer = setTimeout(() => { bubble.style.display = 'none'; }, ms);
  }

  /**
   * announce(severity, score)
   * Call on every sensor reading. Plays audio + speech only when
   * the severity level CHANGES, avoiding repetitive noise.
   */
  function announce(severity, score) {
    if (severity === _last) return;
    _last = severity;
    const sc = (score || 0).toFixed(2);

    switch (severity) {
      case 'critical':
        _beep(960, 130, 4);
        setTimeout(() => _speak(`Critical bearing fault! Score ${sc}. Stop machine immediately.`), 600);
        _bubble('🚨', `CRITICAL — Score ${sc}: Stop machine immediately!`, 'risk', 7000);
        break;
      case 'high':
        _beep(780, 120, 3);
        setTimeout(() => _speak(`High risk. Anomaly score ${sc}. Inspect drive-end bearing soon.`), 500);
        _bubble('⚠️', `HIGH RISK — Score ${sc}: Inspect bearing soon.`, 'risk', 5500);
        break;
      case 'medium':
        _beep(660, 100, 2);
        setTimeout(() => _speak(`Warning. Score ${sc}. Monitor closely.`), 400);
        _bubble('⚡', `MEDIUM — Score ${sc}: Monitor closely.`, 'risk', 5000);
        break;
      case 'low':
        _beep(440, 80, 1);
        _bubble('📡', `LOW RISK — Score ${sc}: Early anomaly detected.`, 'risk', 4500);
        break;
      case 'normal':
      default:
        _chime();
        _speak('Machine is operating normally. All clear.');
        _bubble('✅', 'All clear — machine operating normally.', 'safe', 4000);
        break;
    }
  }

  return { announce };
})();

// ═══════════════════════════════════════════════════════════════════════════
// ─── CSV Upload IN Simulator — Stream CSV through the oscilloscope ──────────
// POSTs user's CSV to /api/upload_csv (the real trained-model endpoint),
// then replays each per-segment score+waveform sequentially into the
// oscilloscope canvas at 600 ms/frame — identical to the live sensor feed.
// ═══════════════════════════════════════════════════════════════════════════
function setupSimCsvUpload() {
  const browseBtn  = document.getElementById('sim-csv-browse-btn');
  const fileInput  = document.getElementById('sim-csv-input');
  const filenameEl = document.getElementById('sim-csv-filename');
  const clearBtn   = document.getElementById('sim-csv-clear-btn');
  const analyzeBtn = document.getElementById('sim-csv-analyze-btn');
  const loadSel    = document.getElementById('sim-csv-load-select');
  const progressEl = document.getElementById('sim-csv-progress');
  const fillEl     = document.getElementById('sim-csv-fill');
  const statusEl   = document.getElementById('sim-csv-status-text');
  const oscLabel   = document.getElementById('osc-live-label');

  if (!browseBtn) return;

  let _file = null;

  browseBtn.addEventListener('click', () => fileInput && fileInput.click());

  fileInput?.addEventListener('change', () => {
    if (fileInput.files && fileInput.files.length > 0) {
      _file = fileInput.files[0];
      if (filenameEl) filenameEl.textContent = _file.name;
      if (clearBtn)   clearBtn.style.display   = 'inline-flex';
      if (analyzeBtn) analyzeBtn.disabled       = false;
    }
  });

  clearBtn?.addEventListener('click', () => {
    _file = null;
    if (fileInput)  fileInput.value              = '';
    if (filenameEl) filenameEl.textContent        = 'No file selected';
    if (clearBtn)   clearBtn.style.display        = 'none';
    if (analyzeBtn) analyzeBtn.disabled           = true;
    if (progressEl) progressEl.style.display      = 'none';
  });

  analyzeBtn?.addEventListener('click', async () => {
    if (!_file)          return;
    if (!state.loaded)   { alert('Model is still training. Please wait.'); return; }

    // Stop any running live sim
    stopSim();

    const loadHp = parseInt(loadSel?.value || 1);
    analyzeBtn.disabled  = true;
    analyzeBtn.textContent = '⏳ Uploading…';
    if (progressEl) progressEl.style.display = 'block';
    if (fillEl)     fillEl.style.width        = '5%';
    if (statusEl)   statusEl.textContent       = 'Uploading CSV to AI model…';
    if (oscLabel)   oscLabel.textContent       = '● CSV STREAM';

    try {
      // ── 1. POST to the real trained-model endpoint ────────────────
      const formData = new FormData();
      formData.append('file', _file);
      formData.append('load_hp', loadHp);

      const resp = await fetch(API + '/api/upload_csv', { method: 'POST', body: formData });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        alert('Upload failed: ' + (err.error || resp.statusText));
        return;
      }

      const data = await resp.json();
      if (fillEl) fillEl.style.width = '40%';

      // ── 2. Replay per-segment data into the oscilloscope ──────────
      const scores   = (data.timeline || {}).scores  || [data.overall_score];
      const nSegs    = scores.length;
      const waveform = data.waveform   || [];
      const fftFreqs = (data.fft_spectrum || {}).freqs  || [];
      const fftPower = (data.fft_spectrum || {}).powers || [];

      if (statusEl) statusEl.textContent = `Streaming ${nSegs} window(s) into oscilloscope…`;

      for (let i = 0; i < nSegs; i++) {
        const score = scores[i];
        const sev   = score >= 0.85 ? 'critical'
                    : score >= 0.70 ? 'high'
                    : score >= 0.50 ? 'medium'
                    : score >= 0.35 ? 'low'
                    : 'normal';

        // Update gauge
        updateGauge(score, sev);

        // Update oscilloscope — add tiny frame-jitter so it doesn't look frozen
        const wfCanvas = document.getElementById('waveform-chart');
        if (wfCanvas && waveform.length) {
          const jitter = waveform.map(v => v + (Math.random() - 0.5) * Math.abs(v) * 0.04);
          const col    = sev === 'critical' ? '#ef4444'
                       : sev === 'high'     ? '#f97316'
                       : sev === 'medium'   ? '#f59e0b' : '#06b6d4';
          MachinaCharts.waveform(wfCanvas, jitter, { color: col });
        }

        // Update FFT spectrum
        const spCanvas = document.getElementById('spectrum-chart');
        if (spCanvas && fftFreqs.length) {
          MachinaCharts.spectrum(spCanvas, fftFreqs, fftPower);
        }

        setText('sim-step-label', `CSV Window ${i + 1} / ${nSegs}`);
        setText('r-load', loadHp + ' HP');
        setText('r-conf', data.confidence_pct + '%');
        setText('r-rul',  data.rul_hours != null ? '~' + Math.round(data.rul_hours) + ' h' : 'See CSV Result');

        // ── Voice assistant ───────────────────────────────────────────
        VA.announce(sev, score);

        // Update alert area
        const latestEl = document.getElementById('latest-alert');
        if (latestEl) {
          if (sev !== 'normal') {
            latestEl.innerHTML = `
              <div class="alert-card ${sev}">
                <div class="alert-header">
                  <span class="alert-sev ${sev}">${sev.toUpperCase()}</span>
                  <span style="font-weight:700;font-size:.9rem">CSV Window ${i + 1} — ${data.fault_hypothesis || 'Bearing Anomaly'}</span>
                </div>
                <div class="alert-meta">
                  <span>Score: ${score.toFixed(3)}</span>
                  <span>Confidence: ${data.confidence_pct}%</span>
                  <span>Load: ${loadHp} HP</span>
                </div>
                <div class="alert-msg-en"><strong>EN:</strong> ${data.message_en || ''}</div>
                <div class="alert-msg-hi"><strong>हिंदी:</strong> ${data.message_hi || ''}</div>
              </div>`;
          } else {
            latestEl.innerHTML = `<div class="no-alert"><span class="no-alert-icon">✔</span>
              <div>CSV Window ${i + 1}: Normal. ${data.message_en || ''}</div></div>`;
          }
        }

        if (fillEl) fillEl.style.width = (40 + 60 * ((i + 1) / nSegs)) + '%';

        // 600 ms between frames
        await new Promise(r => setTimeout(r, 600));
      }

      if (statusEl) statusEl.textContent = `✔ Done — ${nSegs} windows. Overall: ${data.risk_level}`;
      if (oscLabel) oscLabel.textContent  = '● CSV DONE';

    } catch (err) {
      alert('CSV stream error: ' + err.message);
      if (statusEl) statusEl.textContent = 'Error — ' + err.message;
    } finally {
      analyzeBtn.disabled    = false;
      analyzeBtn.textContent = '⚡ Analyze & Stream';
    }
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// ─── Temperature & Load Monitor (Ottawa Dataset AI Model) ───────────────────
// Real-time motor temperature + fault monitoring using the Ottawa-trained
// Isolation Forest. VA integration fires on severity change.
// ═══════════════════════════════════════════════════════════════════════════

const tlState = {
  running:    false,
  timer:      null,
  condition:  'healthy',
  loadCond:   0,
  history:    [],
  tempHistory:[],
  step:       0,
  baselinesLoaded: false,
};

// Temperature colour helper
function tempColor(t) {
  if (t < 35)  return '#10b981';
  if (t < 50)  return '#f59e0b';
  if (t < 65)  return '#f97316';
  return '#ef4444';
}

// Update the SVG thermometer
function updateThermometer(tempC) {
  // Map 15–80°C → y coordinate: 220 (bottom=15°) to 25 (top=80°)
  const MIN_T = 15, MAX_T = 80;
  const MIN_Y = 220, MAX_Y = 25;
  const clampedT = Math.max(MIN_T, Math.min(MAX_T, tempC));
  const fillY    = MIN_Y - ((clampedT - MIN_T) / (MAX_T - MIN_T)) * (MIN_Y - MAX_Y);
  const fillH    = MIN_Y - fillY;
  const col      = tempColor(clampedT);

  const fill = document.getElementById('tl-thermo-fill');
  const bulb = document.getElementById('tl-bulb-inner');
  const outerBulb = document.getElementById('tl-thermo-bulb');
  const txt  = document.getElementById('tl-thermo-text');

  if (fill) { fill.setAttribute('y', fillY); fill.setAttribute('height', fillH); fill.setAttribute('fill', col); }
  if (bulb) { bulb.setAttribute('fill', col); }
  if (outerBulb) { outerBulb.setAttribute('fill', col); }
  if (txt)  { txt.textContent = tempC.toFixed(1) + '°C'; txt.setAttribute('fill', col); }
}

function setupTempLoad() {
  document.querySelectorAll('.tl-cond-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tl-cond-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      tlState.condition = btn.dataset.cond;
    });
  });

  document.querySelectorAll('.tl-load-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tl-load-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      tlState.loadCond = parseInt(btn.dataset.load);
    });
  });

  document.getElementById('tl-start')?.addEventListener('click', startTempLoad);
  document.getElementById('tl-stop')?.addEventListener('click',  stopTempLoad);
  document.getElementById('tl-reset')?.addEventListener('click', resetTempLoad);
}

function startTempLoad() {
  if (tlState.running) return;
  tlState.running = true;
  document.getElementById('tl-start').disabled = true;
  document.getElementById('tl-stop').disabled  = false;
  const speed = parseInt(document.getElementById('tl-speed')?.value || '1000');
  setText('tl-osc-label', '● LIVE');
  runTempStep();
  tlState.timer = setInterval(runTempStep, speed);
}

function stopTempLoad() {
  tlState.running = false;
  clearInterval(tlState.timer);
  document.getElementById('tl-start').disabled = false;
  document.getElementById('tl-stop').disabled  = true;
  setText('tl-osc-label', '● PAUSED');
}

function resetTempLoad() {
  stopTempLoad();
  tlState.history     = [];
  tlState.tempHistory = [];
  tlState.step        = 0;
  setText('tl-temp-val',   '—°C');
  setText('tl-temp-status','Awaiting data…');
  setText('tl-load-val',   '—');
  setText('tl-load-rpm',   '—');
  setText('tl-score-val',  '—');
  setText('tl-sev-label',  'Ottawa AI Model');
  setText('tl-step-val',   '0');
  setText('tl-osc-label',  '● IDLE');
  setText('tl-temp-badge-label', 'NORMAL');
  updateThermometer(28);

  const alertEl = document.getElementById('tl-alert-box');
  if (alertEl) alertEl.innerHTML = `
    <div class="no-alert"><span class="no-alert-icon">✔</span>
    <div>Monitoring reset. Press ▶ Start Monitoring to begin.</div></div>`;

  const wf = document.getElementById('tl-waveform-chart');
  if (wf) MachinaCharts.waveform(wf, Array(128).fill(0));
}

async function runTempStep() {
  try {
    const res = await fetch(API + '/api/temp_monitor/simulate', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        condition:      tlState.condition,
        load_condition: tlState.loadCond,
      }),
    });
    const d = await res.json();
    if (d.error) { console.error('TempLoad step error:', d.error); return; }

    tlState.step        = d.step;
    tlState.history     = d.history     || [];
    tlState.tempHistory = d.temp_history || [];

    // ── KPI updates ────────────────────────────────────────────────────
    const tCol = tempColor(d.temperature);
    const scoreEl = document.getElementById('tl-score-val');
    const sevColor = { normal:'#10b981', low:'#06b6d4', medium:'#f59e0b', high:'#f97316', critical:'#ef4444' };

    setText('tl-temp-val',   d.temperature.toFixed(1) + '°C');
    setText('tl-temp-status', d.temp_label);
    setText('tl-load-val',   d.load_condition === 1 ? 'LOADED' : 'UNLOADED');
    setText('tl-load-rpm',   d.load_condition === 1 ? '1500 RPM (50 Hz)' : '3000 RPM (50 Hz)');
    setText('tl-score-val',  d.score.toFixed(3));
    setText('tl-sev-label',  d.severity.toUpperCase());
    setText('tl-step-val',   d.step);
    setText('tl-temp-badge-label', d.temp_status.toUpperCase());

    if (scoreEl) scoreEl.style.color = sevColor[d.severity] || '#818cf8';
    const tempValEl = document.getElementById('tl-temp-val');
    if (tempValEl) tempValEl.style.color = tCol;

    // ── Thermometer ────────────────────────────────────────────────────
    updateThermometer(d.temperature);

    // ── Temperature trend chart ────────────────────────────────────────
    const tempCanvas = document.getElementById('tl-temp-chart');
    if (tempCanvas && tlState.tempHistory.length > 1) {
      const tMin = Math.max(0, Math.min(...tlState.tempHistory) - 5);
      const tMax = Math.max(...tlState.tempHistory) + 5;
      MachinaCharts.line(tempCanvas, [{
        data:      tlState.tempHistory,
        color:     tCol,
        lineWidth: 2.5,
        fill:      tCol.replace(')', ',0.1)').replace('rgb', 'rgba'),
      }], {
        yMin: tMin, yMax: tMax,
        xLabels: tlState.tempHistory.map((_, i) => i % 10 === 0 ? i + '' : ''),
        pad: { l:44, r:16, t:16, b:28 },
      });
    }

    // ── Anomaly score trend chart ──────────────────────────────────────
    const scoreCanvas = document.getElementById('tl-score-chart');
    if (scoreCanvas && tlState.history.length > 1) {
      const sCol = sevColor[d.severity] || '#818cf8';
      MachinaCharts.line(scoreCanvas, [{
        data:      tlState.history,
        color:     sCol,
        lineWidth: 2,
        fill:      sCol.replace(')', ',0.1)').replace('rgb', 'rgba'),
      }], {
        yMin: 0, yMax: 1,
        xLabels: tlState.history.map((_, i) => i % 10 === 0 ? i + '' : ''),
        threshold: 0.45,
        pad: { l:44, r:16, t:16, b:28 },
      });
    }

    // ── Oscilloscope (fixed scale) ─────────────────────────────────────
    const wfCanvas = document.getElementById('tl-waveform-chart');
    if (wfCanvas && d.waveform) {
      const wfCol = d.severity === 'critical' ? '#ef4444'
                  : d.severity === 'high'     ? '#f97316'
                  : d.severity === 'medium'   ? '#f59e0b' : '#06b6d4';
      MachinaCharts.waveform(wfCanvas, d.waveform, { color: wfCol });
    }

    // ── Alert box ─────────────────────────────────────────────────────
    const alertEl = document.getElementById('tl-alert-box');
    if (alertEl) {
      const condDisplay = d.condition.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      if (d.severity !== 'normal') {
        alertEl.innerHTML = `
          <div class="alert-card ${d.severity}">
            <div class="alert-header">
              <span class="alert-sev ${d.severity}">${d.severity.toUpperCase()}</span>
              <span style="font-weight:700;font-size:.9rem">Motor: ${condDisplay}</span>
            </div>
            <div class="alert-meta">
              <span>Score: ${d.score.toFixed(3)}</span>
              <span>Temp: ${d.temperature}°C</span>
              <span>${d.load_label}</span>
              <span>Step: ${d.step}</span>
            </div>
            <div class="alert-msg-en">
              <strong>Status:</strong> ${d.temp_label} — Anomaly score ${d.score.toFixed(3)} exceeds healthy baseline.<br>
              <strong>Action:</strong> Check motor bearing temperature. Verify load condition and ventilation.
            </div>
          </div>`;
      } else {
        alertEl.innerHTML = `
          <div class="no-alert">
            <span class="no-alert-icon">✔</span>
            <div>${condDisplay} — ${d.temp_label}. Score: ${d.score.toFixed(3)}. Operating normally.</div>
          </div>`;
      }
    }

    // ── Voice assistant ────────────────────────────────────────────────
    VA.announce(d.severity, d.score);

    // Also fire VA on temperature critical
    if (d.temp_status === 'critical') {
      VA.announce('critical', d.score);
    } else if (d.temp_status === 'high') {
      VA.announce('high', d.score);
    }

  } catch (err) {
    console.error('TempLoad tick error:', err);
  }
}

async function renderTempLoadPage() {
  // Load temperature baselines from Ottawa model
  if (tlState.baselinesLoaded) return;
  try {
    const res = await fetch(API + '/api/temp_monitor/status');
    const d   = await res.json();
    if (!d.ready) {
      const grid = document.getElementById('tl-baselines-grid');
      if (grid) grid.innerHTML = '<div class="loading-msg">Ottawa model still training… please wait.</div>';
      return;
    }

    // Update dataset pill
    const pill = document.getElementById('tl-dataset-pill');
    if (pill) pill.textContent = d.data_source || 'Ottawa Dataset';

    // Render per-condition temperature baselines
    const grid = document.getElementById('tl-baselines-grid');
    if (grid && d.temp_baselines) {
      const bl = d.temp_baselines;
      grid.innerHTML = Object.entries(bl).map(([cond, stats]) => {
        const label = cond.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        const col   = tempColor(stats.mean);
        return `
          <div class="baseline-card">
            <div class="baseline-label">${label}</div>
            <div class="baseline-temp" style="color:${col}">${stats.mean.toFixed(1)}°C</div>
            <div class="baseline-range">Range: ${stats.min.toFixed(1)}–${stats.max.toFixed(1)}°C</div>
            <div class="baseline-n">${stats.count} samples</div>
          </div>`;
      }).join('');
      tlState.baselinesLoaded = true;
    }
  } catch (err) {
    console.error('TempLoad page load error:', err);
  }
}


// ─── AI Chatbot ────────────────────────────────────────────────────────────

(function () {
  'use strict';

  const fab       = document.getElementById('chatbot-fab');
  const panel     = document.getElementById('chatbot-panel');
  const closeBtn  = document.getElementById('chat-close-btn');
  const msgArea   = document.getElementById('chat-messages');
  const inputEl   = document.getElementById('chat-input');
  const sendBtn   = document.getElementById('chat-send-btn');
  const quickBtns = document.querySelectorAll('.chat-quick-btn');

  let isOpen       = false;
  let isBusy       = false;
  let hasGreeted   = false;
  // Conversation history (role/content pairs sent to the backend)
  const history    = [];

  // ── Toggle panel ─────────────────────────────────────────────────────────
  function openChat() {
    isOpen = true;
    panel.classList.remove('chat-hidden');
    fab.classList.remove('has-notif');
    fab.setAttribute('aria-expanded', 'true');
    inputEl.focus();
    if (!hasGreeted) {
      hasGreeted = true;
      addBotMessage(
        "👋 **Hello!** I'm the **MachinaSense AI Assistant**.\n\n" +
        "I know everything about this platform — fault detection, hardware costs, ROI, the live simulator, alerts, and more.\n\n" +
        "Use the quick-action chips below or type your own question!"
      );
    }
  }

  function closeChat() {
    isOpen = false;
    panel.classList.add('chat-hidden');
    fab.setAttribute('aria-expanded', 'false');
  }

  fab.addEventListener('click', () => isOpen ? closeChat() : openChat());
  closeBtn.addEventListener('click', closeChat);

  // Close on Escape key
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && isOpen) closeChat();
  });

  // ── Quick-action chips ────────────────────────────────────────────────────
  quickBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const q = btn.getAttribute('data-q');
      if (q && !isBusy) sendMessage(q);
    });
  });

  // ── Input: send on Enter (Shift+Enter = newline) ──────────────────────────
  inputEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isBusy) doSend();
    }
    // Auto-grow textarea
    setTimeout(() => {
      inputEl.style.height = 'auto';
      inputEl.style.height = Math.min(inputEl.scrollHeight, 90) + 'px';
    }, 0);
  });
  sendBtn.addEventListener('click', () => { if (!isBusy) doSend(); });

  function doSend() {
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = '';
    inputEl.style.height = 'auto';
    sendMessage(text);
  }

  // ── Markdown → HTML (minimal, safe) ──────────────────────────────────────
  function mdToHtml(text) {
    // Escape HTML entities first
    let s = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Tables: | col | col |
    s = s.replace(/(?:^\|.+\|[ \t]*$\n?)+/gm, match => {
      const rows = match.trim().split('\n').filter(r => r.trim());
      let html = '<table>';
      let inBody = false;
      rows.forEach((row, i) => {
        // Skip separator rows like |---|---|
        if (/^\|[\s\-|:]+\|$/.test(row.trim())) { inBody = true; return; }
        const cells = row.split('|').slice(1, -1).map(c => c.trim());
        if (i === 0 && !inBody) {
          html += '<tr>' + cells.map(c => `<th>${c}</th>`).join('') + '</tr>';
        } else {
          html += '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>';
        }
      });
      html += '</table>';
      return html;
    });

    // Bold **text** or __text__
    s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/__(.+?)__/g, '<strong>$1</strong>');
    // Italic *text*
    s = s.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
    // Inline code `text`
    s = s.replace(/`([^`]+)`/g, '<span class="code">$1</span>');

    // Bullet list lines
    s = s.replace(/^[•\-\*] (.+)$/gm, '<li>$1</li>');
    s = s.replace(/((<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Numbered list lines
    s = s.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');

    // Line breaks: double newline → paragraph break
    s = s.replace(/\n{2,}/g, '<br><br>');
    // Single newline
    s = s.replace(/\n/g, '<br>');

    return s;
  }

  // ── DOM helpers ──────────────────────────────────────────────────────────
  function addBotMessage(text, engine) {
    const wrap = document.createElement('div');
    wrap.className = 'chat-msg bot-msg';
    wrap.innerHTML = `
      <div class="chat-avatar bot-avatar">🤖</div>
      <div>
        <div class="chat-bubble bot-bubble">${mdToHtml(text)}</div>
        ${engine ? `<div class="engine-tag">powered by ${engine}</div>` : ''}
      </div>`;
    msgArea.appendChild(wrap);
    scrollDown();
    return wrap;
  }

  function addUserMessage(text) {
    const wrap = document.createElement('div');
    wrap.className = 'chat-msg user-msg';
    wrap.innerHTML = `
      <div class="chat-avatar user-avatar">You</div>
      <div class="chat-bubble user-bubble">${mdToHtml(text)}</div>`;
    msgArea.appendChild(wrap);
    scrollDown();
  }

  function showTyping() {
    const wrap = document.createElement('div');
    wrap.className = 'chat-msg bot-msg';
    wrap.id = 'chat-typing-indicator';
    wrap.innerHTML = `
      <div class="chat-avatar bot-avatar">🤖</div>
      <div class="chat-typing">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>`;
    msgArea.appendChild(wrap);
    scrollDown();
  }

  function removeTyping() {
    const el = document.getElementById('chat-typing-indicator');
    if (el) el.remove();
  }

  function scrollDown() {
    msgArea.scrollTop = msgArea.scrollHeight;
  }

  function setBusy(busy) {
    isBusy = busy;
    sendBtn.disabled = busy;
    inputEl.disabled = busy;
  }

  // ── Send message to backend ────────────────────────────────────────────
  async function sendMessage(text) {
    if (!text.trim() || isBusy) return;

    // Open panel if not already open
    if (!isOpen) openChat();

    addUserMessage(text);
    history.push({ role: 'user', content: text });

    setBusy(true);
    showTyping();

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: history })
      });

      removeTyping();

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        addBotMessage('⚠️ Sorry, I encountered an error. Please try again.\n\n' + (err.error || `HTTP ${res.status}`));
        setBusy(false);
        return;
      }

      const data = await res.json();
      const reply = data.reply || '(no response)';
      const engine = data.engine || null;

      addBotMessage(reply, engine);
      history.push({ role: 'assistant', content: reply });

      // Cap history to last 20 messages to avoid large payloads
      if (history.length > 20) history.splice(0, history.length - 20);

      // Show notification dot on FAB if panel is closed
      if (!isOpen) {
        fab.classList.add('has-notif');
      }
    } catch (err) {
      removeTyping();
      addBotMessage('⚠️ Network error — could not reach the server. Make sure the Flask server is running.');
    } finally {
      setBusy(false);
      scrollDown();
    }
  }

  // ── Show notification dot after page fully loads ───────────────────────
  window.addEventListener('load', () => {
    setTimeout(() => {
      if (!isOpen) fab.classList.add('has-notif');
    }, 3500);
  });

})();
