/**
 * charts.js — MachinaSense Lightweight Canvas Charting
 * Zero external dependencies. Pure Canvas 2D API.
 *
 * Exports (via window.MachinaCharts):
 *   line(canvas, datasets, opts)
 *   bar(canvas, labels, values, opts)
 *   hbar(canvas, labels, values, opts)
 *   histogram(canvas, normalBins, faultBins, opts)
 *   waveform(canvas, data, opts)   ← fixed-scale oscilloscope (no zoom)
 *   spectrum(canvas, freqs, powers, opts)
 */

const C = (() => {

  // ─── Colour palette ───────────────────────────────────────────────────
  const COLORS = {
    accent:  '#6366f1', accent2: '#818cf8',
    green:   '#22c55e', yellow:  '#eab308',
    orange:  '#f97316', red:     '#ef4444',
    muted:   '#8892b0', border:  '#2d3148',
    text:    '#e8eaf6', surface: '#1a1d27',
    grid:    'rgba(45,49,72,0.6)',
  };

  function _dpr() { return window.devicePixelRatio || 1; }

  function _setup(canvas) {
    const dpr  = _dpr();
    const rect = canvas.getBoundingClientRect();
    canvas.width  = rect.width  * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    return { ctx, w: rect.width, h: rect.height };
  }

  function _gridAndAxes(ctx, w, h, pad, xLabels, yMin, yMax, yTicks = 5) {
    ctx.clearRect(0, 0, w, h);
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth   = 1;
    for (let i = 0; i <= yTicks; i++) {
      const y = pad.t + ph * (1 - i / yTicks);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
      const val = yMin + (yMax - yMin) * (i / yTicks);
      ctx.fillStyle = COLORS.muted;
      ctx.font = '10px system-ui';
      ctx.textAlign = 'right';
      ctx.fillText(val.toFixed(2), pad.l - 5, y + 3);
    }

    if (xLabels && xLabels.length) {
      ctx.fillStyle = COLORS.muted;
      ctx.font = '10px system-ui';
      ctx.textAlign = 'center';
      const step = Math.max(1, Math.floor(xLabels.length / 8));
      xLabels.forEach((lbl, i) => {
        if (i % step !== 0) return;
        const x = pad.l + (i / (xLabels.length - 1)) * pw;
        ctx.fillText(lbl, x, h - 4);
      });
    }
    return { pw, ph, pad };
  }

  // ─── Line chart ──────────────────────────────────────────────────────
  function line(canvas, datasets, opts = {}) {
    const { ctx, w, h } = _setup(canvas);
    const pad    = opts.pad || { l: 42, r: 16, t: 16, b: 28 };
    const yMin   = opts.yMin !== undefined ? opts.yMin : 0;
    const yMax   = opts.yMax !== undefined ? opts.yMax : 1;
    const xLabels = opts.xLabels || [];
    const { pw, ph } = _gridAndAxes(ctx, w, h, pad, xLabels, yMin, yMax);

    datasets.forEach(ds => {
      const data = ds.data;
      if (!data || !data.length) return;

      ctx.beginPath();
      ctx.strokeStyle = ds.color || COLORS.accent;
      ctx.lineWidth   = ds.lineWidth || 2;
      ctx.lineJoin    = 'round';

      data.forEach((val, i) => {
        const x = pad.l + (i / Math.max(data.length - 1, 1)) * pw;
        const y = pad.t + ph * (1 - (val - yMin) / (yMax - yMin));
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();

      if (ds.fill) {
        ctx.lineTo(pad.l + pw, pad.t + ph);
        ctx.lineTo(pad.l, pad.t + ph);
        ctx.closePath();
        ctx.fillStyle = ds.fill;
        ctx.fill();
      }

      if (opts.threshold !== undefined) {
        const ty = pad.t + ph * (1 - (opts.threshold - yMin) / (yMax - yMin));
        ctx.beginPath();
        ctx.strokeStyle = COLORS.red;
        ctx.lineWidth   = 1;
        ctx.setLineDash([5, 4]);
        ctx.moveTo(pad.l, ty); ctx.lineTo(pad.l + pw, ty);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = COLORS.red;
        ctx.font = '10px system-ui'; ctx.textAlign = 'left';
        ctx.fillText('Threshold', pad.l + 4, ty - 4);
      }
    });

    if (opts.legend && opts.legend.length) {
      let lx = pad.l;
      opts.legend.forEach((item, i) => {
        const col = (datasets[i] || {}).color || COLORS.accent;
        ctx.fillStyle = col;
        ctx.fillRect(lx, 4, 12, 8);
        ctx.fillStyle = COLORS.muted;
        ctx.font = '10px system-ui'; ctx.textAlign = 'left';
        ctx.fillText(item, lx + 15, 12);
        lx += ctx.measureText(item).width + 36;
      });
    }
  }

  // ─── Bar chart ───────────────────────────────────────────────────────
  function bar(canvas, labels, values, opts = {}) {
    const { ctx, w, h } = _setup(canvas);
    const pad  = opts.pad || { l: 46, r: 16, t: 20, b: 50 };
    const yMin = 0;
    const yMax = opts.yMax || Math.max(...values, 0.001) * 1.2;
    const { pw, ph } = _gridAndAxes(ctx, w, h, pad, [], yMin, yMax, 4);

    const n = labels.length;
    const bw  = (pw / n) * 0.6;
    const gap = (pw / n) * 0.4;

    labels.forEach((lbl, i) => {
      const x  = pad.l + i * (pw / n) + gap / 2;
      const bh = ph * (values[i] / yMax);
      const y  = pad.t + ph - bh;
      const color = (opts.colors && opts.colors[i]) || COLORS.accent;

      ctx.fillStyle = color;
      const r = 4;
      ctx.beginPath();
      ctx.moveTo(x + r, y); ctx.lineTo(x + bw - r, y);
      ctx.quadraticCurveTo(x + bw, y, x + bw, y + r);
      ctx.lineTo(x + bw, y + bh); ctx.lineTo(x, y + bh);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.closePath(); ctx.fill();

      ctx.fillStyle = COLORS.muted;
      ctx.font = '10px system-ui'; ctx.textAlign = 'center';
      ctx.fillText(lbl, x + bw / 2, h - 6);

      ctx.fillStyle = COLORS.text;
      ctx.font = '10px system-ui'; ctx.textAlign = 'center';
      ctx.fillText(values[i].toFixed(3), x + bw / 2, y - 4);
    });
  }

  // ─── Horizontal bar ──────────────────────────────────────────────────
  function hbar(canvas, labels, values, opts = {}) {
    const { ctx, w, h } = _setup(canvas);
    const pad  = { l: 150, r: 60, t: 10, b: 10 };
    const xMax = opts.xMax || Math.max(...values, 0.1) * 1.15;
    const ph   = h - pad.t - pad.b;
    const pw   = w - pad.l - pad.r;
    const n    = labels.length;
    const rh   = Math.min((ph / n) * 0.65, 22);
    const gy   = (ph / n) * 0.35;

    ctx.clearRect(0, 0, w, h);
    labels.forEach((lbl, i) => {
      const y  = pad.t + i * (ph / n) + gy / 2;
      const bw = pw * (values[i] / xMax);
      const color = (opts.colors && opts.colors[i]) || COLORS.accent;

      ctx.fillStyle = COLORS.grid;
      ctx.beginPath(); ctx.roundRect(pad.l, y, pw, rh, 3); ctx.fill();

      ctx.fillStyle = color;
      ctx.beginPath(); ctx.roundRect(pad.l, y, bw, rh, 3); ctx.fill();

      ctx.fillStyle = COLORS.muted;
      ctx.font = '11px system-ui'; ctx.textAlign = 'right';
      ctx.fillText(lbl, pad.l - 8, y + rh / 2 + 4);

      ctx.fillStyle = COLORS.text;
      ctx.font = 'bold 11px system-ui'; ctx.textAlign = 'left';
      ctx.fillText(values[i].toFixed(2) + 'σ', pad.l + bw + 6, y + rh / 2 + 4);
    });
  }

  // ─── Histogram (dual-class) ───────────────────────────────────────────
  function histogram(canvas, normalBins, faultBins, opts = {}) {
    const { ctx, w, h } = _setup(canvas);
    const pad  = { l: 42, r: 16, t: 24, b: 28 };
    const allY = [...(normalBins.y || []), ...(faultBins.y || [])];
    const yMax = Math.max(...allY, 1) * 1.1;
    const xLabels = (normalBins.x || []).map(x => x.toFixed(2));
    const { pw, ph } = _gridAndAxes(ctx, w, h, pad, xLabels, 0, yMax, 4);

    const n  = (normalBins.x || []).length;
    const bw = pw / n;

    (normalBins.y || []).forEach((val, i) => {
      const x  = pad.l + i * bw;
      const bh = ph * (val / yMax);
      ctx.fillStyle = 'rgba(34,197,94,0.55)';
      ctx.fillRect(x + 1, pad.t + ph - bh, bw - 2, bh);
    });
    (faultBins.y || []).forEach((val, i) => {
      const x  = pad.l + i * bw;
      const bh = ph * (val / yMax);
      ctx.fillStyle = 'rgba(239,68,68,0.55)';
      ctx.fillRect(x + 1, pad.t + ph - bh, bw - 2, bh);
    });

    const items = [
      { label: 'Normal', color: 'rgba(34,197,94,0.8)' },
      { label: 'Fault',  color: 'rgba(239,68,68,0.8)' },
    ];
    let lx = pad.l;
    items.forEach(it => {
      ctx.fillStyle = it.color; ctx.fillRect(lx, 4, 10, 10);
      ctx.fillStyle = COLORS.muted; ctx.font = '10px system-ui'; ctx.textAlign = 'left';
      ctx.fillText(it.label, lx + 14, 13);
      lx += ctx.measureText(it.label).width + 30;
    });
  }

  // ─── Waveform oscilloscope ────────────────────────────────────────────
  //
  // FIXED SCALE — never auto-zooms in or out during continuous streaming.
  //
  // A module-level _stableMax ratchets UP when a new frame is larger than
  // current, and decays DOWN at only 0.5% per frame. It is then snapped
  // to the nearest clean tick (0.1 → 0.2 → 0.5 → 1 → 2 → 5 → …) so
  // the Y-axis always shows a round number.
  //
  // Options:
  //   opts.fixedScale  {number}  — explicit ±amplitude (bypasses auto)
  //   opts.scaleMode   {string}  — 'auto' for legacy per-frame zoom
  //   opts.color       {string}  — waveform stroke colour
  //   opts.pad         {object}  — {l,r,t,b} padding override
  //
  function waveform(canvas, data, opts = {}) {
    const { ctx, w, h } = _setup(canvas);
    ctx.clearRect(0, 0, w, h);
    if (!data || !data.length) return;

    const pad  = opts.pad || { l: 38, r: 20, t: 15, b: 22 };
    const pw   = w - pad.l - pad.r;
    const ph   = h - pad.t - pad.b;
    const midY = pad.t + ph / 2;

    // ── Stable amplitude window ──────────────────────────────────────
    if (!waveform._stableMax) waveform._stableMax = 1.0;
    const scaleMode = opts.scaleMode || 'fixed';
    let maxAbs;

    if (scaleMode === 'auto') {
      maxAbs = Math.max(...data.map(v => Math.abs(v)), 0.05) * 1.15;
    } else if (opts.fixedScale != null) {
      maxAbs = opts.fixedScale;
    } else {
      const frameMax = Math.max(...data.map(v => Math.abs(v)), 0.05);
      // Grow fast, decay very slowly (0.5 % per frame)
      waveform._stableMax = Math.max(waveform._stableMax * 0.995, frameMax, 0.05);
      // Snap to a clean tick so axis labels are always round
      const TICKS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100];
      maxAbs = TICKS.find(t => t >= waveform._stableMax * 1.05) || (waveform._stableMax * 1.1);
    }

    // ── Horizontal grid lines + Y-axis labels ────────────────────────
    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
      const frac = i / yTicks;
      const val  = maxAbs * (1 - 2 * frac);   // +max → 0 → -max
      const y    = pad.t + ph * frac;

      // Grid line
      ctx.strokeStyle = 'rgba(255,255,255,0.05)';
      ctx.lineWidth   = 1;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();

      // Label
      ctx.fillStyle = COLORS.muted;
      ctx.font = '9px system-ui';
      ctx.textAlign = 'right';
      ctx.fillText(val.toFixed(2), pad.l - 4, y + 3);
    }

    // ── Zero (centre) line ───────────────────────────────────────────
    ctx.strokeStyle = 'rgba(255,255,255,0.14)';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, midY); ctx.lineTo(pad.l + pw, midY);
    ctx.stroke();

    // ── X-axis sample-number labels ──────────────────────────────────
    ctx.fillStyle = COLORS.muted;
    ctx.font      = '9px system-ui';
    ctx.textAlign = 'center';
    const nSamples = data.length;
    [0, 0.25, 0.5, 0.75, 1].forEach(frac => {
      const x   = pad.l + frac * pw;
      const idx = Math.round(frac * (nSamples - 1));
      ctx.fillText(idx, x, h - 4);
    });

    // ── Waveform line ────────────────────────────────────────────────
    const color = opts.color || '#06b6d4';
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth   = 1.5;
    ctx.lineJoin    = 'round';

    data.forEach((val, i) => {
      const x       = pad.l + (i / (data.length - 1)) * pw;
      // Clamp: never let an extreme sample stretch the visible range
      const clamped = Math.max(-maxAbs, Math.min(maxAbs, val));
      const y       = midY - (clamped / maxAbs) * (ph / 2);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // ─── FFT Frequency spectrum ───────────────────────────────────────────
  function spectrum(canvas, freqs, powers, opts = {}) {
    const { ctx, w, h } = _setup(canvas);
    ctx.clearRect(0, 0, w, h);
    if (!freqs || !freqs.length || !powers || !powers.length) return;

    const pad  = opts.pad || { l: 30, r: 15, t: 15, b: 24 };
    const pw   = w - pad.l - pad.r;
    const ph   = h - pad.t - pad.b;
    const maxP = Math.max(...powers, 0.01) * 1.15;
    const n    = Math.min(freqs.length, powers.length);
    const bw   = Math.max(1, (pw / n) * 0.75);
    const gap  = (pw / n) * 0.25;

    powers.slice(0, n).forEach((p, i) => {
      const x     = pad.l + i * (pw / n) + gap / 2;
      const bh    = ph * (p / maxP);
      const y     = pad.t + ph - bh;
      const ratio = p / maxP;
      ctx.fillStyle = ratio > 0.65 ? '#ef4444' : ratio > 0.35 ? '#f59e0b' : '#06b6d4';
      ctx.fillRect(x, y, bw, bh);
    });

    ctx.fillStyle = COLORS.muted;
    ctx.font = '10px system-ui';
    ctx.textAlign = 'left';
    ctx.fillText('0 Hz', pad.l, h - 4);
    ctx.textAlign = 'right';
    const lastF = freqs[n - 1] || 6000;
    ctx.fillText(`${Math.round(lastF)} Hz`, pad.l + pw, h - 4);
  }

  return { line, bar, hbar, histogram, waveform, spectrum };
})();

window.MachinaCharts = C;
