// Shared helpers for every page (no framework, no build step, no CDN — works offline for live demos).

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || res.statusText);
  return body;
}

function fmtMs(ms) {
  if (ms == null) return '–';
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}
const fmtUsd = v => v == null ? '–' : `$${Number(v).toFixed(v < 0.01 ? 4 : 3)}`;
const fmtTime = iso => iso ? new Date(iso).toLocaleString() : '–';

const OUTCOME = {
  passed: { icon: '✔', label: 'passed', cls: 'status-passed' },
  escalated: { icon: '⚑', label: 'escalated', cls: 'status-escalated' },
  failed: { icon: '✖', label: 'failed', cls: 'status-failed' },
  error: { icon: '✖', label: 'error', cls: 'status-error' },
  running: { icon: '◌', label: 'running', cls: '' },
};
function outcomeChip(run) {
  const key = run.status === 'running' ? 'running' : (run.outcome || run.status);
  const o = OUTCOME[key] || { icon: '•', label: key, cls: '' };
  return `<span class="chip ${o.cls}">${o.icon} ${esc(o.label)}</span>`;
}
const archChip = a => `<span class="chip arch-${esc(a)}"><span class="dot"></span>${esc(a)}</span>`;

function diffHtml(diff) {
  if (!diff) return '<div class="empty">No patch</div>';
  const lines = diff.split('\n').map(l => {
    let c = '';
    if (l.startsWith('+++') || l.startsWith('---') || l.startsWith('diff --git')) c = 'file';
    else if (l.startsWith('@@')) c = 'hunk';
    else if (l.startsWith('+')) c = 'add';
    else if (l.startsWith('-')) c = 'del';
    return `<span class="ln ${c}">${esc(l) || ' '}</span>`;
  });
  return `<div class="diff"><pre>${lines.join('')}</pre></div>`;
}

// ------------------------------------------------------------ tooltip
let _tip;
function tip(html, ev) {
  _tip = _tip || Object.assign(document.body.appendChild(document.createElement('div')), { className: 'tooltip' });
  if (!html) { _tip.style.display = 'none'; return; }
  _tip.innerHTML = html;
  _tip.style.display = 'block';
  const x = Math.min(ev.clientX + 14, window.innerWidth - _tip.offsetWidth - 8);
  const y = Math.min(ev.clientY + 14, window.innerHeight - _tip.offsetHeight - 8);
  _tip.style.left = `${x}px`; _tip.style.top = `${y}px`;
}
function bindTips(root) {
  $$('[data-tip]', root).forEach(el => {
    el.addEventListener('mousemove', e => tip(el.dataset.tip, e));
    el.addEventListener('mouseleave', () => tip(null));
  });
}

// ------------------------------------------------------------ theme + nav
function initChrome(active) {
  const saved = (() => { try { return localStorage.getItem('cf-theme'); } catch { return null; } })();
  if (saved) document.documentElement.dataset.theme = saved;
  const nav = document.createElement('header');
  nav.className = 'topbar';
  nav.innerHTML = `
    <a class="brand" href="/"><span class="logo">CF</span>CodeFix AI</a>
    <nav class="nav">
      <a href="/" class="${active === 'new' ? 'active' : ''}">New fix</a>
      <a href="/runs.html" class="${active === 'runs' ? 'active' : ''}">Runs</a>
      <a href="/dashboard.html" class="${active === 'dash' ? 'active' : ''}">Dashboard</a>
    </nav>
    <span class="spacer"></span>
    <button class="theme-btn" id="themeBtn" title="Toggle theme">◐ Theme</button>`;
  document.body.prepend(nav);
  $('#themeBtn').onclick = () => {
    const cur = document.documentElement.dataset.theme
      || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('cf-theme', next); } catch { /* storage unavailable */ }
  };
}

// ------------------------------------------------------------ charts (hand-rolled SVG)
// Horizontal bar chart. items: [{label, value, color?, tip?}] ; highlight index gets an annotation.
function hbar(el, items, { fmt = v => v, highlight = -1, note = '' } = {}) {
  if (!items.length) { el.innerHTML = '<div class="empty">No data yet</div>'; return; }
  const W = 560, rowH = 30, left = 110, right = 70, top = 6;
  const H = top + items.length * rowH + 22;
  const max = Math.max(...items.map(i => i.value), 1e-9);
  const x = v => left + (W - left - right) * (v / max);
  const ticks = [0, 0.5, 1].map(f => f * max);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
  ticks.forEach(t => {
    s += `<line class="gridline" x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${H - 20}"/>`;
    s += `<text x="${x(t)}" y="${H - 6}" text-anchor="middle">${esc(fmt(t))}</text>`;
  });
  items.forEach((it, i) => {
    const y = top + i * rowH + 6, h = rowH - 12, w = Math.max(2, x(it.value) - left);
    const fill = it.color || 'var(--series-1)';
    s += `<text class="label" x="${left - 10}" y="${y + h / 2 + 4}" text-anchor="end">${esc(it.label)}</text>`;
    s += `<path class="bar" d="M${left},${y} h${w - 4} a4,4 0 0 1 4,4 v${h - 8} a4,4 0 0 1 -4,4 h-${w - 4} z" fill="${fill}"
            data-tip="${esc(it.tip || `<b>${esc(it.label)}</b>${esc(fmt(it.value))}`)}"/>`;
    s += `<text class="value" x="${left + w + 6}" y="${y + h / 2 + 4}">${esc(fmt(it.value))}${i === highlight ? ' ◀ ' + esc(note) : ''}</text>`;
  });
  el.innerHTML = s + '</svg>';
  bindTips(el);
}

// Grouped vertical bars: groups [{label, values:[{key,value,color}]}]
function vgroups(el, groups, keys, { fmt = v => v, maxV = null } = {}) {
  if (!groups.length) { el.innerHTML = '<div class="empty">No data yet</div>'; return; }
  const W = 560, H = 220, left = 36, bottom = 28, top = 10;
  const max = maxV ?? Math.max(...groups.flatMap(g => g.values.map(v => v.value)), 1e-9);
  const gw = (W - left) / groups.length, bw = Math.min(26, (gw - 16) / keys.length - 2);
  const y = v => top + (H - top - bottom) * (1 - v / max);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
  [0, 0.5, 1].forEach(f => {
    const v = f * max;
    s += `<line class="gridline" x1="${left}" x2="${W}" y1="${y(v)}" y2="${y(v)}"/><text x="${left - 6}" y="${y(v) + 4}" text-anchor="end">${esc(fmt(v))}</text>`;
  });
  groups.forEach((g, gi) => {
    const gx = left + gi * gw + (gw - keys.length * (bw + 2)) / 2;
    g.values.forEach((v, vi) => {
      const bx = gx + vi * (bw + 2), by = y(v.value), bh = Math.max(0, H - bottom - by);
      if (bh > 0) s += `<path class="bar" d="M${bx},${H - bottom} v-${Math.max(bh - 4, 0)} a4,4 0 0 1 4,-4 h${bw - 8} a4,4 0 0 1 4,4 v${Math.max(bh - 4, 0)} z" fill="${v.color}"
          data-tip="<b>${esc(g.label)} · ${esc(v.key)}</b>${esc(fmt(v.value))}"/>`;
    });
    s += `<text class="label" x="${left + gi * gw + gw / 2}" y="${H - 8}" text-anchor="middle">${esc(g.label)}</text>`;
  });
  s += `<line class="axis" x1="${left}" x2="${W}" y1="${H - bottom}" y2="${H - bottom}"/>`;
  el.innerHTML = s + '</svg>';
  bindTips(el);
}

// Gantt: spans [{agent,start,end,status}] -> one row per agent
const AGENT_ORDER = ['manager', 'diagnoser', 'fixer', 'tester', 'pr'];
const AGENT_COLOR = { manager: 'var(--series-5)', diagnoser: 'var(--series-1)', fixer: 'var(--series-2)', tester: 'var(--series-3)', pr: 'var(--series-4)' };
function gantt(el, spans, managerCalls = []) {
  const all = [...spans, ...managerCalls];
  if (!all.length) { el.innerHTML = '<div class="empty">No agent activity yet</div>'; return; }
  const t0 = Math.min(...all.map(s => Date.parse(s.start)));
  const t1 = Math.max(...all.map(s => Date.parse(s.end)), t0 + 1);
  const agents = AGENT_ORDER.filter(a => all.some(s => s.agent === a));
  const W = 760, left = 90, rowH = 30, top = 4, H = top + agents.length * rowH + 24;
  const x = t => left + (W - left - 10) * ((t - t0) / (t1 - t0));
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
  for (let i = 0; i <= 4; i++) {
    const t = t0 + (t1 - t0) * i / 4;
    s += `<line class="gridline" x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${H - 20}"/><text x="${x(t)}" y="${H - 6}" text-anchor="middle">${((t - t0) / 1000).toFixed(1)}s</text>`;
  }
  agents.forEach((a, i) => {
    const y = top + i * rowH + 6;
    s += `<text class="label" x="${left - 10}" y="${y + 13}" text-anchor="end">${a}</text>`;
    all.filter(sp => sp.agent === a).forEach(sp => {
      const sx = x(Date.parse(sp.start)), w = Math.max(3, x(Date.parse(sp.end)) - sx);
      const fail = sp.status === 'failure';
      s += `<rect class="bar" x="${sx}" y="${y}" width="${w}" height="${rowH - 12}" rx="4" fill="${AGENT_COLOR[a]}"
             ${fail ? 'stroke="var(--critical)" stroke-width="2.5"' : ''}
             data-tip="<b>${a} ${fail ? '✖ failure' : '✔ ' + (sp.status || 'ok')}</b>${esc(sp.kind || '')} · ${((Date.parse(sp.end) - Date.parse(sp.start)) / 1000).toFixed(2)}s"/>`;
    });
  });
  el.innerHTML = s + '</svg>';
  bindTips(el);
}
