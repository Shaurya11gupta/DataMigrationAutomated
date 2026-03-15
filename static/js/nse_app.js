/**
 * Name Similarity Engine - Interactive Pipeline Visualization
 * =============================================================
 */

// ── State ──
let _currentResult = null;
let _dictionary = {};

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
  loadExamples();
  loadDictionary();

  // Enter key triggers analysis
  document.getElementById('name-a').addEventListener('keydown', e => {
    if (e.key === 'Enter') runAnalysis();
  });
  document.getElementById('name-b').addEventListener('keydown', e => {
    if (e.key === 'Enter') runAnalysis();
  });
});


// ── Load example chips ──
async function loadExamples() {
  try {
    const resp = await fetch('/api/examples');
    const data = await resp.json();
    const container = document.getElementById('example-chips');
    container.innerHTML = '';

    data.examples.forEach(ex => {
      const chip = document.createElement('button');
      chip.className = 'example-chip';
      chip.textContent = ex.label;
      chip.title = `${ex.a}  vs  ${ex.b}`;
      chip.onclick = () => {
        document.getElementById('name-a').value = ex.a;
        document.getElementById('name-b').value = ex.b;
        runAnalysis();
      };
      container.appendChild(chip);
    });
  } catch (err) {
    console.error('Failed to load examples:', err);
  }
}


// ── Load dictionary ──
async function loadDictionary() {
  try {
    const resp = await fetch('/api/dictionary');
    const data = await resp.json();
    _dictionary = data.dictionary;
    document.getElementById('dict-count').textContent = `${data.size} entries`;
    renderDictionary(_dictionary);
  } catch (err) {
    console.error('Failed to load dictionary:', err);
  }
}


function renderDictionary(dict) {
  const grid = document.getElementById('dict-grid');
  grid.innerHTML = '';

  const entries = Object.entries(dict).sort((a, b) => a[0].localeCompare(b[0]));
  entries.forEach(([abbrev, full]) => {
    const el = document.createElement('div');
    el.className = 'dict-entry';
    el.dataset.abbrev = abbrev;
    el.dataset.full = full;
    el.innerHTML = `
      <span class="dict-abbrev">${abbrev}</span>
      <span class="dict-sep">&rarr;</span>
      <span class="dict-full">${full}</span>
    `;
    grid.appendChild(el);
  });
}


function filterDict() {
  const q = document.getElementById('dict-search').value.toLowerCase().trim();
  const entries = document.querySelectorAll('.dict-entry');
  entries.forEach(el => {
    const show = !q ||
      el.dataset.abbrev.includes(q) ||
      el.dataset.full.includes(q);
    el.style.display = show ? '' : 'none';
  });
}


// ── Main analysis ──
async function runAnalysis() {
  const nameA = document.getElementById('name-a').value.trim();
  const nameB = document.getElementById('name-b').value.trim();

  if (!nameA || !nameB) {
    alert('Please enter both column names.');
    return;
  }

  const btn = document.getElementById('btn-analyze');
  const btnText = btn.querySelector('.btn-text');
  const btnSpinner = btn.querySelector('.btn-spinner');

  btn.disabled = true;
  btnText.style.display = 'none';
  btnSpinner.style.display = 'inline-block';

  // Highlight active architecture node
  highlightArchNode('arch-input');

  try {
    const resp = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name_a: nameA, name_b: nameB }),
    });

    if (!resp.ok) {
      const errData = await resp.json();
      alert(errData.error || 'Analysis failed');
      return;
    }

    _currentResult = await resp.json();
    renderResults(_currentResult);

  } catch (err) {
    alert('Error: ' + err.message);
  } finally {
    btn.disabled = false;
    btnText.style.display = '';
    btnSpinner.style.display = 'none';
  }
}


// ── Render all results with staged animation ──
function renderResults(r) {
  const container = document.getElementById('results-container');
  container.style.display = 'block';

  // Smooth scroll to results
  setTimeout(() => {
    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 100);

  // Animate through architecture nodes
  const nodes = ['arch-regex', 'arch-seg', 'arch-expand', 'arch-sim'];
  nodes.forEach((id, i) => {
    setTimeout(() => highlightArchNode(id), 200 + i * 400);
  });

  // Render each step with delays
  setTimeout(() => renderStep1(r), 200);
  setTimeout(() => renderStep2(r), 500);
  setTimeout(() => renderStep3(r), 800);
  setTimeout(() => renderStep4(r), 1100);

  // Reset step animations
  document.querySelectorAll('.step-section').forEach(el => {
    el.style.animation = 'none';
    el.offsetHeight; // reflow
    el.style.animation = '';
  });
}


// ── Architecture node highlighting ──
function highlightArchNode(id) {
  document.querySelectorAll('.arch-node').forEach(n => n.classList.remove('active'));
  const node = document.getElementById(id);
  if (node) node.classList.add('active');
}


// ── Step 1: Regex Split ──
function renderStep1(r) {
  const s1 = r.step1_regex;

  document.getElementById('s1-input-a').innerHTML =
    `<span class="token-chip token-chip-input">${escHtml(s1.a.input)}</span>`;
  document.getElementById('s1-input-b').innerHTML =
    `<span class="token-chip token-chip-input">${escHtml(s1.b.input)}</span>`;

  document.getElementById('s1-output-a').innerHTML =
    s1.a.tokens.map((t, i) =>
      `<span class="token-chip token-chip-output" style="animation-delay:${i * 0.08}s">${escHtml(t)}</span>`
    ).join(' ');

  document.getElementById('s1-output-b').innerHTML =
    s1.b.tokens.map((t, i) =>
      `<span class="token-chip token-chip-output" style="animation-delay:${i * 0.08}s">${escHtml(t)}</span>`
    ).join(' ');
}


// ── Step 2: Segmentation ──
function renderStep2(r) {
  const s2 = r.step2_segmentation;

  document.getElementById('s2-table-a').innerHTML = s2.a.map(seg =>
    `<div class="seg-row">
       <span class="seg-input">${escHtml(seg.input)}</span>
       <span class="seg-arrow">&rarr;</span>
       <div class="seg-parts">
         ${seg.output.map(p => `<span class="seg-part">${escHtml(p)}</span>`).join('')}
       </div>
     </div>`
  ).join('');

  document.getElementById('s2-table-b').innerHTML = s2.b.map(seg =>
    `<div class="seg-row">
       <span class="seg-input">${escHtml(seg.input)}</span>
       <span class="seg-arrow">&rarr;</span>
       <div class="seg-parts">
         ${seg.output.map(p => `<span class="seg-part">${escHtml(p)}</span>`).join('')}
       </div>
     </div>`
  ).join('');
}


// ── Step 3: Expansion ──
function renderStep3(r) {
  const s3 = r.step3_expansion;

  document.getElementById('s3-cards-a').innerHTML = s3.a.map(renderExpandCard).join('');
  document.getElementById('s3-cards-b').innerHTML = s3.b.map(renderExpandCard).join('');
}


function renderExpandCard(item) {
  const srcClass = `src-${item.source}`;
  const badgeClass = `badge-${item.source}`;

  let details = [];

  if (item.dict_lookup !== null) {
    details.push(`<div class="expand-detail-item"><span class="detail-dot dot-dict"></span> Dict: "${item.dict_lookup}"</div>`);
  } else {
    details.push(`<div class="expand-detail-item"><span class="detail-dot dot-dict"></span> Dict: not found</div>`);
  }

  if (item.classifier_prob !== null) {
    const clsResult = item.classifier_expand ? 'EXPAND' : 'KEEP';
    const clsColor = item.classifier_expand ? 'color:var(--accent-green)' : 'color:var(--text-muted)';
    details.push(`<div class="expand-detail-item"><span class="detail-dot dot-clf"></span> Classifier: <span style="${clsColor}">${clsResult}</span> (${(item.classifier_prob * 100).toFixed(1)}%)</div>`);
  }

  if (item.model_output !== null) {
    details.push(`<div class="expand-detail-item"><span class="detail-dot dot-expander"></span> Expander: "${item.model_output}"</div>`);
  }

  return `
    <div class="expand-card">
      <div class="expand-card-header">
        <span class="expand-token-in">${escHtml(item.input)}</span>
        <span class="expand-arrow">&rarr;</span>
        <span class="expand-token-out ${srcClass}">${escHtml(item.final)}</span>
        <span class="expand-source-badge ${badgeClass}">${item.source}</span>
      </div>
      <div class="expand-detail-row">${details.join('')}</div>
    </div>
  `;
}


// ── Step 4: Similarity ──
function renderStep4(r) {
  const s4 = r.step4_similarity;

  // Highlight conflicting tokens in expanded phrases
  const tca = s4.token_conflicts_a || {};
  const tcb = s4.token_conflicts_b || {};

  function highlightPhrase(phrase, tokenConflicts) {
    const words = phrase.split(' ');
    return words.map(w => {
      if (tokenConflicts[w.toLowerCase()]) {
        return `<span class="token-conflict-highlight" title="${tokenConflicts[w.toLowerCase()].join(', ')}">${w}</span>`;
      }
      return w;
    }).join(' ');
  }

  document.getElementById('s4-phrase-a').innerHTML = highlightPhrase(s4.phrase_a, tca);
  document.getElementById('s4-phrase-b').innerHTML = highlightPhrase(s4.phrase_b, tcb);

  // Score circle
  const score = s4.final_similarity;
  const pct = Math.max(0, Math.min(1, score));
  const circumference = 2 * Math.PI * 54; // r=54
  const offset = circumference * (1 - pct);
  const ring = document.getElementById('score-ring-fill');

  // Color based on score
  let ringColor, scoreColor;
  if (score >= 0.85) {
    ringColor = '#10b981';
    scoreColor = 'var(--accent-green)';
  } else if (score >= 0.65) {
    ringColor = '#f59e0b';
    scoreColor = 'var(--accent-amber)';
  } else {
    ringColor = '#f43f5e';
    scoreColor = 'var(--accent-rose)';
  }

  ring.style.stroke = ringColor;
  ring.style.strokeDashoffset = offset;

  const scoreValue = document.getElementById('s4-score-value');
  scoreValue.textContent = (score * 100).toFixed(1) + '%';
  scoreValue.style.color = scoreColor;

  const confidence = document.getElementById('s4-confidence');
  confidence.textContent = s4.confidence;
  confidence.style.color = scoreColor;

  // Bars
  animateBar('bar-base', s4.base_similarity);
  animateBar('bar-penalty', s4.penalty);
  animateBar('bar-final', s4.final_similarity);

  document.getElementById('val-base').textContent = (s4.base_similarity * 100).toFixed(1) + '%';
  document.getElementById('val-penalty').textContent = (s4.penalty * 100).toFixed(1) + '%';
  document.getElementById('val-final').textContent = (s4.final_similarity * 100).toFixed(1) + '%';

  // Conflicts
  const conflictsSection = document.getElementById('conflicts-section');
  const conflictsList = document.getElementById('conflicts-list');

  if (s4.conflicts && s4.conflicts.length > 0) {
    conflictsSection.style.display = '';

    const typeLabels = {
      directional: 'Directional Conflict',
      role: 'Role Conflict',
      temporal: 'Temporal Conflict',
      unit: 'Unit Mismatch',
    };
    const typeIcons = {
      directional: '\u2194',  // arrows
      role: '\u26A0',         // warning
      temporal: '\u23F0',     // clock
      unit: '\u2696',         // scales
    };

    let html = '';

    // Overall conflict badges
    html += '<div class="conflict-badges">';
    html += s4.conflicts.map(c =>
      `<span class="conflict-badge">${typeIcons[c] || ''} ${typeLabels[c] || c}</span>`
    ).join('');
    html += '</div>';

    // Per-token conflict detail table
    if (s4.conflict_details && s4.conflict_details.length > 0) {
      html += '<table class="conflict-detail-table">';
      html += '<thead><tr><th>Type</th><th>Token A</th><th>vs</th><th>Token B</th><th>Penalty Weight</th></tr></thead>';
      html += '<tbody>';
      // Deduplicate conflict details by token pair
      const seen = new Set();
      for (const d of s4.conflict_details) {
        const key = `${d.type}:${d.token_a}:${d.token_b}`;
        if (seen.has(key)) continue;
        seen.add(key);
        html += `<tr>
          <td><span class="conflict-type-badge conflict-type-${d.type}">${typeIcons[d.type] || ''} ${typeLabels[d.type] || d.type}</span></td>
          <td class="conflict-token conflict-token-a">${d.token_a}</td>
          <td class="conflict-vs">\u2260</td>
          <td class="conflict-token conflict-token-b">${d.token_b}</td>
          <td class="conflict-weight">${(d.weight * 100).toFixed(0)}%</td>
        </tr>`;
      }
      html += '</tbody></table>';
    }

    conflictsList.innerHTML = html;
  } else {
    conflictsSection.style.display = 'none';
  }

  // Runtime
  document.getElementById('runtime-badge').textContent =
    `Pipeline completed in ${r.runtime_ms.toFixed(1)} ms` +
    (r.models_loaded ? ' (ML models active)' : ' (dictionary-only fallback)');
}


function animateBar(id, value) {
  const bar = document.getElementById(id);
  // Reset then animate
  bar.style.width = '0%';
  setTimeout(() => {
    bar.style.width = (Math.min(1, Math.max(0, value)) * 100) + '%';
  }, 50);
}


// ── Utilities ──
function escHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
