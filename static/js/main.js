/* ============================================================
   DataMigrationCrane – Shared JS Utilities
   ============================================================ */

function showLoading(text) {
  const overlay = document.getElementById('loading-overlay');
  const loadingText = document.getElementById('loading-text');
  if (overlay) overlay.style.display = 'flex';
  if (loadingText) loadingText.textContent = text || 'Processing...';
}

function hideLoading() {
  const overlay = document.getElementById('loading-overlay');
  if (overlay) overlay.style.display = 'none';
}

/* POST JSON to an API endpoint */
async function apiPost(url, data) {
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return resp.json();
}

/* Simple tab switching */
function initTabs(container) {
  const btns = container.querySelectorAll('.tab-btn');
  const panels = container.querySelectorAll('.tab-panel');
  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add('active');
    });
  });
}

/* Format a confidence value as colored class */
function confClass(v) {
  if (v >= 0.75) return 'conf-high';
  if (v >= 0.5) return 'conf-med';
  return 'conf-low';
}

function confPercent(v) {
  return Math.round(v * 100) + '%';
}
