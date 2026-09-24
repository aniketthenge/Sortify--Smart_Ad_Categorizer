function toast(msg) {
  let t = document.querySelector('.toast');
  if (!t) { t = document.createElement('div'); t.className = 'toast'; document.body.appendChild(t); }
  t.textContent = msg; t.classList.add('show');
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove('show'), 2200);
}

async function postJSON(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

// Dashboard: start a live scrape and stream its log.
function initScrape() {
  const btn = document.getElementById('scrapeBtn');
  const panel = document.getElementById('scrapePanel');
  const logEl = document.getElementById('scrapeLog');
  const status = document.getElementById('scrapeStatus');
  if (!btn) return;

  async function poll() {
    const s = await (await fetch('/api/scrape/status')).json();
    logEl.textContent = s.log.join('\n'); logEl.scrollTop = logEl.scrollHeight;
    if (s.running) { status.textContent = 'running…'; status.className = 'pill running'; setTimeout(poll, 1500); return; }
    btn.disabled = false; btn.textContent = 'Scrape live ads now';
    if (s.result) {
      status.textContent = s.result.status; status.className = 'pill ' + s.result.status;
      if (s.result.status === 'success') { toast(`${s.result.new} new ads added`); setTimeout(() => location.reload(), 1500); }
    }
  }
  btn.onclick = async () => {
    btn.disabled = true; btn.textContent = 'Scraping…'; panel.classList.remove('hidden');
    try { await postJSON('/api/scrape'); } catch (e) { toast(e.message); }
    poll();
  };
  // resume the view if a scrape is already in progress
  fetch('/api/scrape/status').then(r => r.json()).then(s => {
    if (s.running) { panel.classList.remove('hidden'); btn.disabled = true; btn.textContent = 'Scraping…'; poll(); }
  });
}

// Ads page: relabel from the dropdown.
function initLabelling() {
  document.querySelectorAll('.label-select').forEach(sel => {
    sel.addEventListener('change', async () => {
      try {
        await postJSON(`/api/ads/${sel.dataset.id}/label`, { category: sel.value });
        sel.closest('.ad').classList.remove('review');
        sel.parentElement.querySelector('.conf').textContent = '✔ human label';
        toast('Saved. Retrain on the Model page to learn from it.');
      } catch (e) { toast('Error: ' + e.message); }
    });
  });
}

// Model page: retrain.
function initRetrain() {
  const btn = document.getElementById('retrainBtn');
  const msg = document.getElementById('retrainMsg');
  if (!btn) return;
  btn.onclick = async () => {
    btn.disabled = true; btn.textContent = 'Training… (≈15 s)';
    try {
      const r = await postJSON('/api/retrain');
      msg.textContent = `Retrained on ${r.meta.n_samples} examples. Accuracy ${(r.meta.evaluation.accuracy * 100).toFixed(1)}%. Re-classified ${r.reclassified} ads.`;
      msg.classList.remove('hidden'); setTimeout(() => location.reload(), 1800);
    } catch (e) { toast('Error: ' + e.message); btn.disabled = false; }
  };
}
