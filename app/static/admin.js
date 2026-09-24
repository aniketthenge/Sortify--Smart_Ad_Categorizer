// Staff page only - never loaded by customer pages.

function initLabelling() {
  document.querySelectorAll('.label-select').forEach(sel => {
    sel.addEventListener('change', async () => {
      try {
        await postJSON(`/api/admin/ads/${sel.dataset.id}/label`, { category: sel.value });
        sel.closest('.review-row').classList.add('done');
        toast('Saved');
      } catch (e) { toast('Error: ' + e.message); }
    });
  });
}

function initRetrain() {
  const btn = document.getElementById('retrainBtn');
  const msg = document.getElementById('retrainMsg');
  btn.onclick = async () => {
    btn.disabled = true; btn.textContent = 'Retraining… (about a minute)';
    try {
      const r = await postJSON('/api/admin/retrain');
      msg.textContent = `Retrained on ${r.n_samples} examples. Accuracy ${(r.accuracy * 100).toFixed(1)}%. Updated ${r.reclassified} ads.`;
      msg.classList.remove('hidden'); setTimeout(() => location.reload(), 1800);
    } catch (e) { toast('Error: ' + e.message); btn.disabled = false; }
  };
}

function initAdminRefresh() {
  const btn = document.getElementById('refreshBtn');
  const busy = on => { btn.disabled = on; btn.textContent = on ? 'Collecting ads… (2–5 min)' : 'Collect latest ads now'; };
  async function poll() {
    const s = await (await fetch('/api/admin/refresh/status')).json();
    if (s.running) { busy(true); setTimeout(poll, 3000); return; }
    busy(false);
    if (s.result) toast(s.result.status === 'success' ? `${s.result.new} new ads` : 'Collection failed - see the console window');
  }
  btn.onclick = async () => { busy(true); await postJSON('/api/admin/refresh'); poll(); };
  poll();
}

function initLiveWindow() {
  const btn = document.getElementById('liveWinBtn');
  const sel = document.getElementById('liveWinSection');
  async function watch() {
    const s = await (await fetch('/api/admin/live-window/status')).json();
    btn.disabled = s.running; btn.textContent = s.running ? 'Window open' : 'Open in a window';
    if (s.running) setTimeout(watch, 2000);
  }
  btn.onclick = async () => { btn.disabled = true; await postJSON('/api/admin/live-window', { section: sel.value }); setTimeout(watch, 1500); };
  watch();
}
