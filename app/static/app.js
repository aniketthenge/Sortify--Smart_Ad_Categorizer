function toast(msg) {
  let t = document.querySelector('.toast');
  if (!t) { t = document.createElement('div'); t.className = 'toast'; document.body.appendChild(t); }
  t.textContent = msg; t.classList.add('show');
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove('show'), 2400);
}

async function postJSON(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

// Live Site: show the labelled page; fetch a newer one when needed; click an ad to see where it is.
function initLiveSnapshot(section, fresh) {
  const wait = document.getElementById('liveWait');
  const fail = document.getElementById('liveFail');
  const again = document.getElementById('liveRefresh');
  const shot = document.getElementById('shot');
  const hl = document.getElementById('hl');

  async function load() {
    wait.classList.remove('hidden'); fail.classList.add('hidden');
    let tries = 0;
    try {
      const r = await postJSON('/api/live', { section });
      if (r.state === 'ready') return location.reload();
      const poll = async () => {
        const s = await (await fetch('/api/live/status?section=' + encodeURIComponent(section))).json();
        if (s.state === 'ready') return location.reload();
        if (s.state === 'failed' || ++tries > 90) { wait.classList.add('hidden'); fail.classList.remove('hidden'); return; }
        if (s.state === 'idle') await postJSON('/api/live', { section });  // was waiting behind another page
        setTimeout(poll, 2000);
      };
      setTimeout(poll, 2000);
    } catch (e) { wait.classList.add('hidden'); fail.classList.remove('hidden'); }
  }
  if (!fresh) load();
  if (again) again.onclick = load;

  let current = null, next = 0;
  document.querySelectorAll('.live-item').forEach(btn => btn.addEventListener('click', () => {
    const spots = JSON.parse(btn.dataset.spots);
    if (current !== btn) { current = btn; next = 0; }
    const s = spots[next++ % spots.length];
    const scale = shot.clientWidth / parseFloat(getComputedStyle(shot).getPropertyValue('--w'));
    Object.assign(hl.style, { left: s.x * scale + 'px', top: s.y * scale + 'px',
                              width: s.w * scale + 'px', height: s.h * scale + 'px', borderColor: btn.dataset.color });
    hl.classList.remove('hidden'); hl.classList.remove('pulse'); void hl.offsetWidth; hl.classList.add('pulse');
    document.querySelectorAll('.live-item.on').forEach(b => b.classList.remove('on')); btn.classList.add('on');
    shot.scrollTo({ top: Math.max(0, s.y * scale - shot.clientHeight / 3), behavior: 'smooth' });
    if (window.innerWidth < 900) shot.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }));
}
