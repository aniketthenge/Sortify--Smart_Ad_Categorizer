"""
Live view: open the real lokmattimes.com in a visible browser window and label
every ad on the page with its category, in place, as the ads load.

A background thread drives the window with Playwright. Every ~1.5 s it looks
for sponsored cards / ad frames it hasn't labelled yet, categorises them, and
injects an outline + category badge onto each one, plus a small panel listing
all ads on the page (click an entry to scroll to that ad). It keeps working as
the visitor scrolls, clicks through to other Lokmat pages, or new ads load.
"""
from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from typing import Callable

from .scraper import BASE_URL, GOOGLE_FRAME_JS, USER_AGENT, _parse_google, _parse_taboola, launch_browser

# Find sponsored Taboola cards not labelled yet; tag each with a stable data attribute.
SCAN_JS = r"""
() => {
  window.__acSeq = window.__acSeq || 0;
  const out = [];
  document.querySelectorAll('.videoCube.trc_spotlight_item.syndicatedItem:not([data-ac-key])').forEach(card => {
    const key = 'ac' + (++window.__acSeq);
    card.setAttribute('data-ac-key', key);
    const a = card.querySelector('a[href]');
    const pick = sel => { const e = card.querySelector(sel); return e ? e.innerText.trim() : ''; };
    out.push({ key, sponsored: true, title: pick('.video-title, .trc_ellipsis, [class*=title]'),
               description: pick('.video-description, [class*=description]'),
               advertiser: pick('.branding, [class*=branding]'), raw_text: card.innerText.trim(),
               landing_url: a ? a.href : '', image_url: '' });
  });
  return out;
}
"""

# Draw outline + badge on one element and register it in the page panel.
LABEL_JS = r"""
([el, info]) => {
  if (!el || el.getAttribute('data-ac-done')) return;
  el.setAttribute('data-ac-done', '1');
  el.setAttribute('data-ac-id', info.id);
  const box = el.tagName === 'IFRAME' ? (el.parentElement || el) : el;
  if (getComputedStyle(box).position === 'static') box.style.position = 'relative';
  box.style.outline = `3px solid ${info.color}`;
  box.style.outlineOffset = '-3px';
  const b = document.createElement('ac-badge');
  b.textContent = info.category;
  b.style.cssText = `position:absolute;top:6px;left:6px;z-index:2147483646;background:${info.color};color:#fff;
    font:600 12px/1.2 system-ui,Segoe UI,sans-serif;padding:4px 8px;border-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,.35);
    pointer-events:none;text-shadow:0 1px 1px rgba(0,0,0,.35);max-width:90%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;`;
  box.appendChild(b);

  // Panel (Shadow DOM so the site's CSS can't restyle it)
  let host = document.getElementById('ac-panel-host');
  if (!host) {
    host = document.createElement('div'); host.id = 'ac-panel-host';
    host.style.cssText = 'position:fixed;right:16px;bottom:16px;z-index:2147483647;';
    document.documentElement.appendChild(host);
    const root = host.attachShadow({ mode: 'open' });
    root.innerHTML = `<style>
      .p{width:320px;max-height:60vh;display:flex;flex-direction:column;background:#fff;color:#1d2330;border:1px solid #e4e7ec;
         border-radius:10px;box-shadow:0 8px 30px rgba(0,0,0,.18);font:13px/1.4 system-ui,Segoe UI,sans-serif;overflow:hidden}
      .h{display:flex;align-items:center;gap:8px;padding:10px 12px;border-bottom:1px solid #e4e7ec;font-weight:700;cursor:pointer}
      .h .logo{background:#c8102e;color:#fff;border-radius:5px;padding:1px 5px;font-size:11px}
      .h .n{margin-left:auto;font-weight:500;color:#6b7385}
      .l{overflow:auto;padding:4px 0}
      .i{display:flex;gap:8px;padding:6px 12px;cursor:pointer;align-items:flex-start}
      .i:hover{background:#f3f4f6}
      .d{width:9px;height:9px;border-radius:50%;flex:none;margin-top:5px}
      .c{font-weight:600} .t{color:#6b7385;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:260px}
      .x{margin-left:auto;color:#6b7385;font-size:12px;padding-left:6px}
      .min .l{display:none}
    </style><div class="p"><div class="h"><span class="logo">AC</span>Ads on this page<span class="n">0 ads</span></div><div class="l"></div></div>`;
    root.querySelector('.h').onclick = () => root.querySelector('.p').classList.toggle('min');
  }
  const root = host.shadowRoot;
  // One row per distinct ad; repeats of the same ad further down the feed are counted on that row.
  window.__acRows = window.__acRows || {};
  let entry = window.__acRows[info.id];
  if (!entry) {
    const row = document.createElement('div'); row.className = 'i';
    row.innerHTML = `<span class="d" style="background:${info.color}"></span><div><div class="c"></div><div class="t"></div></div><span class="x"></span>`;
    row.querySelector('.c').textContent = info.category;
    row.querySelector('.t').textContent = info.title;
    entry = window.__acRows[info.id] = { row, boxes: [], next: 0 };
    row.onclick = () => {
      entry.boxes = entry.boxes.filter(b => b.isConnected);
      if (!entry.boxes.length) return;
      const target = entry.boxes[entry.next++ % entry.boxes.length];
      target.scrollIntoView({ behavior: 'smooth', block: 'center' });
      target.animate([{ boxShadow: `0 0 0 0 ${info.color}` }, { boxShadow: '0 0 0 14px transparent' }], { duration: 900, iterations: 2 });
    };
    root.querySelector('.l').appendChild(row);
  }
  entry.boxes.push(box);
  entry.row.querySelector('.x').textContent = entry.boxes.length > 1 ? '×' + entry.boxes.length : '';
  root.querySelector('.n').textContent = Object.keys(window.__acRows).length + ' ads';
}
"""


class LiveView:
    """One visible browser window at a time."""

    def __init__(self, categorize: Callable[[list[dict]], list[dict]], colors: dict[str, str],
                 store: Callable[[list[dict]], None] | None = None, log: Callable[[str], None] | None = None):
        self.categorize, self.colors, self.store = categorize, colors, store
        self.log = log or (lambda line: print(line, flush=True))
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def open(self, path: str = "/") -> bool:
        if self.running:
            return False
        self.error = None
        url = path if path.startswith("http") else BASE_URL + path
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run(url)), daemon=True)
        self._thread.start()
        return True

    async def _launch(self, p):
        return await launch_browser(p, headless=False, args=["--start-maximized"])

    async def _run(self, url: str):
        from playwright.async_api import async_playwright
        try:
            async with async_playwright() as p:
                browser = await self._launch(p)
                context = await browser.new_context(user_agent=USER_AGENT, no_viewport=True)
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                self.log(f"[live] opened {url}")
                while not page.is_closed() and browser.is_connected():
                    try:
                        await self._label_new_ads(page)
                    except Exception as exc:  # page navigating / frame detached: try again next tick
                        if page.is_closed():
                            break
                        self.log(f"[live] retry: {type(exc).__name__}")
                    await asyncio.sleep(1.5)
                if browser.is_connected():
                    await browser.close()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.log(f"[live] ERROR {self.error}")
        self.log("[live] window closed")

    async def _label_new_ads(self, page):
        n = await label_new_ads(page, self.categorize, self.colors, self.store)
        if n:
            self.log(f"[live] labelled {n} ads")


async def label_new_ads(page, categorize, colors, store=None) -> int:
    """Find ads on `page` not labelled yet, categorise them and draw outline + badge + panel entry."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    found: list[tuple] = []  # (element handle, ad dict)

    for card in await page.evaluate(SCAN_JS):
        ad = _parse_taboola(card, page.url, now)
        if ad:
            el = await page.query_selector(f'[data-ac-key="{card["key"]}"]')
            if el:
                found.append((el, ad))

    for frame in page.frames:
        if frame == page.main_frame or not (
                (frame.name or "").startswith("google_ads_iframe")
                or any(k in (frame.url or "") for k in ("googlesyndication", "safeframe"))):
            continue
        try:
            el = await frame.frame_element()
            if await el.get_attribute("data-ac-done") or not await el.is_visible():
                continue
            ad = _parse_google(await frame.evaluate(GOOGLE_FRAME_JS), page.url, now)
        except Exception:
            continue
        if ad:
            found.append((el, ad))

    if not found:
        return 0
    results = categorize([ad for _, ad in found])
    for (el, ad), res in zip(found, results):
        info = {"id": ad["id"], "category": res["category"], "color": colors.get(res["category"], "#666"),
                "title": ad["title"][:90]}
        await el.evaluate("(el, info) => (" + LABEL_JS + ")([el, info])", info)
    if store:
        store([ad for _, ad in found])
    return len(found)


# Positions (page coordinates) of every labelled ad, for the customer-facing snapshot.
POSITIONS_JS = r"""
() => {
  const host = document.getElementById('ac-panel-host'); if (host) host.remove();
  const out = [];
  document.querySelectorAll('[data-ac-done]').forEach(el => {
    const box = el.tagName === 'IFRAME' ? (el.parentElement || el) : el;
    const r = box.getBoundingClientRect();
    if (r.width < 20 || r.height < 20) return;
    out.push({ x: Math.round(r.left + scrollX), y: Math.round(r.top + scrollY),
               w: Math.round(r.width), h: Math.round(r.height), id: el.getAttribute('data-ac-id') || '' });
  });
  return out;
}
"""


class Snapshotter:
    """Customer-facing live view that works from any computer: the server opens the real page,
    scrolls it like a visitor, labels the ads, and saves a picture plus each ad's position.
    One capture at a time; results are cached per section for `ttl` seconds."""

    WIDTH, MAX_HEIGHT, SCROLLS = 1280, 14000, 14

    def __init__(self, categorize, colors, store=None, out_dir=".", ttl: int = 600, log=None):
        from pathlib import Path
        self.categorize, self.colors, self.store, self.ttl = categorize, colors, store, ttl
        self.out_dir = Path(out_dir)
        self.log = log or (lambda line: print(line, flush=True))
        self._lock = threading.Lock()
        self._busy: str | None = None
        self.results: dict[str, dict] = {}
        self.errors: dict[str, str] = {}

    @staticmethod
    def key(section: str) -> str:
        return section.strip("/").replace("/", "-") or "home"

    def fresh(self, section: str) -> dict | None:
        import time
        r = self.results.get(self.key(section))
        return r if r and time.time() - r["taken_ts"] < self.ttl else None

    def latest(self, section: str) -> dict | None:
        return self.results.get(self.key(section))

    def busy(self, section: str | None = None) -> bool:
        return self._busy is not None and (section is None or self._busy == self.key(section))

    def request(self, section: str) -> str:
        """Start a capture unless a fresh one exists. Returns 'ready', 'working', 'queued' or 'failed'."""
        if self.fresh(section):
            return "ready"
        if self.busy(section):
            return "working"
        if not self._lock.acquire(blocking=False):
            return "queued"  # another section is being captured; the page polls again
        self._busy = self.key(section)
        self.errors.pop(self._busy, None)
        threading.Thread(target=self._run, args=(section,), daemon=True).start()
        return "working"

    def _run(self, section: str):
        key = self.key(section)
        try:
            self.results[key] = asyncio.run(self._capture(section, key))
        except Exception as exc:
            self.errors[key] = f"{type(exc).__name__}: {exc}"
            self.log(f"[snapshot] ERROR {section}: {self.errors[key]}")
        finally:
            self._busy = None
            self._lock.release()

    async def _capture(self, section: str, key: str) -> dict:
        import time
        from playwright.async_api import async_playwright
        ads: dict[str, dict] = {}

        def categorize(batch):
            res = self.categorize(batch)
            for ad, r in zip(batch, res):
                ads[ad["id"]] = {"title": ad["title"], "advertiser": ad.get("advertiser", ""),
                                 "category": r["category"], "color": self.colors.get(r["category"], "#666")}
            return res

        async with async_playwright() as p:
            browser = await launch_browser(p, headless=True)
            page = await browser.new_page(user_agent=USER_AGENT, viewport={"width": self.WIDTH, "height": 900})
            await page.goto(BASE_URL + section, wait_until="domcontentloaded", timeout=60000)
            for _ in range(self.SCROLLS):  # scroll like a visitor so lazy ad widgets load
                await label_new_ads(page, categorize, self.colors, self.store)
                await page.mouse.wheel(0, 900)
                await page.wait_for_timeout(1300)
            await label_new_ads(page, categorize, self.colors, self.store)
            boxes = await page.evaluate(POSITIONS_JS)
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(600)
            full_h = await page.evaluate("document.documentElement.scrollHeight")
            height = int(min(self.MAX_HEIGHT, full_h, max([b["y"] + b["h"] for b in boxes] + [2000]) + 300))
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / f"{key}.jpg"
            await page.screenshot(path=str(path), type="jpeg", quality=72, full_page=True,
                                  clip={"x": 0, "y": 0, "width": self.WIDTH, "height": height})
            await browser.close()

        placed = [{**b, **ads[b["id"]]} for b in boxes if b["id"] in ads and b["y"] + b["h"] <= height]
        # One list entry per distinct ad, in page order, with every place it appears.
        listing: dict[str, dict] = {}
        for b in sorted(placed, key=lambda b: (b["y"], b["x"])):
            e = listing.setdefault(b["id"], {k: b[k] for k in ("id", "title", "advertiser", "category", "color")}
                                   | {"spots": []})
            e["spots"].append({k: b[k] for k in ("x", "y", "w", "h")})
        self.log(f"[snapshot] {section}: {len(placed)} ad placements, {len(listing)} distinct ads")
        return {"section": section, "key": key, "taken_ts": time.time(),
                "taken_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
                "width": self.WIDTH, "height": height, "ads": list(listing.values()),
                "placements": len(placed), "image": f"{key}.jpg", "version": int(time.time())}
