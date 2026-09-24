"""
Lokmat Times ad scraper.

Ads on lokmattimes.com are injected by JavaScript (Taboola native widgets and
Google Ad Manager display slots), so a plain HTTP request never sees them.
We drive a headless Chromium via Playwright, scroll each page so lazy widgets
load, then pull out:

  * Taboola "syndicatedItem" cards  -> sponsored native ads (headline,
    description, advertiser, landing URL, thumbnail)
  * Google ad iframes (best effort) -> visible text, landing URL, image alt

Organic Taboola recommendations (links back to lokmattimes.com articles) are
skipped - they are editorial content, not ads.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import parse_qs, urlparse

BASE_URL = "https://www.lokmattimes.com"

# Section pages to crawl. Different sections attract different advertisers,
# which gives a more varied dataset.
DEFAULT_SECTIONS = [
    "/",
    "/national/",
    "/business/",
    "/sports/",
    "/entertainment/",
    "/lifestyle/",
    "/technology/",
    "/chhatrapati-sambhajinagar/",
    "/pune/",
    "/international/",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# JS run inside the page: collect Taboola cards.
TABOOLA_JS = r"""
() => [...document.querySelectorAll('.videoCube.trc_spotlight_item')].map(card => {
  const a = card.querySelector('a[href]');
  const title = card.querySelector('.video-title, .trc_ellipsis, [class*=title]');
  const desc = card.querySelector('.video-description, [class*=description]');
  const brand = card.querySelector('.branding, [class*=branding]');
  const img = card.querySelector('[style*="background-image"], img');
  let image = '';
  if (img) {
    if (img.tagName === 'IMG') image = img.src || '';
    else {
      const m = (img.getAttribute('style') || '').match(/url\(["']?([^"')]+)/);
      if (m) image = m[1];
    }
  }
  return {
    sponsored: card.classList.contains('syndicatedItem'),
    title: title ? title.innerText.trim() : '',
    description: desc ? desc.innerText.trim() : '',
    advertiser: brand ? brand.innerText.trim() : '',
    raw_text: card.innerText.trim(),
    landing_url: a ? a.href : '',
    image_url: image,
  };
})
"""

# JS run inside each Google ad iframe.
GOOGLE_FRAME_JS = r"""
() => {
  if (!document.body) return null;
  const a = [...document.querySelectorAll('a[href]')].map(x => x.href);
  const imgs = [...document.querySelectorAll('img[alt]')].map(x => x.alt).filter(Boolean);
  return { text: document.body.innerText.trim(), links: a.slice(0, 5), alts: imgs.slice(0, 5) };
}
"""


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except ValueError:
        return ""


def _resolve_google_click(url: str) -> str:
    """Google click URLs carry the real destination in ?adurl=..."""
    try:
        qs = parse_qs(urlparse(url).query)
        for key in ("adurl", "url", "dest"):
            if key in qs and qs[key][0].startswith("http"):
                return qs[key][0]
    except ValueError:
        pass
    return url


TRACKING_PARAM = re.compile(r"^(utm_|tbl|taboola|gclid|fbclid|dclid|msclkid|click_?id|campaign|site|platform|"
                            r"thumbnail|title|cpc|pubid|adid|ad_id|creative|placement|network|source|ref$)", re.I)


def strip_tracking(url: str) -> str:
    """Drop ad-network tracking parameters from a landing URL, keep the page itself."""
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
    try:
        parts = urlsplit(url or "")
    except ValueError:
        return url or ""
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not TRACKING_PARAM.match(k)]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


JUNK_TITLES = {"click here", "learn more", "read more", "ad", "ads", "sponsored", "advertisement", "adchoices"}


def is_junk(title: str, landing_url: str) -> bool:
    """Creatives with no usable text (e.g. 'Click Here', or title == bare domain)."""
    t = _clean(title).lower()
    return (not t) or t in JUNK_TITLES or t == _domain(landing_url) or len(t) < 4


def clean_advertiser(name: str) -> str:
    """'Science Supply Companies | Search Ads' -> 'Science Supply Companies'."""
    return re.sub(r"\s*\|\s*(search\s*ads?|sponsored).*$", "", _clean(name), flags=re.I)


def ad_id(title: str, advertiser: str, landing_url: str) -> str:
    """Stable id so the same creative seen on many pages is stored once."""
    key = f"{_clean(title).lower()}|{_clean(advertiser).lower()}|{_domain(landing_url)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _parse_taboola(card: dict, page_url: str, now: str) -> dict | None:
    if not card.get("sponsored"):
        return None
    landing = strip_tracking(card.get("landing_url", ""))
    if _domain(landing).endswith("lokmattimes.com"):
        return None
    title = _clean(card.get("title"))
    desc = _clean(card.get("description"))
    advertiser = _clean(card.get("advertiser"))
    raw_lines = [l.strip() for l in (card.get("raw_text") or "").splitlines() if l.strip()]
    # Fallback when class names change: first line = title, "Brand|Sponsored" line = advertiser.
    if not title and raw_lines:
        title = raw_lines[0]
    if not advertiser:
        for line in raw_lines:
            if "|" in line and "sponsored" in line.lower():
                advertiser = line.split("|")[0].strip()
                break
    if not desc and len(raw_lines) > 1:
        desc = raw_lines[1] if "sponsored" not in raw_lines[1].lower() else ""
    advertiser = clean_advertiser(advertiser)
    if is_junk(title, landing):
        return None
    return {
        "id": ad_id(title, advertiser, landing),
        "title": title,
        "description": desc,
        "advertiser": advertiser,
        "landing_url": landing,
        "domain": _domain(landing),
        "image_url": card.get("image_url", ""),
        "network": "Taboola",
        "source_page": page_url,
        "scraped_at": now,
    }


def _parse_google(data: dict, page_url: str, now: str) -> dict | None:
    if not data:
        return None
    links = [_resolve_google_click(l) for l in data.get("links", [])]
    links = [l for l in links if l.startswith("http") and "google" not in _domain(l)]
    text_lines = [l.strip() for l in (data.get("text") or "").splitlines() if l.strip()]
    text_lines = [l for l in text_lines if l.lower() not in {"ad", "ads by google", "why this ad?", "x", "i"}]
    alts = data.get("alts", [])
    if not links and not text_lines:
        return None
    landing = strip_tracking(links[0]) if links else ""
    title = _clean(text_lines[0] if text_lines else (alts[0] if alts else ""))
    if is_junk(title, landing):
        return None
    desc = _clean(" ".join(text_lines[1:4]) or " ".join(alts))
    return {
        "id": ad_id(title, "", landing),
        "title": title,
        "description": desc,
        "advertiser": _domain(landing),
        "landing_url": landing,
        "domain": _domain(landing),
        "image_url": "",
        "network": "Google Ads",
        "source_page": page_url,
        "scraped_at": now,
    }


class ChromeNotAvailable(RuntimeError):
    pass


async def launch_browser(p, headless: bool = True, args: list[str] | None = None):
    """Start Google Chrome (the only browser this app uses). Chrome is signed by Google, so Windows
    Smart App Control allows it, unlike the unsigned browser Playwright can download."""
    try:
        return await p.chromium.launch(channel="chrome", headless=headless, args=args or [])
    except Exception as exc:
        raise ChromeNotAvailable(
            "Google Chrome is needed for the Live Site and ad collection but could not be started. "
            "Install it from https://www.google.com/chrome/ and try again. Details: " + str(exc).splitlines()[0]) from exc


async def _scrape_page(context, url: str, scrolls: int, log: Callable[[str], None]) -> list[dict]:
    page = await context.new_page()
    ads: list[dict] = []
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        for _ in range(scrolls):
            await page.mouse.wheel(0, 1200)
            await page.wait_for_timeout(600)
        await page.wait_for_timeout(3000)  # let the last widgets render
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        cards = await page.evaluate(TABOOLA_JS)
        for c in cards:
            ad = _parse_taboola(c, url, now)
            if ad:
                ads.append(ad)

        for frame in page.frames:
            if frame == page.main_frame:
                continue
            furl = frame.url or ""
            if not any(k in furl for k in ("googlesyndication", "safeframe", "doubleclick.net/pagead", "google_ads_iframe")) \
                    and not (frame.name or "").startswith("google_ads_iframe"):
                continue
            try:
                data = await frame.evaluate(GOOGLE_FRAME_JS)
            except Exception:  # cross-origin frame detached / navigated
                continue
            ad = _parse_google(data, url, now)
            if ad:
                ads.append(ad)
        log(f"  {url}: {len(ads)} ad impressions")
    except Exception as exc:  # one bad page should not kill the whole run
        log(f"  {url}: FAILED ({type(exc).__name__}: {exc})")
    finally:
        await page.close()
    return ads


async def scrape_async(sections=None, scrolls: int = 20, headless: bool = True,
                       log: Callable[[str], None] = print) -> list[dict]:
    from playwright.async_api import async_playwright

    sections = sections or DEFAULT_SECTIONS
    urls = [s if s.startswith("http") else BASE_URL + s for s in sections]
    seen: dict[str, dict] = {}
    async with async_playwright() as p:
        browser = await launch_browser(p, headless)
        context = await browser.new_context(user_agent=USER_AGENT, viewport={"width": 1366, "height": 900})
        for url in urls:
            log(f"Scraping {url} ...")
            for ad in await _scrape_page(context, url, scrolls, log):
                seen.setdefault(ad["id"], ad)
        await browser.close()
    log(f"Done. {len(seen)} unique ads.")
    return list(seen.values())


def scrape(sections=None, scrolls: int = 20, headless: bool = True,
           log: Callable[[str], None] = print) -> list[dict]:
    """Synchronous wrapper used by the CLI and the Flask background job."""
    return asyncio.run(scrape_async(sections, scrolls, headless, log))


if __name__ == "__main__":
    import json
    import sys
    result = scrape(sys.argv[1:] or None)
    print(json.dumps(result[:5], indent=2, ensure_ascii=False))
