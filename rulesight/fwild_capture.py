"""F-Wild capture (design §13.1): 50 real website front pages × 3 viewports (1440 / 834 / 390 CSS px) with Playwright.

    python fwild_capture.py --sites sites.txt --out data/fwild [--n 50]
sites.txt: one URL per line (Tranco top 2,000, 50 picked by category). Appends the screenshots to a manifest fragment
fwild.json that fetch_data.py merges (dataset "f_wild", label null; the defects RuleSight reports are confirmed by two people).
"""
import argparse, json, time
from pathlib import Path

VIEWPORTS = {"desktop": (1440, 900, "desktop"), "tablet": (834, 1112, "tablet"), "phone": (390, 844, "phone")}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sites", required=True); ap.add_argument("--out", default="data/fwild"); ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import sync_playwright
    urls = [u.strip() for u in Path(a.sites).read_text(encoding="utf-8").splitlines() if u.strip() and not u.startswith("#")][: a.n]
    screens = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        for i, url in enumerate(urls):
            for key, (w, h, dev) in VIEWPORTS.items():
                ctx = b.new_context(viewport={"width": w, "height": h}, device_scale_factor=1, locale="en-US"); pg = ctx.new_page()
                try:
                    pg.goto(url, wait_until="domcontentloaded", timeout=45000); pg.wait_for_timeout(2500)
                    f = out / f"fw{i:03d}_{key}.png"; pg.screenshot(path=str(f), full_page=False)
                    screens.append({"sid": f"FW-{i:03d}-{key}", "dataset": "f_wild", "split": key, "file": str(f), "label": None, "types": [], "boxes": [], "device_key": dev, "url": url, "captured": time.strftime("%Y-%m-%d")})
                except Exception as e:
                    print("skip", url, key, str(e)[:80])
                ctx.close()
        b.close()
    (out / "fwild.json").write_text(json.dumps(screens, ensure_ascii=False, indent=1), encoding="utf-8"); print(len(screens), "screenshots")

if __name__ == "__main__":
    main()
