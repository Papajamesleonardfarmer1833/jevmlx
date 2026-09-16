#!/usr/bin/env python3
"""Best-effort helper to post to X with Playwright.

This is a convenience, not a guarantee: X changes its DOM regularly and
automation may break. The reliable path is copy/pasting from the .txt files
in this folder.

Setup (once, on a machine that has Playwright - e.g. the PC):
    pip install playwright
    playwright install chromium

Usage:
    python post_with_playwright.py 02_thread.txt          # dry run (fills the composer, does not post)
    python post_with_playwright.py 02_thread.txt --send   # actually post

The script opens a browser with a persistent profile (./.x-profile) so you
only log in once. It waits for you to log in by hand the first time.
Posts/files are separated by lines containing only --- ; posts that still
contain a [[PLACEHOLDER]] are skipped automatically.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path


def load_posts(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^---\s*$", text, flags=re.M)
    posts: list[str] = []
    for part in parts:
        lines = part.splitlines()
        while lines and (re.match(r"^POST \d+\s*$", lines[0].strip()) or re.match(r"^-+$", lines[0].strip())):
            lines.pop(0)
        body = "\n".join(lines).strip()
        if body:
            posts.append(body)
    return posts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path, help="one of the .txt files in this folder")
    ap.add_argument("--send", action="store_true", help="actually click Post (default: dry run)")
    ap.add_argument("--profile", default=".x-profile", help="browser profile directory (default: .x-profile)")
    args = ap.parse_args()

    posts = load_posts(args.file)
    skipped = [p for p in posts if "[[" in p]
    posts = [p for p in posts if "[[" not in p]
    for s in skipped:
        print(f"skipping post with placeholder: {s[:60]!r}...")
    if not posts:
        sys.exit("nothing to post")
    print(f"{len(posts)} post(s) loaded from {args.file}")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("playwright not installed. Run: pip install playwright && playwright install chromium")

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(args.profile, headless=False, viewport={"width": 1280, "height": 900})
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto("https://x.com/compose/post")
        print("Waiting for the composer (log in by hand if asked)...")
        page.wait_for_selector('div[data-testid="tweetTextarea_0"]', timeout=180_000)

        for i, post in enumerate(posts):
            if i > 0:
                try:
                    page.click('button[data-testid="addButton"]', timeout=5000)
                except Exception:
                    input("Could not find the '+ post' button. Click it manually, then press Enter here.")
            sel = f'div[data-testid="tweetTextarea_{i}"]'
            page.wait_for_selector(sel, timeout=20_000)
            page.click(sel)
            page.keyboard.insert_text(post)
            time.sleep(1)
            print(f"filled post {i + 1}/{len(posts)}")

        page.screenshot(path="x-post-preview.png")
        print("Preview screenshot saved: x-post-preview.png")

        if not args.send:
            input("Dry run complete. Review the browser window, then press Enter to close.")
        else:
            input("Press Enter to click Post (Ctrl+C to abort).")
            try:
                page.click('button[data-testid="tweetButton"]', timeout=10_000)
            except Exception:
                page.click('button[data-testid="tweetButtonInline"]', timeout=10_000)
            time.sleep(6)
            print("Clicked Post. Check the browser to confirm.")
        ctx.close()


if __name__ == "__main__":
    main()
