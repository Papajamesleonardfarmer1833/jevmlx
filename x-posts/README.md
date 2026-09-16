# X posts (copy-paste ready)

Plain `.txt` files. Copy the text as-is and paste into X.

| File | What it is |
|---|---|
| `01_main_post.txt` | single short announcement post |
| `02_thread.txt` | full 7-post thread (benchmarks + diagram + honest caveats) |
| `03_space_launch.txt` | post for launching the Hugging Face Space demo |
| `04_alt_short.txt` | one-paragraph alternative if you don't like threads |
| `post_with_playwright.py` | optional best-effort automation (needs Playwright + a logged-in X session) |

## Before you post

The Space URL is already filled in (`rorshopping/parallel-constrained-decisions`, live now).
The thread's POST 6 contains a `[QUALITY_RESULT ...]` placeholder if the 1.5B/7B/8B accuracy
comparison isn't filled in yet — either fill it or delete that whole post.

## Posting

Recommended: open the .txt, select all, copy, paste into X, post. For the
thread, post the parts in order as replies.

Optional automation (run on the PC where Playwright is installed):

```bash
pip install playwright && playwright install chromium
python post_with_playwright.py 02_thread.txt          # dry run: opens browser, fills posts, does not send
python post_with_playwright.py 02_thread.txt --send   # actually posts
```

The first run creates a `.x-profile/` browser profile — log into X by hand once,
the profile remembers it. Selectors can break when X changes its UI; if the
script can't find a button it will ask you to click it manually and press Enter.
