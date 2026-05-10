# Vision-Based Desktop Automation with Dynamic Icon Grounding

Locate the Notepad icon anywhere on a Windows desktop, double-click it, and use it to save 10 blog posts pulled from the JSONPlaceholder API. The grounding step is **vision-only** — no Win32 enumeration of desktop icons, no hardcoded coordinates — so the same pipeline works regardless of where the icon is on the desktop.

## Approach

The grounding pipeline has three stages, run on a single screenshot:

1. **Open-vocabulary detection (GroundingDINO).** The text prompt `"application icon."` produces a set of candidate boxes. Open-vocab means we don't need a labelled training set for "Notepad icon" — the same prompt works for any icon-shaped element.
2. **Template matching across detected crops.** Each candidate is cropped (with padding for the label text below) and scored against vendored Notepad icon templates using a shape + color score. This filters out non-Notepad icons that GroundingDINO also picked up (Recycle Bin, This PC, etc.).
3. **OCR tiebreak.** When more than one detection passes the template threshold, EasyOCR reads the label below each candidate and picks the one whose text is closest to `"Notepad"`. OCR is skipped entirely when only one detection passes — usually the case — so the model loads only when it would change the answer.

If no detection passes the template threshold, grounding returns "no match" rather than the highest-scoring fallback. The retry loop in `main.py` then restarts from `Win+D`. After 3 failed attempts on the same post, it raises.


## Setup

Requires:
- Windows 10/11 at 1920×1080
- An NVIDIA GPU with a driver supporting **CUDA 11.3** (driver ≥ 465.19). The pinned `torch==1.10.1+cu113` wheel needs this.
- A Notepad shortcut on the desktop *before* running.
- [`uv`](https://docs.astral.sh/uv/) installed (`irm https://astral.sh/uv/install.ps1 | iex`).

```powershell
git clone <this-repo>
cd inference
uv sync
.\run.bat
```

`uv sync` creates `.venv\`, installs everything from `uv.lock` (~3 GB the first time including the CUDA torch build), and writes the project venv. `run.bat` wraps `uv run python main.py`. On the first launch:

- GroundingDINO weights (~350 MB) auto-download into `weights/` via `helpers/weights.py`.
- EasyOCR downloads its CRAFT + CRNN models (~150 MB) into `~\.EasyOCR\`.

Subsequent runs skip both downloads.

## Pipeline (per post)

```
Win+D
  -> screenshot desktop (taskbar cropped out)
  -> ground notepad icon (GroundingDINO + template + lazy OCR)
  -> if no confident match: retry up to 3x with 1s delay, else fail this post
  -> double-click icon center
  -> validate Notepad launched (Win32 window-title check via pygetwindow)
  -> click into text area, paste "Title: {title}\n\n{body}"
  -> Ctrl+S, paste full save path, Enter
  -> screenshot save dialog
  -> if "Confirm Save As" duplicate dialog detected (template match): Left, Enter
  -> Alt+F4 to close Notepad
```

10 posts → 10 independent cycles.

## Project layout

```
inference/
├── main.py                # orchestration
├── grounding.py           # Grounder class + load_grounder() (one-time setup, then per-image detect)
├── validate_notepad.py    # template-matching validator (used for the duplicate dialog)
├── fetch_posts.py         # JSONPlaceholder client + cache
├── helpers/
│   ├── detection.py       # GroundingDINO inference + NMS + size filter
│   ├── template_cv.py     # template matching (shape + color)
│   ├── ocr.py             # EasyOCR wrapper
│   ├── visualization.py   # box drawing
│   ├── boxes.py           # cxcywh<->xyxy conversions
│   └── weights.py         # weights download with progress
├── groundingdino/         # vendored upstream package (IDEA-Research/GroundingDINO)
├── notepad_templates/     # template images for icon matching
├── templates/             # template images for the duplicate-save dialog
├── pyproject.toml         # uv config
├── .python-version        # 3.9
└── run.bat                # uv run python main.py
```

## Robustness behavior

| Failure mode | Handling |
|---|---|
| Icon not found on desktop | `with_retry` retries grounding 3× with 1s delay; then the outer per-post loop restarts from `Win+D`. After 3 outer restarts the post fails. |
| No detection passes template threshold | `grounding.py` returns `box=None`; `main.py` raises `RuntimeError`; retry loop fires. |
| Notepad window doesn't appear | `find_notepad_window` polls window titles for `APP_LAUNCH_DELAY * 2` seconds, then retries from `Win+D`. |
| API unreachable | `fetch_posts.py` falls back to `~/Desktop/tjm-project/.posts_cache.json` from the last successful fetch. |
| Save target file already exists | "Confirm Save As" dialog detected by template match → press `Left, Enter` to confirm overwrite. |
| Multiple matching icons (Notepad among Notepad++, etc.) | Template matching scores each crop; if multiple pass, OCR similarity to "Notepad" tiebreaks. |

## Performance notes

- The grounder is loaded **once** at startup and reused across all 10 posts. Each post only pays per-image cost (~1–2 s on GPU): GroundingDINO inference + crop extraction + template matching. OCR runs only on tiebreaks.
- Cold-start (model load + CUDA init + EasyOCR load) is ~5–9 s, paid once per run.
