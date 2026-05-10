"""End-to-end automation:
1. Win+D, screenshot, ground notepad icon with grounding.py.
2. Double-click icon, screenshot, validate notepad opened with validate_notepad.py.
3. fetch_posts.py for the 10 posts.
4. For each post: type into notepad, save as Desktop/tjm-project/post_{id}.txt,
   handle the "Confirm Save As" duplicate dialog.

Run via:  uv run python main.py   (or .\\run.bat)
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import pyautogui
import pygetwindow as gw
import pyperclip

INFERENCE_DIR = Path(__file__).resolve().parent
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))
from grounding import load_grounder, default_template_paths


VALIDATE_SCRIPT = INFERENCE_DIR / "validate_notepad.py"
FETCH_POSTS_SCRIPT = INFERENCE_DIR / "fetch_posts.py"

GROUNDER_CONFIG = INFERENCE_DIR / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
GROUNDER_WEIGHTS = INFERENCE_DIR / "weights" / "groundingdino_swint_ogc.pth"
GROUNDER_WEIGHTS_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha/groundingdino_swint_ogc.pth"
)

NOTEPAD_TEMPLATE_ARG = r"templates\notepad_template.png"
DUPLICATE_TEMPLATE_ARG = r"templates\duplicate_template.png"

SCREENSHOT_DIR = INFERENCE_DIR / "screenshots"
RESULT_DIR = INFERENCE_DIR / "output"

STEP_DELAY = 0.5
APP_LAUNCH_DELAY = 2.0
MAX_RETRIES = 3
RETRY_DELAY = 1.0
GROUND_PROMPT = "application icon."
TASKBAR_HEIGHT = 40


def step_sleep():
    time.sleep(STEP_DELAY)


def with_retry(fn, what, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            result = fn()
            if result is None:
                raise RuntimeError(f"{what}: no result")
            return result
        except Exception as exc:
            last_err = exc
            print(f"[RETRY {attempt}/{max_retries}] {what} failed: {exc}")
            if attempt < max_retries:
                time.sleep(retry_delay)
    raise RuntimeError(f"{what} failed after {max_retries} attempts: {last_err}")


def show_desktop():
    pyautogui.hotkey("win", "d")


def find_notepad_window(timeout: float = 5.0):
    """Wait up to `timeout` seconds for any visible Notepad window; return the most recent."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        matches = []
        for w in gw.getAllWindows():
            try:
                title = (w.title or "").strip()
                visible = bool(w.visible)
            except Exception:
                continue
            if not title or not visible:
                continue
            if title == "Notepad" or title.endswith(" - Notepad") or title.endswith("- Notepad"):
                matches.append(w)
        if matches:
            return matches[-1]
        time.sleep(0.2)
    return None


def activate_window(win):
    try:
        if getattr(win, "isMinimized", False):
            win.restore()
        win.activate()
    except Exception as exc:
        print(f"  [WARN] activate_window: {exc}")


def take_screenshot(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    screen_w, screen_h = pyautogui.size()
    pyautogui.screenshot(str(path), region=(0, 0, screen_w, screen_h - TASKBAR_HEIGHT))
    return path


def run_grounding(grounder, image_path: Path, output_path: Path, prompt: str = GROUND_PROMPT):
    """Call grounder.detect on the screenshot and return (cx, cy, (x1,y1,x2,y2))."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = grounder.detect(str(image_path), prompt, str(output_path))
    if result.get("box") is None:
        raise RuntimeError("Grounder returned no detections")
    x1, y1, x2, y2 = result["box"]
    return (x1 + x2) // 2, (y1 + y2) // 2, result["box"]


def run_validate(screenshot_path: Path, template_arg: str):
    """Run validate_notepad.py via CLI and parse [FOUND]/[NOT FOUND] from stdout."""
    cmd = [
        sys.executable, str(VALIDATE_SCRIPT),
        str(screenshot_path), "", template_arg,
    ]
    proc = subprocess.run(
        cmd, cwd=str(INFERENCE_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL, text=True,
    )
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    found = "[FOUND]" in proc.stdout
    if not found:
        return {"found": False}

    top_left = None
    dot = None
    for line in proc.stdout.splitlines():
        if "[FOUND]" in line:
            m_at = re.search(r"at \((\d+),\s*(\d+)\)", line)
            m_dot = re.search(r"dot at \((\d+),\s*(\d+)\)", line)
            if m_at:
                top_left = (int(m_at.group(1)), int(m_at.group(2)))
            if m_dot:
                dot = (int(m_dot.group(1)), int(m_dot.group(2)))

    template_path = INFERENCE_DIR / template_arg
    rect_center = None
    rect_size = None
    tmpl = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED) if template_path.exists() else None
    if tmpl is not None and top_left is not None:
        th, tw = tmpl.shape[:2]
        rect_size = (tw, th)
        rect_center = (top_left[0] + tw // 2, top_left[1] + th // 2)

    return {
        "found": True,
        "top_left": top_left,
        "dot": dot,
        "rect_center": rect_center,
        "rect_size": rect_size,
    }


def run_fetch_posts():
    """Run fetch_posts.py and return the 10 cached posts."""
    cmd = [sys.executable, str(FETCH_POSTS_SCRIPT)]
    proc = subprocess.run(
        cmd, cwd=str(INFERENCE_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL, text=True,
    )
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_posts.py exited {proc.returncode}")

    if str(INFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(INFERENCE_DIR))
    from fetch_posts import get_desktop_path
    desktop = get_desktop_path()
    cache = Path(desktop) / "tjm-project" / ".posts_cache.json"
    if not cache.exists():
        raise FileNotFoundError(f"posts cache not found: {cache}")
    with open(cache, "r", encoding="utf-8") as f:
        posts = json.load(f)
    return posts[:10], Path(desktop) / "tjm-project"


def paste_text(text: str):
    pyperclip.copy(text)
    pyautogui.hotkey("ctrl", "v")


def process_post(grounder, post, post_num, total, save_dir):
    """Run the full detection + open + type + save cycle for a single post."""
    print(f"\n=== Post {post_num}/{total} (id={post['id']}) ===")

    # Steps 1-7 restart from scratch if Notepad validation fails
    notepad_window = None
    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            print(f"  [RESTART {attempt}/{MAX_RETRIES}] Notepad not confirmed — starting over")
            time.sleep(RETRY_DELAY)

        # Step 1: Win+D
        print("  [1] Win+D")
        show_desktop()
        step_sleep()

        # Step 2: screenshot desktop
        print("  [2] Screenshot desktop")
        desktop_shot = take_screenshot(SCREENSHOT_DIR / f"desktop_{post['id']}.png")
        step_sleep()

        # Step 3+4: ground notepad icon
        print("  [3+4] Ground notepad icon")
        try:
            cx, cy, _ = with_retry(
                lambda: run_grounding(grounder, desktop_shot, RESULT_DIR / f"result_{post['id']}.jpg"),
                "Grounding notepad icon",
            )
        except RuntimeError as e:
            print(f"  [WARN] Grounding failed: {e}")
            continue
        print(f"    icon center -> ({cx}, {cy})")
        step_sleep()

        # Step 5: double-click to open Notepad
        print("  [5] Double-click Notepad icon")
        pyautogui.doubleClick(cx, cy)
        time.sleep(APP_LAUNCH_DELAY)

        # Step 6: screenshot (kept for the per-post audit trail)
        print("  [6] Screenshot Notepad")
        take_screenshot(SCREENSHOT_DIR / f"notepad_{post['id']}.png")
        step_sleep()

        # Step 7: validate Notepad launched via Win32 window-title check
        print("  [7] Validate Notepad via window title")
        notepad_window = find_notepad_window(timeout=APP_LAUNCH_DELAY * 2)
        if notepad_window is None:
            print("  [WARN] No window with 'Notepad' in title found")
            continue

        print(
            f"    title -> {notepad_window.title!r}  "
            f"rect=({notepad_window.left},{notepad_window.top},"
            f"{notepad_window.width}x{notepad_window.height})"
        )
        activate_window(notepad_window)
        step_sleep()
        break  # validation passed

    if notepad_window is None:
        raise RuntimeError(f"Post {post['id']}: Notepad could not be confirmed after {MAX_RETRIES} attempts")

    # Step 10: click center of Notepad window so caret lands in the text area
    print("  [10] Click into text area")
    cx_w = notepad_window.left + notepad_window.width // 2
    cy_w = notepad_window.top + notepad_window.height // 2
    pyautogui.leftClick(cx_w, cy_w)
    step_sleep()

    # Step 11: paste post content
    print("  [11] Paste post content")
    content = f"Title: {post['title']}\n\n{post['body']}"
    paste_text(content)
    step_sleep()

    # Step 12: Ctrl+S — opens Save As directly on both classic and Win11 Notepad,
    # bypassing the unsaved-changes prompt that the click-X flow depended on.
    print("  [12] Ctrl+S")
    pyautogui.hotkey("ctrl", "s")
    step_sleep()

    # Step 14: paste full save path with Ctrl+V, then Enter
    print("  [14] Paste save path")
    post_path = str(save_dir / f"post_{post['id']}.txt")
    paste_text(post_path)
    step_sleep()
    pyautogui.press("enter")
    step_sleep()

    # Step 15: screenshot
    print("  [15] Screenshot save dialog")
    save_shot = take_screenshot(SCREENSHOT_DIR / f"save_{post['id']}.png")
    step_sleep()

    # Step 16: check for duplicate ("Confirm Save As") dialog
    print("  [16] Check for duplicate")
    dup = run_validate(save_shot, DUPLICATE_TEMPLATE_ARG)
    step_sleep()

    # Step 17: handle duplicate
    if dup.get("found"):
        print("  [17] Duplicate detected -> Left + Enter")
        pyautogui.press("left")
        step_sleep()
        pyautogui.press("enter")
        step_sleep()
    else:
        print("  [17] No duplicate dialog")

    # Step 18: close Notepad via Alt+F4 (file is saved, no unsaved-changes prompt)
    print("  [18] Close Notepad (Alt+F4)")
    pyautogui.hotkey("alt", "f4")
    step_sleep()


def main():
    pyautogui.FAILSAFE = True
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Build the grounder ONCE — model + EasyOCR + CUDA stay warm for all 10 posts.
    print("[SETUP] Loading grounder (one-time)")
    grounder = load_grounder(
        config_path=str(GROUNDER_CONFIG),
        weights_path=str(GROUNDER_WEIGHTS),
        template_paths=default_template_paths(),
        weights_url=GROUNDER_WEIGHTS_URL,
    )

    # Fetch posts once; the loop does everything else fresh per post
    print("[SETUP] Fetching posts")
    posts, save_dir = run_fetch_posts()
    print(f"  Got {len(posts)} posts; save_dir={save_dir}")
    step_sleep()

    for i, post in enumerate(posts, start=1):
        process_post(grounder, post, i, len(posts), save_dir)

    print("\n[DONE] All posts processed.")


if __name__ == "__main__":
    main()
