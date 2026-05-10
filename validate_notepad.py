import os
import sys
import cv2
import numpy as np

TEMPLATE_PATH = r'templates\notepad_template'
DOT_COLOR = (0, 0, 255)  # red
DOT_RADIUS = 5
THRESHOLD = 0.8


def find_and_mark(screenshot_path: str, output_path: str = None, threshold: float = THRESHOLD, template_path: str = TEMPLATE_PATH):
    screenshot = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
    if screenshot is None:
        raise ValueError(f"Cannot read screenshot: {screenshot_path}")

    template_raw = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    if template_raw is None:
        raise ValueError(f"Cannot read template: {template_path}")

    if template_raw.ndim == 3 and template_raw.shape[2] == 4:
        mask = template_raw[:, :, 3]
        template_bgr = template_raw[:, :, :3]
    else:
        mask = None
        template_bgr = template_raw

    th, tw = template_bgr.shape[:2]

    gray_ss = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    gray_tmpl = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    if mask is not None:
        result = cv2.matchTemplate(gray_ss, gray_tmpl, cv2.TM_CCOEFF_NORMED, mask=mask)
    else:
        result = cv2.matchTemplate(gray_ss, gray_tmpl, cv2.TM_CCOEFF_NORMED)

    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < threshold:
        print(f"[NOT FOUND] Best score {max_val:.3f} is below threshold {threshold}")
        return False, screenshot

    top_left = max_loc
    # top-right of the match, then 10px left and 10px down
    dot_x = top_left[0] + tw - 10
    dot_y = top_left[1] + 10

    annotated = screenshot.copy()
    cv2.circle(annotated, (dot_x, dot_y), DOT_RADIUS, DOT_COLOR, -1)

    # draw match rectangle for visibility
    bottom_right = (top_left[0] + tw, top_left[1] + th)
    cv2.rectangle(annotated, top_left, bottom_right, (0, 255, 0), 2)

    if not output_path:
        os.makedirs("output", exist_ok=True)
        filename = os.path.basename(screenshot_path)
        base, ext = filename.rsplit(".", 1)
        output_path = os.path.join("output", f"{base}_marked.{ext}")

    cv2.imwrite(output_path, annotated)
    print(f"[FOUND] Score {max_val:.3f} at {top_left}  dot at ({dot_x}, {dot_y})")
    print(f"[SAVED] {output_path}")
    return True, annotated


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_notepad.py <screenshot_path> [output_path] [template_path]")
        sys.exit(1)

    screenshot_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    template_path = sys.argv[3] if len(sys.argv) > 3 else TEMPLATE_PATH
    find_and_mark(screenshot_path, output_path, template_path=template_path)
