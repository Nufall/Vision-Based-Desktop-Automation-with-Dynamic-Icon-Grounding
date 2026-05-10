import argparse
import glob
import os

import cv2
import torch

from groundingdino.util.inference import load_model, load_image

from helpers.detection import detect_app_icons
from helpers.ocr import read_text, text_similarity
from helpers.template_cv import (
    resolve_templates,
    pick_best_template,
    extract_label_crop,
    remove_background,
)
from helpers.visualization import draw_boxes
from helpers.weights import download_file


DEFAULT_WEIGHTS_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha/groundingdino_swint_ogc.pth"
)


class Grounder:
    """Reusable detector. Construct once via load_grounder(); call .detect() per image."""

    def __init__(self, model, device, templates, label_text="match"):
        self.model = model
        self.device = device
        self.templates = templates
        self.label_text = label_text

    def detect(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        *,
        box_threshold: float = 0.1,
        text_threshold: float = 0.1,
        nms_threshold: float = 0.20,
        containment_threshold: float = 0.7,
        max_box_size: int = 150,
        template_threshold: float = 0.45,
        shape_weight: float = 0.85,
        crops_dir: str = "crops",
        debug_dir: str = None,
        top_pad: int = 10,
        bottom_pad: int = 30,
        side_pad: int = 30,
        ocr_target: str = "Notepad",
        label_text: str = None,
    ):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        label_text = label_text if label_text is not None else self.label_text

        image_source, image_tensor = load_image(image_path)
        full_image_bgr = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
        height, width = full_image_bgr.shape[:2]

        print(f"[INFO] Full image size: {width}x{height}")
        print(f"[INFO] Running GroundingDINO with prompt: {prompt}")

        boxes, scores, phrases = detect_app_icons(
            model=self.model,
            image_tensor=image_tensor,
            prompt=prompt,
            device=self.device,
            width=width,
            height=height,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            nms_threshold=nms_threshold,
            containment_threshold=containment_threshold,
            max_box_size=max_box_size,
        )

        print(f"[INFO] Detections after NMS: {len(boxes)}")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if len(boxes) == 0:
            print("[INFO] Detected 0 objects")
            cv2.imwrite(output_path, full_image_bgr)
            print(f"[DONE] Output saved to: {output_path}")
            return {
                "box": None, "score": None, "label": None,
                "saved_to": output_path, "all_boxes": [],
            }

        os.makedirs(crops_dir, exist_ok=True)
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

        print(f"\n[CROP] Extracting with padding top={top_pad} side={side_pad} bottom={bottom_pad}, then removing background...")

        crops = []  # (label_crop, clean_crop, icon_rect)
        for idx, box in enumerate(boxes):
            label_crop, icon_rect = extract_label_crop(
                full_image_bgr, box,
                top_pad=top_pad, bottom_pad=bottom_pad, side_pad=side_pad,
            )
            clean_crop = remove_background(label_crop, icon_rect) if label_crop is not None else None

            if label_crop is not None:
                cv2.imwrite(os.path.join(crops_dir, f"crop_{idx:02d}.png"), label_crop)
            if clean_crop is not None:
                cv2.imwrite(os.path.join(crops_dir, f"crop_{idx:02d}_clean.png"), clean_crop)

            crops.append((label_crop, clean_crop, icon_rect))

        # Template matching first — OCR is now lazy, only used as tiebreaker.
        template_scores = []
        template_labels_used = []
        for idx, (label_crop, _, icon_rect) in enumerate(crops):
            rx, ry, rw, rh = icon_rect
            icon_patch = label_crop[ry:ry + rh, rx:rx + rw] if label_crop is not None else None

            chosen, score = pick_best_template(icon_patch, self.templates, shape_weight=shape_weight)
            label_used = chosen[3] if chosen is not None else "none"
            template_scores.append(score)
            template_labels_used.append(label_used)

            print(f"[INFO] Detection {idx:02d}: [{label_used}] template={score:.4f}")

            if debug_dir and icon_patch is not None:
                cv2.imwrite(
                    os.path.join(debug_dir, f"patch_{idx:02d}_{label_used}_score{score:.3f}.png"),
                    icon_patch,
                )

        passing = [i for i, s in enumerate(template_scores) if s >= template_threshold]
        ocr_texts = {}  # idx -> text, populated lazily

        if not passing:
            best_idx = max(range(len(template_scores)), key=lambda i: template_scores[i])
            best_score = template_scores[best_idx]
            best_label = template_labels_used[best_idx]
            print(
                f"\n[WARN] No detection passed template_threshold={template_threshold:.2f}. "
                f"Best was detection {best_idx} (score={best_score:.4f}). "
                f"Returning no match."
            )
            annotated = draw_boxes(
                full_image_bgr.copy(),
                boxes, scores, phrases,
                best_idx=None, best_score=best_score, label_text=label_text,
            )
            base, ext = os.path.splitext(output_path)
            final_output = f"{base}-nomatch{ext}"
            cv2.imwrite(final_output, annotated)
            print(f"[DONE] No template passed threshold. Output saved to: {final_output}")
            return {
                "box": None,
                "score": float(best_score),
                "label": None,
                "saved_to": final_output,
                "all_boxes": [tuple(int(v) for v in b) for b in boxes],
            }
        elif len(passing) == 1:
            best_idx = passing[0]
            best_score = template_scores[best_idx]
            best_label = template_labels_used[best_idx]
            best_idx_to_draw = best_idx
            print(
                f"\n[INFO] One detection passed threshold: detection {best_idx} "
                f"(score={best_score:.4f}) — OCR skipped"
            )
        else:
            # Multiple detections pass — OCR only the passing crops to break the tie
            print(f"\n[OCR] Running on {len(passing)} passing detection(s) for tie-break...")
            for i in passing:
                crop = crops[i][0]
                ocr_texts[i] = read_text(crop) if crop is not None else ""
                print(f"  Detection {i:02d}: '{ocr_texts[i]}'")

            best_idx = max(passing, key=lambda i: text_similarity(ocr_texts[i], ocr_target))
            best_score = template_scores[best_idx]
            best_label = template_labels_used[best_idx]
            best_idx_to_draw = best_idx
            sims = {i: text_similarity(ocr_texts[i], ocr_target) for i in passing}
            print(
                f"[INFO] OCR similarities to {ocr_target!r}: "
                + ", ".join(f"det{i}={sims[i]:.3f}" for i in passing)
            )
            print(
                f"[INFO] Selected detection {best_idx} (OCR='{ocr_texts[best_idx]}', "
                f"sim={sims[best_idx]:.3f}, template={best_score:.4f})"
            )

        ocr_for_best = ocr_texts.get(best_idx, "")
        print(
            f"[INFO] Best match: detection {best_idx}  template [{best_label}]  "
            f"score={best_score:.4f}  OCR='{ocr_for_best}'"
        )

        annotated = draw_boxes(
            full_image_bgr.copy(),
            boxes,
            scores,
            phrases,
            best_idx=best_idx_to_draw,
            best_score=best_score,
            label_text=label_text,
        )

        base, ext = os.path.splitext(output_path)
        final_output = f"{base}-{best_label}{ext}"
        cv2.imwrite(final_output, annotated)
        print(f"[DONE] Template used: [{best_label}]  Output saved to: {final_output}")

        x1, y1, x2, y2 = [int(v) for v in boxes[best_idx]]
        return {
            "box": (x1, y1, x2, y2),
            "score": float(best_score),
            "label": best_label,
            "saved_to": final_output,
            "all_boxes": [tuple(int(v) for v in b) for b in boxes],
        }


def load_grounder(
    config_path: str,
    weights_path: str,
    template_paths,
    weights_url: str = DEFAULT_WEIGHTS_URL,
    label_text: str = "match",
) -> Grounder:
    """One-time setup: download weights, build model, move to device, resolve templates."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    download_file(weights_url, weights_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model = load_model(config_path, weights_path)
    model = model.to(device)

    templates = resolve_templates(template_paths)

    return Grounder(model=model, device=device, templates=templates, label_text=label_text)


def default_template_paths():
    templates_dir = os.path.join(os.path.dirname(__file__), "notepad_templates")
    found = sorted(glob.glob(os.path.join(templates_dir, "*.png")))
    return found if found else [os.path.join(templates_dir, "notepad.png")]


_default_template_paths = default_template_paths  # backwards-compat alias


def main():
    parser = argparse.ArgumentParser(
        description="Detect icons with GroundingDINO and pick the best template match."
    )

    parser.add_argument("--image", required=True, help="Path to input image")

    parser.add_argument(
        "--template",
        nargs="+",
        default=None,
        help="One or more template image paths (PNG with alpha recommended). "
             "If omitted, auto-discovers all *.png files in the templates/ directory.",
    )

    parser.add_argument("--prompt", default="app icon", help='Detection prompt, e.g. "app icon"')

    parser.add_argument(
        "--label",
        default=None,
        help="Label drawn on the matched box. Defaults to the first template's basename.",
    )

    parser.add_argument(
        "--config",
        default="groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="Path to GroundingDINO config file",
    )

    parser.add_argument(
        "--weights",
        default="weights/groundingdino_swint_ogc.pth",
        help="Path where weights should be stored",
    )

    parser.add_argument(
        "--weights-url",
        default=DEFAULT_WEIGHTS_URL,
        help="URL to download GroundingDINO weights from",
    )

    parser.add_argument(
        "--output",
        default="output/groundingdino_result.jpg",
        help="Path to save annotated output image",
    )

    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.1,
        help="GroundingDINO box confidence threshold",
    )

    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.1,
        help="GroundingDINO text confidence threshold",
    )

    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.20,
        help="NMS IoU threshold. Lower values remove more overlapping boxes.",
    )

    parser.add_argument(
        "--containment-threshold",
        type=float,
        default=0.7,
        help="IoMin threshold for containment NMS — removes a box when this "
             "fraction of its area overlaps a higher-scoring box. Catches "
             "boxes fully inside another that standard IoU NMS misses.",
    )

    parser.add_argument(
        "--max-box-size",
        type=int,
        default=150,
        help="Maximum allowed width or height of a detection box in pixels. "
             "Boxes larger than this are skipped before NMS.",
    )

    parser.add_argument(
        "--template-threshold",
        type=float,
        default=0.45,
        help="Minimum combined template score required to mark a match",
    )

    parser.add_argument(
        "--shape-weight",
        type=float,
        default=0.85,
        help="Weight for the shape score vs. the color score (0..1). Color is "
             "easily fooled by other blue+white desktop icons, so it should "
             "act as a small tiebreaker, not half the score.",
    )

    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Optional directory to save per-detection icon patches for debugging",
    )

    parser.add_argument(
        "--crops-dir",
        default="crops",
        help="Directory to save per-detection label crops (box + bottom padding). Default: crops/",
    )

    parser.add_argument(
        "--top-pad",
        type=int,
        default=10,
        help="Extra pixels added above each detection box for crops and template matching. Default: 10",
    )

    parser.add_argument(
        "--bottom-pad",
        type=int,
        default=30,
        help="Extra pixels added below each detection box for crops and template matching. Default: 30",
    )

    parser.add_argument(
        "--side-pad",
        type=int,
        default=30,
        help="Extra pixels added left and right of each detection box for crops and template matching. Default: 30",
    )

    parser.add_argument(
        "--ocr-target",
        default="Notepad",
        help="Text to look for via OCR when breaking ties among multiple matches. Default: Notepad",
    )

    args = parser.parse_args()

    template_paths = args.template if args.template else _default_template_paths()
    label_text = args.label or os.path.splitext(os.path.basename(template_paths[0]))[0]

    grounder = load_grounder(
        config_path=args.config,
        weights_path=args.weights,
        template_paths=template_paths,
        weights_url=args.weights_url,
        label_text=label_text,
    )

    grounder.detect(
        image_path=args.image,
        prompt=args.prompt,
        output_path=args.output,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_threshold=args.nms_threshold,
        containment_threshold=args.containment_threshold,
        max_box_size=args.max_box_size,
        template_threshold=args.template_threshold,
        shape_weight=args.shape_weight,
        debug_dir=args.debug_dir,
        crops_dir=args.crops_dir,
        top_pad=args.top_pad,
        bottom_pad=args.bottom_pad,
        side_pad=args.side_pad,
        ocr_target=args.ocr_target,
    )


if __name__ == "__main__":
    main()
