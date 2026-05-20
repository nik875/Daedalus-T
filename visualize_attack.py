"""
Visualise a Daedalus attack: clean vs adversarial image, and YOLO26's
detections on each.

Runs *standard* YOLO26 inference (the default NMS-free one2one path with
postprocessing) so the result reflects real deployment behaviour.  Kept
separate from daedalus_yolo26.py, which monkey-patches Detect.forward.

Output: a 2x2 grid PNG —
    top row    : clean image            | adversarial image
    bottom row : clean + detections     | adversarial + detections

Usage:
    # Two-image mode (single-image attack output):
    python visualize_attack.py CLEAN_IMAGE ADV_IMAGE [--out comparison.png] [--conf 0.25]

    # Universal mode: build the adversarial image from a clean image + saved delta:
    python visualize_attack.py CLEAN_IMAGE --delta adv_examples/yolo26_universal/delta_final.npy

    # Test transfer to a bigger model than the one attacked:
    python visualize_attack.py CLEAN_IMAGE --delta delta_final.npy --model yolo26x.pt

    # Evaluate against a native MLX (Apple Silicon) model — pass an .npz weight:
    python visualize_attack.py CLEAN_IMAGE --delta delta_final.npy --model yolo26n.npz

    # Or force the MLX backend on an existing .pt (auto-converted on first load):
    python visualize_attack.py CLEAN_IMAGE --delta delta_final.npy --model yolo26n.pt --mlx
"""

import argparse
import os
import cv2
import numpy as np
from ultralytics.utils.plotting import Colors

MODEL_PATH = "yolo26n.pt"
IMG_SIZE = 640

# ultralytics' 20-colour palette, so both backends draw class colours identically.
_COLORS = Colors()

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _load_model(path, force_mlx=False):
    """
    Load YOLO weights, picking the backend by file extension: an .npz/.safetensors
    weight uses the native MLX implementation (yolo-mlx), anything else uses
    ultralytics.  force_mlx routes any weight (including a .pt) through yolo-mlx,
    which converts a .pt to MLX in-process on first load.  Returns (model, is_mlx).
    is_mlx matters because the two backends' plot() differ in channel order
    (see main()).
    """
    if force_mlx or path.lower().endswith((".npz", ".safetensors")):
        from yolo26mlx import YOLO
        # yolo-mlx's own .pt loader assumes the `safetensors` pkg is installed;
        # without it the converter falls back to .npz but the loader still looks
        # for .safetensors and crashes.  Convert to a sibling .npz ourselves and
        # load that (reused on subsequent runs).
        if path.lower().endswith(".pt"):
            npz_path = os.path.splitext(path)[0] + ".npz"
            if not os.path.exists(npz_path):
                from yolo26mlx.converters.convert import convert_yolo26_weights
                convert_yolo26_weights(path, npz_path, verbose=False)
            path = npz_path
        return YOLO(path), True
    from ultralytics import YOLO
    return YOLO(path), False


def _imread(path):
    """cv2.imread that raises a clear error instead of returning None."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def _resolve_name(names, cls_id):
    """Class name from a model's names map, falling back to COCO when the
    model only carries generic placeholders (yolo-mlx stores 'class0', ...)."""
    cand = None
    if isinstance(names, dict):
        cand = names.get(cls_id, names.get(str(cls_id)))
    elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        cand = names[cls_id]
    if cand is not None:
        s = str(cand).strip().lower()
        if not (s.startswith("class") or s.startswith("cls")):
            return str(cand)
    return COCO80[cls_id] if 0 <= cls_id < len(COCO80) else f"class{cls_id}"


def _annotate(result, is_mlx):
    """Draw per-class coloured boxes + labels on result.orig_img (BGR out).

    Replaces each backend's plot(): ultralytics' is fine but yolo-mlx's paints
    every box red.  Drawing here gives identical, class-coloured output for both.
    Boxes are in orig_img pixel space, so we annotate orig_img then let the
    caller resize.
    """
    base = result.orig_img
    img = cv2.cvtColor(base, cv2.COLOR_RGB2BGR) if is_mlx else base.copy()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return img

    lw = max(round(sum(img.shape[:2]) / 2 * 0.003), 2)   # line width ~ ultralytics
    font, fs, tf = cv2.FONT_HERSHEY_SIMPLEX, lw / 3.0, max(lw - 1, 1)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i])
        cls_id = int(boxes.cls[i])
        color = _COLORS(cls_id, bgr=True)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, lw, cv2.LINE_AA)

        label = f"{_resolve_name(result.names, cls_id)} {float(boxes.conf[i]):.2f}"
        (tw, th), _ = cv2.getTextSize(label, font, fs, tf)
        outside = y1 - th - 3 >= 0
        cv2.rectangle(img, (x1, y1 - th - 3 if outside else y1),
                      (x1 + tw, y1 if outside else y1 + th + 3), color, -1, cv2.LINE_AA)
        txt_color = (255, 255, 255) if sum(color) < 384 else (0, 0, 0)
        cv2.putText(img, label, (x1, y1 - 2 if outside else y1 + th + 2),
                    font, fs, txt_color, tf, cv2.LINE_AA)
    return img


def _build_adv_from_delta(clean_path, delta_path, out_dir="."):
    """
    Apply a saved universal delta to a clean image, reproducing the training
    preprocessing (square resize to IMG_SIZE, RGB, [0,1]).  Writes square 640
    clean + adversarial PNGs so YOLO's letterbox is a no-op and both images are
    compared on identical pixels.  Returns (clean_png_path, adv_png_path).
    """
    delta = np.load(delta_path)                       # (H, W, 3) RGB, signed
    clean_bgr = cv2.resize(_imread(clean_path), (IMG_SIZE, IMG_SIZE),
                           interpolation=cv2.INTER_CUBIC)
    clean_rgb01 = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    adv_rgb01 = np.clip(clean_rgb01 + delta, 0.0, 1.0)
    adv_bgr = cv2.cvtColor((adv_rgb01 * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    clean_png = os.path.join(out_dir, "_clean_640.png")
    adv_png = os.path.join(out_dir, "_adv_640.png")
    cv2.imwrite(clean_png, clean_bgr)
    cv2.imwrite(adv_png, adv_bgr)
    return clean_png, adv_png


def _label(img, text):
    """Draw a text banner at the top-left of an image (in-place copy)."""
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clean", help="path to the clean image")
    parser.add_argument("adv", nargs="?", default=None,
                        help="path to the adversarial image (omit when using --delta)")
    parser.add_argument("--delta", default=None,
                        help="universal delta .npy; builds adv = clip(clean + delta, 0, 1)")
    parser.add_argument("--model", default=MODEL_PATH,
                        help="YOLO weights to evaluate against, e.g. yolo26n.pt, "
                             "yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt "
                             "(test transfer to bigger models than the one attacked). "
                             "An .npz weight uses the native MLX backend (yolo-mlx).")
    parser.add_argument("--mlx", action="store_true",
                        help="force the native MLX backend (yolo-mlx) even for a "
                             ".pt weight; the .pt is converted to MLX on first load")
    parser.add_argument("--out", default="comparison.png")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="confidence threshold for displayed detections")
    args = parser.parse_args()

    if args.delta:
        args.clean, args.adv = _build_adv_from_delta(
            args.clean, args.delta, out_dir=os.path.dirname(args.out) or ".")
    elif args.adv is None:
        parser.error("provide an ADV image path or --delta")

    model, is_mlx = _load_model(args.model, force_mlx=args.mlx)

    # yolo-mlx's predict() has no verbose kwarg; only ultralytics does.
    predict_kwargs = {"imgsz": IMG_SIZE, "conf": args.conf}
    if not is_mlx:
        predict_kwargs["verbose"] = False
    rc = model.predict(args.clean, **predict_kwargs)[0]
    ra = model.predict(args.adv,   **predict_kwargs)[0]

    n_clean = len(rc.boxes)
    n_adv = len(ra.boxes)

    # Raw images resized to model input size (BGR for cv2 saving)
    clean_raw = cv2.resize(_imread(args.clean), (IMG_SIZE, IMG_SIZE))
    adv_raw   = cv2.resize(_imread(args.adv),   (IMG_SIZE, IMG_SIZE))

    # Class-coloured annotations drawn here (not via backend plot()) so MLX and
    # ultralytics look identical; yolo-mlx's own plot() paints every box red.
    clean_ann = cv2.resize(_annotate(rc, is_mlx), (IMG_SIZE, IMG_SIZE))
    adv_ann   = cv2.resize(_annotate(ra, is_mlx), (IMG_SIZE, IMG_SIZE))

    top = cv2.hconcat([_label(clean_raw, "clean"),
                       _label(adv_raw,   "adversarial")])
    bot = cv2.hconcat([_label(clean_ann, f"clean: {n_clean} detections"),
                       _label(adv_ann,   f"adversarial: {n_adv} detections")])
    grid = cv2.vconcat([top, bot])
    cv2.imwrite(args.out, grid)

    print(f"model                  : {args.model}")
    print(f"clean detections       : {n_clean}")
    print(f"adversarial detections : {n_adv}")
    print(f"comparison grid saved  -> {args.out}")


if __name__ == "__main__":
    main()
