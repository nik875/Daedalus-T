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

MODEL_PATH = "yolo26n.pt"
IMG_SIZE = 640


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

    # Annotated images: ultralytics plot() is BGR, yolo-mlx plot() is RGB.
    clean_ann = rc.plot()
    adv_ann = ra.plot()
    if is_mlx:
        clean_ann = cv2.cvtColor(clean_ann, cv2.COLOR_RGB2BGR)
        adv_ann = cv2.cvtColor(adv_ann, cv2.COLOR_RGB2BGR)
    clean_ann = cv2.resize(clean_ann, (IMG_SIZE, IMG_SIZE))
    adv_ann   = cv2.resize(adv_ann,   (IMG_SIZE, IMG_SIZE))

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
