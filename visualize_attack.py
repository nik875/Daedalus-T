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
    python visualize_attack.py CLEAN_IMAGE ADV_IMAGE [--out comparison.png] [--conf 0.25]
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "yolo26n.pt"
IMG_SIZE = 640


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
    parser.add_argument("adv", help="path to the adversarial image")
    parser.add_argument("--out", default="comparison.png")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="confidence threshold for displayed detections")
    args = parser.parse_args()

    model = YOLO(MODEL_PATH)

    rc = model.predict(args.clean, imgsz=IMG_SIZE, conf=args.conf, verbose=False)[0]
    ra = model.predict(args.adv,   imgsz=IMG_SIZE, conf=args.conf, verbose=False)[0]

    n_clean = len(rc.boxes)
    n_adv = len(ra.boxes)

    # Raw images resized to model input size (BGR for cv2 saving)
    clean_raw = cv2.resize(cv2.imread(args.clean), (IMG_SIZE, IMG_SIZE))
    adv_raw   = cv2.resize(cv2.imread(args.adv),   (IMG_SIZE, IMG_SIZE))

    # Annotated images from ultralytics (BGR)
    clean_ann = cv2.resize(rc.plot(), (IMG_SIZE, IMG_SIZE))
    adv_ann   = cv2.resize(ra.plot(), (IMG_SIZE, IMG_SIZE))

    top = cv2.hconcat([_label(clean_raw, "clean"),
                       _label(adv_raw,   "adversarial")])
    bot = cv2.hconcat([_label(clean_ann, f"clean: {n_clean} detections"),
                       _label(adv_ann,   f"adversarial: {n_adv} detections")])
    grid = cv2.vconcat([top, bot])
    cv2.imwrite(args.out, grid)

    print(f"clean detections       : {n_clean}")
    print(f"adversarial detections : {n_adv}")
    print(f"comparison grid saved  -> {args.out}")


if __name__ == "__main__":
    main()
