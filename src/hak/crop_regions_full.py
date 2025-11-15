# crop_regions.py

import os
import sys
import glob
import cv2
import mediapipe as mp
import numpy as np

# ---------------- CONFIG ----------------
DEFAULT_INPUT_DIR = "images_in"
OUTPUT_DIR = "crops_out"
PADDING = 0.06   # padding ratio for each region bbox
# ----------------------------------------


# Landmark index sets (MediaPipe FaceMesh indices)
LEFT_CHEEK_IDXS = [234, 93, 137, 132, 58, 172, 136, 150, 149, 176]
RIGHT_CHEEK_IDXS = [454, 323, 366, 361, 288, 397, 365, 379, 378, 400]
FOREHEAD_IDXS = [10, 67, 103, 109, 66, 69, 297, 334, 336, 296, 293]


def get_bbox(pts_xy, indices, w, h, padding_ratio):
    """Return padded bbox (x1, y1, x2, y2) for given landmark indices."""
    region = pts_xy[indices]
    x_min = int(np.min(region[:, 0]))
    x_max = int(np.max(region[:, 0]))
    y_min = int(np.min(region[:, 1]))
    y_max = int(np.max(region[:, 1]))

    dx = int((x_max - x_min) * padding_ratio)
    dy = int((y_max - y_min) * padding_ratio)

    x_min = max(0, x_min - dx)
    x_max = min(w - 1, x_max + dx)
    y_min = max(0, y_min - dy)
    y_max = min(h - 1, y_max + dy)

    return x_min, y_min, x_max, y_max


def process_single_image(path, face_mesh):
    """Detect landmarks and save forehead / cheeks crops for one image."""
    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] Cannot read image: {path}")
        return

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        print(f"[WARN] No face landmarks detected in: {path}")
        return

    lms = res.multi_face_landmarks[0].landmark
    pts = np.array([(int(l.x * w), int(l.y * h)) for l in lms], dtype=np.int32)

    # Forehead
    fx1, fy1, fx2, fy2 = get_bbox(pts, FOREHEAD_IDXS, w, h, PADDING)
    forehead = img[fy1:fy2, fx1:fx2]

    # Left cheek (subject's left)
    lx1, ly1, lx2, ly2 = get_bbox(pts, LEFT_CHEEK_IDXS, w, h, PADDING)
    left_cheek = img[ly1:ly2, lx1:lx2]

    # Right cheek (subject's right)
    rx1, ry1, rx2, ry2 = get_bbox(pts, RIGHT_CHEEK_IDXS, w, h, PADDING)
    right_cheek = img[ry1:ry2, rx1:rx2]

    base = os.path.splitext(os.path.basename(path))[0]
    out_forehead = os.path.join(OUTPUT_DIR, f"{base}_forehead.png")
    out_left = os.path.join(OUTPUT_DIR, f"{base}_left_cheek.png")
    out_right = os.path.join(OUTPUT_DIR, f"{base}_right_cheek.png")

    cv2.imwrite(out_forehead, forehead)
    cv2.imwrite(out_left, left_cheek)
    cv2.imwrite(out_right, right_cheek)

    print(f"[OK] {path} ->")
    print(f"     {out_forehead}")
    print(f"     {out_left}")
    print(f"     {out_right}")


def collect_input_images(cli_arg_path=None):
    """Return list of image paths to process."""
    if cli_arg_path is not None:
        return [cli_arg_path]

    patterns = ["*.jpg", "*.jpeg", "*.png"]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(DEFAULT_INPUT_DIR, pat)))
    return sorted(paths)


def main():
    # CLI: python crop_regions.py [optional_image_path]
    cli_path = sys.argv[1] if len(sys.argv) > 1 else None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_paths = collect_input_images(cli_path)

    if not img_paths:
        if cli_path is not None:
            print(f"[ERROR] No such file or unsupported extension: {cli_path}")
        else:
            print(f"[ERROR] No images found in directory: {DEFAULT_INPUT_DIR}")
        return

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        for p in img_paths:
            process_single_image(p, face_mesh)


if __name__ == "__main__":
    main()
