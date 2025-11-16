# crop_regions_full.py
import os
import sys
import glob
import cv2
import mediapipe as mp
import numpy as np

DEFAULT_INPUT_DIR = "images_in"
OUTPUT_DIR = "crops_out"
PADDING = 0.06  # bbox padding ratio

# MediaPipe FaceMesh landmark indices
LEFT_CHEEK_IDXS = [234, 93, 137, 132, 58, 172, 136, 150, 149, 176]
RIGHT_CHEEK_IDXS = [454, 323, 366, 361, 288, 397, 365, 379, 378, 400]
FOREHEAD_IDXS = [10, 67, 103, 109, 66, 69, 297, 334, 336, 296, 293]


def get_bbox(pts_xy, indices, w, h, padding_ratio):
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


def process_image(path, face_mesh):
    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] cannot read: {path}")
        return

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        print(f"[WARN] no face landmarks in: {path}")
        return

    lms = res.multi_face_landmarks[0].landmark
    pts = np.array([(int(l.x * w), int(l.y * h)) for l in lms], dtype=np.int32)

    # Forehead
    fx1, fy1, fx2, fy2 = get_bbox(pts, FOREHEAD_IDXS, w, h, PADDING)
    forehead = img[fy1:fy2, fx1:fx2]

    # Left cheek
    lx1, ly1, lx2, ly2 = get_bbox(pts, LEFT_CHEEK_IDXS, w, h, PADDING)
    left_cheek = img[ly1:ly2, lx1:lx2]

    # Right cheek
    rx1, ry1, rx2, ry2 = get_bbox(pts, RIGHT_CHEEK_IDXS, w, h, PADDING)
    right_cheek = img[ry1:ry2, rx1:rx2]

    base = os.path.splitext(os.path.basename(path))[0]
    out_forehead = os.path.join(OUTPUT_DIR, f"{base}_forehead.png")
    out_left = os.path.join(OUTPUT_DIR, f"{base}_left_cheek.png")
    out_right = os.path.join(OUTPUT_DIR, f"{base}_right_cheek.png")

    cv2.imwrite(out_forehead, forehead)
    cv2.imwrite(out_left, left_cheek)
    cv2.imwrite(out_right, right_cheek)

    print(f"[FULL] {path} ->")
    print(f"       {out_forehead}")
    print(f"       {out_left}")
    print(f"       {out_right}")


def collect_paths(cli_arg):
    if cli_arg is not None:
        return [cli_arg]
    paths = []
    for pat in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(glob.glob(os.path.join(DEFAULT_INPUT_DIR, pat)))
    return sorted(paths)


def main():
    cli_path = sys.argv[1] if len(sys.argv) > 1 else None
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img_paths = collect_paths(cli_path)
    if not img_paths:
        print("[ERROR] no images found")
        return

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        for p in img_paths:
            process_image(p, face_mesh)


if __name__ == "__main__":
    main()
