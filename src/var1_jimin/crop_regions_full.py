# crop_regions_skinmask_landmark_limited.py

import os
import sys
import glob
import cv2
import mediapipe as mp
import numpy as np

# ---------------- CONFIG ----------------
DEFAULT_INPUT_DIR = "images_in"
OUTPUT_DIR = "crops_out_skinmask_landmark"
PADDING = 0.06   # landmark bbox padding
# 피부 HSV 범위 (현실 피부톤 기준)
LOWER_SKIN = np.array([0, 30, 40])
UPPER_SKIN = np.array([25, 150, 255])
# ----------------------------------------

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

def apply_skin_mask(img_crop):
    """HSV 기반 피부 마스크"""
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)

    # 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    skin = cv2.bitwise_and(img_crop, img_crop, mask=mask)
    return skin

def crop_largest_skin_bbox(skin_img):
    """
    검은색 제외 후 최대 피부 영역 직사각형으로 crop
    """
    coords = np.column_stack(np.where(np.any(skin_img != [0,0,0], axis=2)))
    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # 디버그 bbox
    print(f"Crop bbox: x[{x_min}:{x_max}] y[{y_min}:{y_max}]")

    return skin_img[y_min:y_max+1, x_min:x_max+1]

def process_single_image(path, face_mesh):
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
    base = os.path.splitext(os.path.basename(path))[0]

    # 이마
    fx1, fy1, fx2, fy2 = get_bbox(pts, FOREHEAD_IDXS, w, h, PADDING)
    forehead_crop = img[fy1:fy2, fx1:fx2]
    forehead_skin = apply_skin_mask(forehead_crop)
    forehead_maxrect = crop_largest_skin_bbox(forehead_skin)
    if forehead_maxrect is not None:
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_forehead.png"), forehead_maxrect)

    # 왼쪽 볼
    lx1, ly1, lx2, ly2 = get_bbox(pts, LEFT_CHEEK_IDXS, w, h, PADDING)
    left_crop = img[ly1:ly2, lx1:lx2]
    left_skin = apply_skin_mask(left_crop)
    left_maxrect = crop_largest_skin_bbox(left_skin)
    if left_maxrect is not None:
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_left_cheek.png"), left_maxrect)

    # 오른쪽 볼
    rx1, ry1, rx2, ry2 = get_bbox(pts, RIGHT_CHEEK_IDXS, w, h, PADDING)
    right_crop = img[ry1:ry2, rx1:rx2]
    right_skin = apply_skin_mask(right_crop)
    right_maxrect = crop_largest_skin_bbox(right_skin)
    if right_maxrect is not None:
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_right_cheek.png"), right_maxrect)

    print(f"[OK] {path} -> Forehead/Left/Right cheek max rect skin crops saved.")

def collect_input_images(cli_arg_path=None):
    if cli_arg_path is not None:
        return [cli_arg_path]
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(DEFAULT_INPUT_DIR, pat)))
    return sorted(paths)

def main():
    cli_path = sys.argv[1] if len(sys.argv) > 1 else None
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_paths = collect_input_images(cli_path)
    if not img_paths:
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
