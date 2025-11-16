# generate_output.py

import os
import glob
import cv2
import numpy as np

# ---------------- CONFIG ----------------
INPUT_DIR = "crops_out_skinmask_landmark"  # 검은색 마스크 적용된 이미지
OUTPUT_DIR = "crops_out_final_bbox"
DEBUG_DIR = "crops_out_debug_bbox"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
# ----------------------------------------

def get_nonblack_bbox(img):
    """
    검은색([0,0,0])이 아닌 영역의 최소 바운딩 박스 좌표 계산
    """
    coords = np.column_stack(np.where(np.any(img != [0,0,0], axis=2)))
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    print(f"Crop bbox: x[{x_min}:{x_max}] y[{y_min}:{y_max}]")
    return x_min, y_min, x_max, y_max

def process_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] Cannot read image: {path}")
        return

    bbox = get_nonblack_bbox(img)
    base = os.path.basename(path)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        # 원본 이미지에서 bbox 부분만 잘라서 저장
        cropped = img[y1:y2+1, x1:x2+1]
        out_path = os.path.join(OUTPUT_DIR, base)
        cv2.imwrite(out_path, cropped)
        print(f"[OK] {path} -> {out_path}")

        # 디버그용: 원본 이미지에 빨간색 박스 표시
        debug_img = img.copy()
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0,0,255), 2)
        debug_path = os.path.join(DEBUG_DIR, base)
        cv2.imwrite(debug_path, debug_img)
        print(f"[DEBUG] {debug_path} saved with bbox rectangle.")
    else:
        print(f"[INFO] No non-black region found in {path}")

def main():
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    img_paths = []
    for pat in patterns:
        img_paths.extend(glob.glob(os.path.join(INPUT_DIR, pat)))
    img_paths = sorted(img_paths)

    if not img_paths:
        print(f"[ERROR] No images found in directory: {INPUT_DIR}")
        return

    for path in img_paths:
        process_image(path)

if __name__ == "__main__":
    main()
