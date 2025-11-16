import os
import glob
import cv2
import mediapipe as mp
import numpy as np

# ----------------- CONFIG -----------------
INPUT_DIR = "images_in"      # folder with your face images
OUTPUT_DIR = "crops_out"     # where crops will be saved
PADDING = 0.06               # padding around landmark box (6%)
# ------------------------------------------


os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Landmark index sets (MediaPipe FaceMesh topology)
# subject's LEFT cheek (their left side)
LEFT_CHEEK_IDXS = [
    234, 93, 137, 132, 58, 172, 136, 150, 149, 176
]

# subject's RIGHT cheek (their right side)
RIGHT_CHEEK_IDXS = [
    454, 323, 366, 361, 288, 397, 365, 379, 378, 400
]

# Forehead region (roughly above eyebrows)
FOREHEAD_IDXS = [
    10,          # center forehead
    67, 103, 109, 66, 69,    # above left brow
    297, 334, 336, 296, 293  # above right brow
]


def get_bbox_for_indices(points_xy, indices, w, h, pad_ratio=0.05):
    """Return (x1, y1, x2, y2) padded bbox around given landmark indices."""
    region = points_xy[indices]
    x_min = np.min(region[:, 0])
    x_max = np.max(region[:, 0])
    y_min = np.min(region[:, 1])
    y_max = np.max(region[:, 1])

    pad_x = int((x_max - x_min) * pad_ratio)
    pad_y = int((y_max - y_min) * pad_ratio)

    x_min = max(0, x_min - pad_x)
    x_max = min(w - 1, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(h - 1, y_max + pad_y)

    return x_min, y_min, x_max, y_max


def process_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Skip (cannot read): {path}")
        return

    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        print(f"No face landmarks detected in {path}")
        return

    lms = result.multi_face_landmarks[0].landmark
    pts = np.array([(int(lm.x * w), int(lm.y * h)) for lm in lms], dtype=np.int32)

    # Forehead bbox and crop
    fx1, fy1, fx2, fy2 = get_bbox_for_indices(
        pts, FOREHEAD_IDXS, w, h, pad_ratio=PADDING
    )
    forehead = img[fy1:fy2, fx1:fx2]

    # Left and right cheek bboxes
    lx1, ly1, lx2, ly2 = get_bbox_for_indices(
        pts, LEFT_CHEEK_IDXS, w, h, pad_ratio=PADDING
    )
    rx1, ry1, rx2, ry2 = get_bbox_for_indices(
        pts, RIGHT_CHEEK_IDXS, w, h, pad_ratio=PADDING
    )

    left_cheek = img[ly1:ly2, lx1:lx2]
    right_cheek = img[ry1:ry2, rx1:rx2]

    # Decide which cheek is more visible by bbox area
    area_left = (lx2 - lx1) * (ly2 - ly1)
    area_right = (rx2 - rx1) * (ry2 - ry1)

    if area_left >= area_right:
        cheek = left_cheek
        cheek_side = "left"
    else:
        cheek = right_cheek
        cheek_side = "right"

    base = os.path.splitext(os.path.basename(path))[0]
    out_forehead = os.path.join(OUTPUT_DIR, f"{base}_forehead.png")
    out_cheek = os.path.join(OUTPUT_DIR, f"{base}_cheek_{cheek_side}.png")

    cv2.imwrite(out_forehead, forehead)
    cv2.imwrite(out_cheek, cheek)

    print(f"{path} -> {out_forehead}, {out_cheek} (picked {cheek_side} cheek)")


# Process all jpg/png images in INPUT_DIR
patterns = ["*.jpg", "*.jpeg", "*.png"]
for pat in patterns:
    for img_path in glob.glob(os.path.join(INPUT_DIR, pat)):
        process_image(img_path)
