# crop_regions_auto.py
import os
import sys
import glob
import cv2
import mediapipe as mp
import numpy as np

DEFAULT_INPUT_DIR = "images_in"
OUTPUT_DIR = "crops_out"

FULL_FACE_EYE_RATIO_THR = 0.28
PADDING = 0.06


# ---------- Mediapipe FaceMesh indices ----------
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

LEFT_EYE_LOWER = [145, 159]
RIGHT_EYE_LOWER = [374, 386]

LEFT_EB_UPPER  = [70, 63, 105, 66, 107]
RIGHT_EB_UPPER = [336, 296, 295, 334, 293]

FOREHEAD_IDXS = [10, 67, 103, 109, 338, 297, 337, 151]

LEFT_CHEEK_IDXS  = [234, 93, 137, 132, 58, 172, 136, 150, 149, 176]
RIGHT_CHEEK_IDXS = [454, 323, 366, 361, 288, 397, 365, 379, 378, 400]

UPPER_LIP = [13]
LOWER_LIP = [14]

CHIN_IDXS = [152, 176, 148, 149, 150, 400, 379, 378, 365]


# ---------- Utilities ----------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def landmarks_to_xy(landmarks, w, h):
    return np.array([(int(l.x * w), int(l.y * h)) for l in landmarks], dtype=np.int32)


def collect_paths(cli_arg):
    if cli_arg is not None:
        return [cli_arg]
    paths = []
    for pat in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(glob.glob(os.path.join(DEFAULT_INPUT_DIR, pat)))
    return sorted(paths)


# ---------- Orientation detection ----------
def detect_face_orientation(pts, img_w):
    """Return 'full' or 'side' based on eye distance."""
    xL = pts[LEFT_EYE_OUTER, 0]
    xR = pts[RIGHT_EYE_OUTER, 0]
    eye_dist = abs(xR - xL)

    ratio = eye_dist / float(img_w)

    if ratio >= FULL_FACE_EYE_RATIO_THR:
        return "full"
    else:
        return "side"


# ===============================================================
#                  FULL-FACE CROPPER
# ===============================================================
def crop_full(img, pts):
    h, w = img.shape[:2]
    regions = {}

    # ------------------------------------------
    # Forehead: above eyebrows
    # ------------------------------------------
    eyebrow_pts = pts[LEFT_EB_UPPER + RIGHT_EB_UPPER]
    eyebrow_y = int(np.min(eyebrow_pts[:, 1]))

    fh_pts = pts[FOREHEAD_IDXS]
    x_min = int(np.min(fh_pts[:, 0]))
    x_max = int(np.max(fh_pts[:, 0]))
    y_top = int(np.min(fh_pts[:, 1]))
    y_bottom = eyebrow_y - 2

    dx = int((x_max - x_min) * PADDING)
    dy = int((y_bottom - y_top) * PADDING)

    x1 = clamp(x_min - dx, 0, w - 2)
    x2 = clamp(x_max + dx, x1 + 1, w - 1)
    y1 = clamp(y_top - dy, 0, h - 2)
    y2 = clamp(y_bottom + dy, y1 + 1, h - 1)

    regions["forehead"] = img[y1:y2, x1:x2]

    # ------------------------------------------
    # Cheeks: below eyes, above lip
    # ------------------------------------------
    def cheek_vertical_band(eye_y, upper_lip_y):
        gap = upper_lip_y - eye_y
        top = eye_y + int(0.15 * gap)
        bottom = upper_lip_y - int(0.15 * gap)
        return clamp(top, 0, h - 2), clamp(bottom, 1, h - 1)

    # anchor points
    eye_lower_left_y = int(np.mean(pts[LEFT_EYE_LOWER][:, 1]))
    eye_lower_right_y = int(np.mean(pts[RIGHT_EYE_LOWER][:, 1]))
    upper_lip_y = int(np.mean(pts[UPPER_LIP][:, 1]))

    # left cheek
    y1_l, y2_l = cheek_vertical_band(eye_lower_left_y, upper_lip_y)
    lc = pts[LEFT_CHEEK_IDXS]
    x_min_l = int(np.min(lc[:, 0]))
    x_max_l = int(np.max(lc[:, 0]))
    dx_l = int((x_max_l - x_min_l) * PADDING)
    x1_l = clamp(x_min_l - dx_l, 0, w - 2)
    x2_l = clamp(x_max_l + dx_l, x1_l + 1, w - 1)
    regions["left_cheek"] = img[y1_l:y2_l, x1_l:x2_l]

    # right cheek
    y1_r, y2_r = cheek_vertical_band(eye_lower_right_y, upper_lip_y)
    rc = pts[RIGHT_CHEEK_IDXS]
    x_min_r = int(np.min(rc[:, 0]))
    x_max_r = int(np.max(rc[:, 0]))
    dx_r = int((x_max_r - x_min_r) * PADDING)
    x1_r = clamp(x_min_r - dx_r, 0, w - 2)
    x2_r = clamp(x_max_r + dx_r, x1_r + 1, w - 1)
    regions["right_cheek"] = img[y1_r:y2_r, x1_r:x2_r]

    # ------------------------------------------
    # Chin: strictly under lips
    # ------------------------------------------
    lower_lip_y = int(np.mean(pts[LOWER_LIP][:, 1]))
    chin_pts = pts[CHIN_IDXS]
    chin_bottom_y = int(np.max(chin_pts[:, 1]))

    gap = max(1, chin_bottom_y - lower_lip_y)
    chin_top = lower_lip_y + int(0.25 * gap)
    chin_bottom = chin_bottom_y

    # narrow chin horizontally
    lip_pts = np.vstack([pts[UPPER_LIP], pts[LOWER_LIP]])
    mouth_center_x = int(np.mean(lip_pts[:, 0]))
    chin_half_width = int(0.18 * w)

    x1_c = clamp(mouth_center_x - chin_half_width, 0, w - 2)
    x2_c = clamp(mouth_center_x + chin_half_width, x1_c + 1, w - 1)
    y1_c = clamp(chin_top, 0, h - 2)
    y2_c = clamp(chin_bottom, y1_c + 1, h - 1)

    regions["chin"] = img[y1_c:y2_c, x1_c:x2_c]

    return regions


# ===============================================================
#                  SIDE-FACE CROPPER
# ===============================================================
def crop_side(img, pts):
    h, w = img.shape[:2]
    regions = {}

    # ------------------------------------------
    # Forehead (same logic as full-face)
    # ------------------------------------------
    eyebrow_pts = pts[LEFT_EB_UPPER + RIGHT_EB_UPPER]
    eyebrow_y = int(np.min(eyebrow_pts[:, 1]))

    fh_pts = pts[FOREHEAD_IDXS]
    x_min = int(np.min(fh_pts[:, 0]))
    x_max = int(np.max(fh_pts[:, 0]))
    y_top = int(np.min(fh_pts[:, 1]))
    y_bottom = eyebrow_y - 2

    dx = int((x_max - x_min) * PADDING)
    dy = int((y_bottom - y_top) * PADDING)

    x1 = clamp(x_min - dx, 0, w - 2)
    x2 = clamp(x_max + dx, x1 + 1, w - 1)
    y1 = clamp(y_top - dy, 0, h - 2)
    y2 = clamp(y_bottom + dy, y1 + 1, h - 1)

    regions["forehead"] = img[y1:y2, x1:x2]

    # ------------------------------------------
    # Select visible cheek and ENLARGE it
    # ------------------------------------------
    eye_lower_left_y = int(np.mean(pts[LEFT_EYE_LOWER][:, 1]))
    eye_lower_right_y = int(np.mean(pts[RIGHT_EYE_LOWER][:, 1]))
    upper_lip_y = int(np.mean(pts[UPPER_LIP][:, 1]))

    def cheek_vertical_band(eye_y):
        gap = upper_lip_y - eye_y
        top = eye_y + int(0.10 * gap)
        bottom = upper_lip_y - int(0.10 * gap)
        return clamp(top, 0, h - 2), clamp(bottom, 1, h - 1)

    # left cheek
    y1_l, y2_l = cheek_vertical_band(eye_lower_left_y)
    lc = pts[LEFT_CHEEK_IDXS]
    x1_l = int(np.min(lc[:, 0]))
    x2_l = int(np.max(lc[:, 0]))
    left_area = (x2_l - x1_l) * (y2_l - y1_l)

    # right cheek
    y1_r, y2_r = cheek_vertical_band(eye_lower_right_y)
    rc = pts[RIGHT_CHEEK_IDXS]
    x1_r = int(np.min(rc[:, 0]))
    x2_r = int(np.max(rc[:, 0]))
    right_area = (x2_r - x1_r) * (y2_r - y1_r)

    # pick visible cheek
    if left_area >= right_area:
        side = "left"
        x1c, x2c, y1c, y2c = x1_l, x2_l, y1_l, y2_l
    else:
        side = "right"
        x1c, x2c, y1c, y2c = x1_r, x2_r, y1_r, y2_r

    # enlarge cheek box (side faces need bigger region)
    enlarge = 0.30
    ex = int((x2c - x1c) * enlarge)
    ey = int((y2c - y1c) * enlarge)

    x1_big = clamp(x1c - ex, 0, w - 2)
    x2_big = clamp(x2c + ex, x1_big + 1, w - 1)
    y1_big = clamp(y1c - ey, 0, h - 2)
    y2_big = clamp(y2c + ey, y1_big + 1, h - 1)

    regions[f"cheek_visible_{side}"] = img[y1_big:y2_big, x1_big:x2_big]

    # ------------------------------------------
    # Chin (same logic as full)
    # ------------------------------------------
    lower_lip_y = int(np.mean(pts[LOWER_LIP][:, 1]))
    chin_pts = pts[CHIN_IDXS]
    chin_bottom_y = int(np.max(chin_pts[:, 1]))

    gap = chin_bottom_y - lower_lip_y
    chin_top = lower_lip_y + int(0.25 * gap)
    chin_bottom = chin_bottom_y

    lip_pts = np.vstack([pts[UPPER_LIP], pts[LOWER_LIP]])
    mouth_center_x = int(np.mean(lip_pts[:, 0]))

    chin_half_width = int(0.22 * w)
    x1_c = clamp(mouth_center_x - chin_half_width, 0, w - 2)
    x2_c = clamp(mouth_center_x + chin_half_width, x1_c + 1, w - 1)

    y1_c = clamp(chin_top, 0, h - 2)
    y2_c = clamp(chin_bottom, y1_c + 1, h - 1)

    regions["chin"] = img[y1_c:y2_c, x1_c:x2_c]

    return regions


# ===============================================================
#                      MAIN EXECUTION
# ===============================================================
def process_image(path, face_mesh):
    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] Cannot read {path}")
        return

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        print(f"[WARN] No face landmarks detected: {path}")
        return

    pts = landmarks_to_xy(res.multi_face_landmarks[0].landmark, w, h)

    mode = detect_face_orientation(pts, w)

    if mode == "full":
        regions = crop_full(img, pts)
    else:
        regions = crop_side(img, pts)

    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[{mode.upper()}] {path}")
    for name, patch in regions.items():
        out_path = os.path.join(OUTPUT_DIR, f"{base}_{name}.png")
        cv2.imwrite(out_path, patch)
        print("  →", out_path)


def main():
    cli_path = sys.argv[1] if len(sys.argv) > 1 else None
    img_paths = collect_paths(cli_path)
    if not img_paths:
        print("No images found.")
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
