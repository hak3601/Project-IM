# crop_regions_auto.py
import os
import sys
import glob
import cv2
import mediapipe as mp
import numpy as np

DEFAULT_INPUT_DIR = "images_in"
OUTPUT_DIR = "crops_out"

PADDING = 0.06
FULL_FACE_EYE_RATIO_THR = 0.28  # eye distance / image width threshold


# ------------- FaceMesh landmark indices -------------

# Eyes for orientation + cheek vertical band
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_LOWER = [145, 159]
RIGHT_EYE_LOWER = [374, 386]

# Eyebrows for forehead band
LEFT_EB_UPPER = [70, 63, 105, 66, 107]
RIGHT_EB_UPPER = [336, 296, 295, 334, 293]

# Forehead for FULL mode
FOREHEAD_FULL_IDXS = [10, 67, 103, 109, 338, 297, 337, 151]

# Forehead for SIDE mode (your “perfect” setting)
FOREHEAD_SIDE_IDXS = [10, 67, 103, 109, 66, 69, 297, 334, 336, 296, 293]

# Cheeks (same for full & side)
LEFT_CHEEK_IDXS = [234, 93, 137, 132, 58, 172, 136, 150, 149, 176]
RIGHT_CHEEK_IDXS = [454, 323, 366, 361, 288, 397, 365, 379, 378, 400]

# Lips and chin
UPPER_LIP = [13]   # top of mouth
LOWER_LIP = [14]   # bottom of mouth
CHIN_IDXS = [152, 176, 148, 149, 150, 400, 379, 378, 365]


# ------------- Small helpers -------------

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


def get_bbox(pts_xy, indices, w, h, padding_ratio):
    """Simple bbox with uniform padding (used for side mode)."""
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


# ------------- Orientation detection -------------

def detect_face_orientation(pts, img_w):
    """Return 'full' or 'side' based on eye distance / image width."""
    xL = pts[LEFT_EYE_OUTER, 0]
    xR = pts[RIGHT_EYE_OUTER, 0]
    eye_dist = abs(xR - xL)
    ratio = eye_dist / float(img_w)
    return "full" if ratio >= FULL_FACE_EYE_RATIO_THR else "side"


# =========================================================
#                      FULL-FACE MODE
# =========================================================

def crop_full(img, pts):
    """
    Full-face: return forehead, left_cheek, right_cheek, chin.
    Rules:
      - Forehead: above eyebrows
      - Cheeks: under eyes, above upper lip
      - Chin: only under lips (no lips)
    """
    h, w = img.shape[:2]
    regions = {}

    # ---------- Forehead (above eyebrows) ----------
    eyebrow_pts = pts[LEFT_EB_UPPER + RIGHT_EB_UPPER]
    eyebrow_y = int(np.min(eyebrow_pts[:, 1]))

    fh_pts = pts[FOREHEAD_FULL_IDXS]
    x_min_fh = int(np.min(fh_pts[:, 0]))
    x_max_fh = int(np.max(fh_pts[:, 0]))
    y_top_fh = int(np.min(fh_pts[:, 1]))
    y_bottom_fh = eyebrow_y - 2  # stop just above brows

    dx_fh = int((x_max_fh - x_min_fh) * PADDING)
    dy_fh = int((y_bottom_fh - y_top_fh) * PADDING)

    x1_fh = clamp(x_min_fh - dx_fh, 0, w - 2)
    x2_fh = clamp(x_max_fh + dx_fh, x1_fh + 1, w - 1)
    y1_fh = clamp(y_top_fh - dy_fh, 0, h - 2)
    y2_fh = clamp(y_bottom_fh + dy_fh, y1_fh + 1, h - 1)

    regions["forehead"] = img[y1_fh:y2_fh, x1_fh:x2_fh]

    # ---------- Cheeks: under eyes, above upper lip ----------
    eye_lower_left_y = int(np.mean(pts[LEFT_EYE_LOWER][:, 1]))
    eye_lower_right_y = int(np.mean(pts[RIGHT_EYE_LOWER][:, 1]))
    upper_lip_y = int(np.mean(pts[UPPER_LIP][:, 1]))

    def cheek_band_y(eye_y):
        gap = upper_lip_y - eye_y
        top = eye_y + int(0.15 * gap)
        bottom = upper_lip_y - int(0.15 * gap)
        return clamp(top, 0, h - 2), clamp(bottom, 1, h - 1)

    # left cheek
    y1_l, y2_l = cheek_band_y(eye_lower_left_y)
    lc = pts[LEFT_CHEEK_IDXS]
    x_min_l = int(np.min(lc[:, 0]))
    x_max_l = int(np.max(lc[:, 0]))
    dx_l = int((x_max_l - x_min_l) * PADDING)
    x1_l = clamp(x_min_l - dx_l, 0, w - 2)
    x2_l = clamp(x_max_l + dx_l, x1_l + 1, w - 1)
    regions["left_cheek"] = img[y1_l:y2_l, x1_l:x2_l]

    # right cheek
    y1_r, y2_r = cheek_band_y(eye_lower_right_y)
    rc = pts[RIGHT_CHEEK_IDXS]
    x_min_r = int(np.min(rc[:, 0]))
    x_max_r = int(np.max(rc[:, 0]))
    dx_r = int((x_max_r - x_min_r) * PADDING)
    x1_r = clamp(x_min_r - dx_r, 0, w - 2)
    x2_r = clamp(x_max_r + dx_r, x1_r + 1, w - 1)
    regions["right_cheek"] = img[y1_r:y2_r, x1_r:x2_r]

    # ---------- Chin: only under lips, no lip ----------
    lower_lip_y = int(np.mean(pts[LOWER_LIP][:, 1]))
    chin_pts = pts[CHIN_IDXS]
    chin_bottom_y = int(np.max(chin_pts[:, 1]))

    gap_chin = max(1, chin_bottom_y - lower_lip_y)
    chin_top = lower_lip_y + int(0.25 * gap_chin)  # start 25% below lower lip
    chin_bottom = chin_bottom_y

    lip_pts = np.vstack([pts[UPPER_LIP], pts[LOWER_LIP]])
    mouth_center_x = int(np.mean(lip_pts[:, 0]))

    chin_half_width = int(0.18 * w)  # central 36% of width
    x1_c = clamp(mouth_center_x - chin_half_width, 0, w - 2)
    x2_c = clamp(mouth_center_x + chin_half_width, x1_c + 1, w - 1)
    y1_c = clamp(chin_top, 0, h - 2)
    y2_c = clamp(chin_bottom, y1_c + 1, h - 1)

    regions["chin"] = img[y1_c:y2_c, x1_c:x2_c]

    return regions


# =========================================================
#                      SIDE-FACE MODE
# =========================================================

def crop_side(img, pts):
    """
    Side-face: use your exact bbox logic (no enlargement).
    Returns:
      - 'forehead'
      - 'cheek_visible_left' OR 'cheek_visible_right'
    """
    h, w = img.shape[:2]
    regions = {}

    # -------------------------------------------------------
    # FOREHEAD (use full-face logic: above eyebrows)
    # -------------------------------------------------------
    eyebrow_pts = pts[LEFT_EB_UPPER + RIGHT_EB_UPPER]
    eyebrow_y = int(np.min(eyebrow_pts[:, 1]))  # top of eyebrows

    fh_pts = pts[FOREHEAD_FULL_IDXS]  # reuse full-mode forehead indices
    x_min_fh = int(np.min(fh_pts[:, 0]))
    x_max_fh = int(np.max(fh_pts[:, 0]))
    y_top_fh = int(np.min(fh_pts[:, 1]))
    y_bottom_fh = eyebrow_y - 2

    dx_fh = int((x_max_fh - x_min_fh) * PADDING)
    dy_fh = int((y_bottom_fh - y_top_fh) * PADDING)

    x1_fh = clamp(x_min_fh - dx_fh, 0, w - 2)
    x2_fh = clamp(x_max_fh + dx_fh, x1_fh + 1, w - 1)
    y1_fh = clamp(y_top_fh - dy_fh, 0, h - 2)
    y2_fh = clamp(y_bottom_fh + dy_fh, y1_fh + 1, h - 1)

    regions["forehead"] = img[y1_fh:y2_fh, x1_fh:x2_fh]

    # -------------------------------------------------------
    # CHEEKS 
    # -------------------------------------------------------
    lx1, ly1, lx2, ly2 = get_bbox(pts, LEFT_CHEEK_IDXS, w, h, PADDING)
    rx1, ry1, rx2, ry2 = get_bbox(pts, RIGHT_CHEEK_IDXS, w, h, PADDING)

    left_cheek = img[ly1:ly2, lx1:lx2]
    right_cheek = img[ry1:ry2, rx1:rx2]

    area_left = (lx2 - lx1) * (ly2 - ly1)
    area_right = (rx2 - rx1) * (ry2 - ry1)

    if area_left >= area_right:
        regions["left_cheek"] = left_cheek
    else:
        regions["right_cheek"] = right_cheek

    return regions


# =========================================================
#                      MAIN PIPELINE
# =========================================================

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

    pts = landmarks_to_xy(res.multi_face_landmarks[0].landmark, w, h)
    
    # Decide mode from filename: s_* = side, f_* = full
    fname = os.path.basename(path).lower()
    if fname.startswith("s"):
        mode = "side"
    elif fname.startswith("f"):
        mode = "full"
    else:
        print(f"[ERROR] Filename must start with 's' (side) or 'f' (full): {fname}")
        return

    if mode == "full":
        regions = crop_full(img, pts)
    else:
        regions = crop_side(img, pts)
    '''   
    # --- Determine mode according to eye distance ratio ---
    mode = detect_face_orientation(pts, w)  # 'full' or 'side'

    if mode == "full":
        regions = crop_full(img, pts)
    else:
        regions = crop_side(img, pts)
    #====================================================== 
    ''' 
    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[{mode.upper()}] {path}")
    for name, patch in regions.items():
        out_path = os.path.join(OUTPUT_DIR, f"{base}_{name}.png")
        cv2.imwrite(out_path, patch)
        print("   →", out_path)


def main():
    cli_path = sys.argv[1] if len(sys.argv) > 1 else None
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
