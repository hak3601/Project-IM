# crop_regions_full.py
import os
import sys
import glob
import cv2
import mediapipe as mp
import numpy as np

DEFAULT_INPUT_DIR = "images_in"
OUTPUT_DIR = "crops_out"

PADDING_X = 0.06   # horizontal padding
PADDING_Y = 0.06   # vertical padding

# --- Landmark index groups (MediaPipe FaceMesh) ---

# Rough forehead band (upper forehead area)
FOREHEAD_IDXS = [10, 67, 103, 109, 338, 297, 337, 151]

# Eyebrow upper boundary (used as lower border for forehead)
LEFT_EB_UPPER  = [70, 63, 105, 66, 107]
RIGHT_EB_UPPER = [336, 296, 295, 334, 293]

# Cheek horizontal extents (we'll clamp vertically between eye and lip)
LEFT_CHEEK_IDXS  = [234, 93, 137, 132, 58, 172, 136, 150, 149, 176]
RIGHT_CHEEK_IDXS = [454, 323, 366, 361, 288, 397, 365, 379, 378, 400]

# Eye lower lids – define “under the eyes”
LEFT_EYE_LOWER  = [145, 159]
RIGHT_EYE_LOWER = [374, 386]

# Upper and lower lip anchors
UPPER_LIP = [13]        # mouth top center
LOWER_LIP = [14]        # mouth bottom center

# Chin / jaw for horizontal extent and bottom
CHIN_IDXS = [152, 176, 148, 149, 150, 400, 379, 378, 365]


def get_xy(landmarks, w, h):
    return np.array([(int(l.x * w), int(l.y * h)) for l in landmarks], dtype=np.int32)


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
    pts = get_xy(lms, w, h)

    # Convenience picks
    eyebrow_pts = pts[LEFT_EB_UPPER + RIGHT_EB_UPPER]
    eye_lower_left  = pts[LEFT_EYE_LOWER]
    eye_lower_right = pts[RIGHT_EYE_LOWER]
    upper_lip_pts   = pts[UPPER_LIP]
    lower_lip_pts   = pts[LOWER_LIP]
    chin_pts        = pts[CHIN_IDXS]

    eyebrow_y = int(np.min(eyebrow_pts[:, 1]))             # top of eyebrows
    eye_y_left  = int(np.mean(eye_lower_left[:, 1]))       # lower eyelid left
    eye_y_right = int(np.mean(eye_lower_right[:, 1]))      # lower eyelid right
    upper_lip_y = int(np.mean(upper_lip_pts[:, 1]))        # top of lip
    lower_lip_y = int(np.mean(lower_lip_pts[:, 1]))        # bottom of lip
    chin_bottom_y = int(np.max(chin_pts[:, 1]))            # lowest chin point

    # Small relative offsets so we stay strictly in the desired band
    def band_y(top, bottom, inner_ratio_top=0.05, inner_ratio_bottom=0.05):
        span = bottom - top
        y1 = int(top + inner_ratio_top * span)
        y2 = int(bottom - inner_ratio_bottom * span)
        return max(0, y1), min(h - 1, y2)

    # ----------------- Forehead (above eyebrows) -----------------
    fh_pts = pts[FOREHEAD_IDXS]
    x_min_fh = int(np.min(fh_pts[:, 0]))
    x_max_fh = int(np.max(fh_pts[:, 0]))

    # vertical: from forehead band down to just above eyebrows
    fh_top = int(np.min(fh_pts[:, 1]))
    fh_bottom = eyebrow_y
    y1_fh, y2_fh = fh_top, fh_bottom - 1  # explicitly end above brows

    # padding
    dx_fh = int((x_max_fh - x_min_fh) * PADDING_X)
    dy_fh = int((y2_fh - y1_fh) * PADDING_Y) if y2_fh > y1_fh else 0

    x1_fh = max(0, x_min_fh - dx_fh)
    x2_fh = min(w - 1, x_max_fh + dx_fh)
    y1_fh = max(0, y1_fh - dy_fh)
    y2_fh = max(y1_fh + 1, min(h - 1, y2_fh + dy_fh))

    forehead = img[y1_fh:y2_fh, x1_fh:x2_fh]

    # ----------------- Cheeks (under eyes, above lip) ------------

    # vertical band for cheeks: between eye-lower and upper lip
    cheek_top_left, cheek_bottom_left = band_y(eye_y_left, upper_lip_y)
    cheek_top_right, cheek_bottom_right = band_y(eye_y_right, upper_lip_y)

    # left cheek
    l_pts = pts[LEFT_CHEEK_IDXS]
    lx_min = int(np.min(l_pts[:, 0]))
    lx_max = int(np.max(l_pts[:, 0]))
    dx_l = int((lx_max - lx_min) * PADDING_X)

    x1_l = max(0, lx_min - dx_l)
    x2_l = min(w - 1, lx_max + dx_l)
    y1_l = cheek_top_left
    y2_l = cheek_bottom_left

    left_cheek = img[y1_l:y2_l, x1_l:x2_l]

    # right cheek
    r_pts = pts[RIGHT_CHEEK_IDXS]
    rx_min = int(np.min(r_pts[:, 0]))
    rx_max = int(np.max(r_pts[:, 0]))
    dx_r = int((rx_max - rx_min) * PADDING_X)

    x1_r = max(0, rx_min - dx_r)
    x2_r = min(w - 1, rx_max + dx_r)
    y1_r = cheek_top_right
    y2_r = cheek_bottom_right

    right_cheek = img[y1_r:y2_r, x1_r:x2_r]

    # ----------------- Chin (from under lip to chin) -------------
    cx_min = int(np.min(chin_pts[:, 0]))
    cx_max = int(np.max(chin_pts[:, 0]))

    # vertical: start a bit below lower lip, end at chin bottom
    chin_top_raw, chin_bottom_raw = lower_lip_y, chin_bottom_y
    chin_top, chin_bottom = band_y(chin_top_raw, chin_bottom_raw,
                                   inner_ratio_top=0.1, inner_ratio_bottom=0.0)

    dx_c = int((cx_max - cx_min) * PADDING_X)
    x1_c = max(0, cx_min - dx_c)
    x2_c = min(w - 1, cx_max + dx_c)

    y1_c = chin_top
    y2_c = min(h - 1, chin_bottom + int((chin_bottom - chin_top) * PADDING_Y))

    chin = img[y1_c:y2_c, x1_c:x2_c]

    # ----------------- Save all four regions ---------------------
    base = os.path.splitext(os.path.basename(path))[0]
    out_forehead = os.path.join(OUTPUT_DIR, f"{base}_forehead.png")
    out_left = os.path.join(OUTPUT_DIR, f"{base}_left_cheek.png")
    out_right = os.path.join(OUTPUT_DIR, f"{base}_right_cheek.png")
    out_chin = os.path.join(OUTPUT_DIR, f"{base}_chin.png")

    cv2.imwrite(out_forehead, forehead)
    cv2.imwrite(out_left, left_cheek)
    cv2.imwrite(out_right, right_cheek)
    cv2.imwrite(out_chin, chin)

    print(f"[FULL] {path} ->")
    print(f"       {out_forehead}")
    print(f"       {out_left}")
    print(f"       {out_right}")
    print(f"       {out_chin}")


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
