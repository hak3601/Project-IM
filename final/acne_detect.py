# acne_detect.py
# source /Users/shu/여드름탐지/Project-IM/.venv/bin/activate

# python3 -m venv .venv
# source .venv/bin/activate
# python -m streamlit run src/web/app.py


import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

def get_skin_mask_gmm_ycrcb(img_bgr, n_components=2, thr=0.5):
    """
    YCrCb의 Cr,Cb 만 가지고 GMM을 러닝해서 피부 클러스터 확률이 높은 픽셀만 남김.
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    _, Cr, Cb = cv2.split(ycrcb)

    # 크로마 벡터 (N,2)
    chroma = np.stack([Cr.flatten(), Cb.flatten()], axis=1).astype(np.float32)

    # 속도 위해 샘플링
    if chroma.shape[0] > 50000:
        idx = np.random.choice(chroma.shape[0], 50000, replace=False)
        sample = chroma[idx]
    else:
        sample = chroma

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=0
    )
    gmm.fit(sample)

    # 각 픽셀 log-likelihood → 0~1로 정규화
    log_prob = gmm.score_samples(chroma)
    lp_min, lp_max = log_prob.min(), log_prob.max()
    prob = (log_prob - lp_min) / (lp_max - lp_min + 1e-6)

    skin_mask = (prob > thr).astype(np.uint8).reshape(Cr.shape) * 255

    # 모폴로지로 다듬기
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, k5, iterations=1)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, k5, iterations=1)

    return skin_mask


def get_skin_mask_adaptive(img_bgr):
    """
    고정 YCrCb 박스 기반 마스크와 GMM 기반 마스크를 AND 로 결합.
    고정 박스는 잡음 제거, GMM은 조명/피부톤 적응.
    """
    mask_box = get_skin_mask_ycrcb(img_bgr)
    mask_gmm = get_skin_mask_gmm_ycrcb(img_bgr)

    mask = cv2.bitwise_and(mask_box, mask_gmm)
    mask = keep_largest_component(mask)
    mask = fill_holes(mask)
    return mask


# -----------------------------
# Tunable parameters
# -----------------------------
SMALL_COMP_MIN = 80

# Skin mask: 넓은 범위 + 큰 컴포넌트만 유지
SKIN_CR_LOW, SKIN_CR_HIGH = 133, 185
SKIN_CB_LOW, SKIN_CB_HIGH = 70, 150
SKIN_V_MIN = 35

# Red candidate: Lab delta + Hue gate + hysteresis
LAB_BG_K_SMALL = 21
LAB_BG_K_BIG = 51

H_RED_MAX = 25       # Hue가 0~25 또는 165~179
H_RED_MIN2 = 165

S_MIN = 25
V_MIN = 35

# delta/a 임계 (퍼센타일은 유지하되 "바닥값"을 올려서 과검출 방지)
DELTA_STRONG_PCTL = 92
DELTA_WEAK_PCTL = 78
A_STRONG_PCTL = 80
A_WEAK_PCTL = 60

DELTA_WEAK_FLOOR = 4.0
DELTA_STRONG_FLOOR = 8.0

A_WEAK_FLOOR = 132.0
A_STRONG_FLOOR = 138.0

# Morph
CLOSE_K = 7
OPEN_K = 3

# Watershed
WATERSHED_MIN_AREA = 2500
WATERSHED_DIST_RATIO = 0.50

# Contour area relative to skin
MIN_AREA_RATIO = 0.00008
MAX_AREA_RATIO = 0.12


# -----------------------------
# Utils
# -----------------------------
def _odd(k: int) -> int:
    k = int(k)
    if k < 3:
        return 3
    return k if k % 2 == 1 else k + 1

def remove_small_components(bin_mask, min_area):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    out = np.zeros_like(bin_mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out

def keep_largest_component(bin_mask):
    num, labels, stats, _ = cv2.connectedComponentsWithStats((bin_mask > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return bin_mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(bin_mask)
    out[labels == idx] = 255
    return out

def fill_holes(bin_mask):
    m = (bin_mask > 0).astype(np.uint8) * 255
    padded = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ph, pw = padded.shape[:2]
    flood = padded.copy()
    ffmask = np.zeros((ph + 2, pw + 2), np.uint8)  #(image_h+2, image_w+2)
    cv2.floodFill(flood, ffmask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, flood_inv)
    return filled[1:-1, 1:-1]


# ---------------------------------------
# Skin Mask - YCrCb (robust + largest CC)
# ---------------------------------------
def get_skin_mask_ycrcb(img_bgr):
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)

    skin = (
        (cr >= SKIN_CR_LOW) & (cr <= SKIN_CR_HIGH) &
        (cb >= SKIN_CB_LOW) & (cb <= SKIN_CB_HIGH) &
        (v >= SKIN_V_MIN)
    )
    mask = (skin.astype(np.uint8) * 255)

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)

    # 피부 덩어리(가장 큰 컴포넌트)만 유지
    mask = keep_largest_component(mask)
    mask = fill_holes(mask)

    return mask


# ---------------------------------------
# Red Candidate Mask (Lab delta + Hue gate + hysteresis)
# ---------------------------------------
def get_red_candidate_mask_hsv(img_bgr, skin_mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    _, a_u8, _ = cv2.split(lab)
    a = a_u8.astype(np.float32)

    skin_idx = skin_mask > 0
    if np.count_nonzero(skin_idx) == 0:
        return np.zeros_like(skin_mask)

    # Hue gate: 빨강 계열만 통과
    red_h = ((H <= H_RED_MAX) | (H >= H_RED_MIN2))

    # multi-scale delta (signed)
    k1 = _odd(LAB_BG_K_SMALL)
    k2 = _odd(LAB_BG_K_BIG)
    bg1 = cv2.medianBlur(a_u8, k1).astype(np.float32)
    bg2 = cv2.medianBlur(a_u8, k2).astype(np.float32)
    delta = np.maximum(a - bg1, a - bg2)  # float

    a_vals = a[skin_idx]
    d_vals = delta[skin_idx]

    a_strong = max(float(np.percentile(a_vals, A_STRONG_PCTL)), A_STRONG_FLOOR)
    a_weak   = max(float(np.percentile(a_vals, A_WEAK_PCTL)),   A_WEAK_FLOOR)

    d_strong = max(float(np.percentile(d_vals, DELTA_STRONG_PCTL)), DELTA_STRONG_FLOOR)
    d_weak   = max(float(np.percentile(d_vals, DELTA_WEAK_PCTL)),   DELTA_WEAK_FLOOR)

    # 밝기/채도 하한
    s_thr = max(float(np.percentile(S[skin_idx], 15)), S_MIN)
    v_thr = max(float(np.percentile(V[skin_idx], 10)), V_MIN)

    strong = (
        skin_idx & red_h &
        (S >= s_thr) & (V >= v_thr) &
        (a >= a_strong) & (delta >= d_strong)
    )
    weak = (
        skin_idx & red_h &
        (S >= max(s_thr - 5, S_MIN)) &
        (V >= max(v_thr - 10, V_MIN)) &
        (a >= a_weak) & (delta >= d_weak)
    )

    strong_u8 = np.zeros_like(skin_mask, np.uint8)
    weak_u8 = np.zeros_like(skin_mask, np.uint8)
    strong_u8[strong] = 255
    weak_u8[weak] = 255

    # hysteresis 확장: strong에서 시작해 weak로만 퍼지기
    cur = strong_u8.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(80):
        nxt = cv2.bitwise_and(cv2.dilate(cur, k, iterations=1), weak_u8)
        nxt = cv2.bitwise_or(nxt, strong_u8)
        if np.array_equal(nxt, cur):
            break
        cur = nxt

    cand = cur

    # post-morph
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(CLOSE_K), _odd(CLOSE_K)))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(OPEN_K), _odd(OPEN_K)))

    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kc, iterations=1)
    cand = fill_holes(cand)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, ko, iterations=1)
    cand = remove_small_components(cand, SMALL_COMP_MIN)

    return cand


# ---------------------------------------
# Mask smooth
# ---------------------------------------
def smooth_mask(mask):
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko, iterations=1)
    return m


# ---------------------------------------
# Watershed Separation
# ---------------------------------------
def separate_touching_spots_by_watershed(img_bgr, mask):
    mask_bin = (mask > 0).astype(np.uint8) * 255
    separated = np.zeros_like(mask_bin)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_bin

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < WATERSHED_MIN_AREA:
            cv2.drawContours(separated, [cnt], -1, 255, -1)
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi_mask = mask_bin[y:y+h, x:x+w]
        roi_img = img_bgr[y:y+h, x:x+w]

        dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            cv2.drawContours(separated, [cnt], -1, 255, -1)
            continue

        _, sure_fg = cv2.threshold(dist, WATERSHED_DIST_RATIO * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        sure_bg = cv2.dilate(roi_mask, np.ones((3, 3), np.uint8), iterations=1)
        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = markers.astype(np.int32)

        markers = cv2.watershed(roi_img, markers)

        roi_sep = np.zeros_like(roi_mask)
        for label in np.unique(markers):
            if label <= 1:
                continue
            roi_sep[markers == label] = 255

        separated[y:y+h, x:x+w] |= roi_sep

    return separated


# ---------------------------------------
# Final Contour Filtering
# ---------------------------------------
def filter_contours(img_bgr, mask, skin_mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, _, V = cv2.split(hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    acne_contours = []
    acne_areas = []
    skin_area = int(np.count_nonzero(skin_mask))

    min_area = max(80, int(skin_area * MIN_AREA_RATIO))
    max_area = max(8000, int(skin_area * MAX_AREA_RATIO))

    skin_v = V[skin_mask > 0]
    v_cut = max(float(np.percentile(skin_v, 8)), 40.0) if skin_v.size else 40.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        cnt_mask = np.zeros_like(mask)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

        mean_v = float(V[cnt_mask > 0].mean())
        if mean_v < v_cut:
            continue

        acne_contours.append(cnt)
        acne_areas.append(area)

    total_acne = float(np.sum(acne_areas)) if acne_areas else 0.0
    ratio = total_acne / skin_area if skin_area > 0 else 0.0
    return acne_contours, total_acne, skin_area, ratio


# ---------------------------------------
# Overlay Visual
# ---------------------------------------
def overlay_acne(img_bgr, contours):
    overlay = img_bgr.copy()
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), -1)
    result = cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0)
    cv2.drawContours(result, contours, -1, (0, 255, 255), 1)
    return result


# ---------------------------------------
# MAIN PIPELINE
# ---------------------------------------
def detect_acne_pipeline(img_bgr, debug=False):
    # 고정 박스 + GMM 앙상블 피부 마스크
    skin = get_skin_mask_adaptive(img_bgr)

    red = get_red_candidate_mask_hsv(img_bgr, skin)
    smooth = smooth_mask(red)
    separated = separate_touching_spots_by_watershed(img_bgr, smooth)

    acne_contours, total_acne, total_skin, ratio = filter_contours(
        img_bgr, separated, skin
    )
    overlay = overlay_acne(img_bgr, acne_contours)

    debug_maps = {
        "skin_mask": skin,
        "red_mask": red,
        "smooth_mask": smooth,
        "separated_mask": separated,
    }

    return overlay, {
        "num_spots": len(acne_contours),
        "total_acne_area": total_acne,
        "total_skin_area": total_skin,
        "area_ratio": ratio
    }, debug_maps
