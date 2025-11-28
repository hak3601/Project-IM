# acne_detect.py
# source /Users/shu/여드름탐지/Project-IM/.venv/bin/activate
# python -m streamlit run src/web/app.py



import cv2
import numpy as np

# -----------------------------
# Tunable parameters (여기만 조절)
# -----------------------------
SMALL_COMP_MIN = 50        # 후보 마스크에서 제거할 최소 컴포넌트 픽셀(노이즈 제거)
LAB_BG_K = 21              # a* 배경(저주파) 추정용 medianBlur 커널(홀수)
DELTA_PCTL = 90            # 국소 Δa 상위 퍼센타일(낮출수록 더 많이 잡힘) 88~95
A_PCTL = 55                # a* 절대값 퍼센타일(낮출수록 더 많이 잡힘) 50~70
V_PCTL = 15                # V(밝기) 퍼센타일 하한(어두움 제거) 10~25
S_PCTL = 15                # S(채도) 퍼센타일 하한(회색 그림자 제거) 10~25

WATERSHED_MIN_AREA = 3000  # 이 면적 이상 blob만 watershed로 분리
WATERSHED_DIST_RATIO = 0.55  # 낮을수록 더 잘 쪼개짐(0.30~0.55)d

MIN_AREA_RATIO = 0.00006   # 피부 대비 최소 면적 (작은 점 제거)
MAX_AREA_RATIO = 0.06      # 피부 대비 최대 면적 (너무 큰 패치 제거)


# ---------------------------------------
# util: remove small connected components
# ---------------------------------------
def remove_small_components(bin_mask, min_area):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    out = np.zeros_like(bin_mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


# ---------------------------------------
# Skin Mask - YCrCb
# ---------------------------------------
def get_skin_mask_ycrcb(img_bgr):
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 140, 95], dtype=np.uint8)
    upper = np.array([255, 165, 120], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


# ---------------------------------------
# Red Candidate Mask
# (함수명 유지. 내부 방식 변경: HSV(H) 단독 -> Lab a* 국소 대비(Δa) + 밝기/채도 하한)
# ---------------------------------------
def get_red_candidate_mask_hsv(img_bgr, skin_mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    _, a, _ = cv2.split(lab)  # OpenCV Lab: a의 중립이 대략 128, 클수록 red 쪽

    skin_idx = skin_mask > 0
    if np.count_nonzero(skin_idx) == 0:
        return np.zeros_like(skin_mask)

    # a*의 "주변 대비" (국소 붉음 강조) : delta = a - medianBlur(a)
    bg = cv2.medianBlur(a, LAB_BG_K)
    delta = cv2.subtract(a, bg)  # uint8, 음수는 0으로 클리핑됨

    delta_vals = delta[skin_idx]
    a_vals = a[skin_idx]
    v_vals = V[skin_idx]
    s_vals = S[skin_idx]

    # 퍼센타일 기반 임계값(조명 변화에 강함)
    delta_thr = float(np.percentile(delta_vals, DELTA_PCTL))
    a_thr = float(np.percentile(a_vals, A_PCTL))
    v_thr = float(np.percentile(v_vals, V_PCTL))
    s_thr = float(np.percentile(s_vals, S_PCTL))

    # 너무 낮아지는 경우 방지(그림자/잡티가 올라오는 것 방지)
    delta_thr = max(delta_thr, 3.0)
    a_thr = max(a_thr, 132.0)
    v_thr = max(v_thr, 55.0)
    s_thr = max(s_thr, 25.0)

    cand = np.zeros_like(skin_mask, dtype=np.uint8)
    cond = (
        skin_idx &
        (delta >= delta_thr) &   # 주변보다 더 붉은가?
        (a >= a_thr) &           # 절대적으로도 붉은가?
        (V >= v_thr) &           # 너무 어두운가? (어두우면 제거)
        (S >= s_thr)             # 너무 회색/무채색인가? (그림자 제거)
    )
    cand[cond] = 255
    return cand


# ---------------------------------------
# Mask smooth (Blur+Otsu 제거: 붙어/번짐 방지)
# ---------------------------------------
def smooth_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    m = remove_small_components(m, SMALL_COMP_MIN)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    return m


# ---------------------------------------
# Watershed Separation (큰 덩어리만 분리)
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
        markers = markers.astype(np.int32)  # 중요

        markers = cv2.watershed(roi_img, markers)
        roi_sep = np.zeros_like(roi_mask)

        for label in np.unique(markers):
            if label <= 1:
                continue
            roi_sep[markers == label] = 255

        separated[y:y+h, x:x+w] |= roi_sep

    return separated


# ---------------------------------------
# Final Contour Filtering (어두운 컨투어 제거 + 면적 자동 스케일)
# ---------------------------------------
def filter_contours(img_bgr, mask, skin_mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, _, V = cv2.split(hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    acne_contours = []
    acne_areas = []
    skin_area = int(np.count_nonzero(skin_mask))

    min_area = max(60, int(skin_area * MIN_AREA_RATIO))
    max_area = max(8000, int(skin_area * MAX_AREA_RATIO))

    # 피부 영역 밝기 기준으로 “너무 어두운 컨투어” 컷
    skin_v = V[skin_mask > 0]
    v_cut = max(float(np.percentile(skin_v, 10)), 45.0) if skin_v.size else 45.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        cnt_mask = np.zeros_like(mask)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

        mean_v = float(V[cnt_mask > 0].mean())
        if mean_v < v_cut:
            continue  # 어두운 잡티/그림자 제거

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
    skin = get_skin_mask_ycrcb(img_bgr)
    red = get_red_candidate_mask_hsv(img_bgr, skin)

    smooth = smooth_mask(red)
    separated = separate_touching_spots_by_watershed(img_bgr, smooth)

    acne_contours, total_acne, total_skin, ratio = filter_contours(img_bgr, separated, skin)
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
