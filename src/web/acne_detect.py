# acne_detect.py
import cv2
import numpy as np

# ---------------------------------------
# Skin Mask - YCrCb
# ---------------------------------------
def get_skin_mask_ycrcb(img_bgr):
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 140, 95], dtype=np.uint8)
    upper = np.array([255, 165, 120], dtype=np.uint8)

    mask = cv2.inRange(ycrcb, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ---------------------------------------
# Red Mask - HSV
# ---------------------------------------
def get_red_candidate_mask_hsv(img_bgr, skin_mask):

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 120, 80], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)

    lower2 = np.array([170, 120, 80], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    red_mask = cv2.bitwise_or(mask1, mask2)
    red_mask = cv2.bitwise_and(red_mask, skin_mask)

    return red_mask


# ---------------------------------------
# Mask smooth
# ---------------------------------------
def smooth_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    blur = cv2.GaussianBlur(m, (3, 3), 0)
    _, out = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


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
        if area < 600:
            cv2.drawContours(separated, [cnt], -1, 255, -1)
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi_mask = mask_bin[y:y+h, x:x+w]
        roi_img = img_bgr[y:y+h, x:x+w]

        dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.55 * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        sure_bg = cv2.dilate(roi_mask, np.ones((3,3), np.uint8))
        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

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

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    _, a, _ = cv2.split(lab)

    a_blur = cv2.GaussianBlur(a, (5, 5), 0)
    grad_x = cv2.Sobel(a_blur, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(a_blur, cv2.CV_32F, 0, 1)
    grad = cv2.magnitude(grad_x, grad_y)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    acne_contours = []
    acne_areas = []
    skin_area = np.count_nonzero(skin_mask)

    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area < 5 or area > 8000:
            continue

        peri = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

        cnt_mask = np.zeros_like(mask)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

        mean_grad = float(grad[cnt_mask > 0].mean())
        if mean_grad < 1.5:
            continue

        acne_contours.append(cnt)
        acne_areas.append(area)

    total_acne = float(np.sum(acne_areas))
    ratio = total_acne / skin_area if skin_area > 0 else 0

    return acne_contours, total_acne, skin_area, ratio


# ---------------------------------------
# Overlay Visual
# ---------------------------------------
def overlay_acne(img_bgr, contours):
    overlay = img_bgr.copy()
    cv2.drawContours(overlay, contours, -1, (0,0,255), -1)

    result = cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0)
    cv2.drawContours(result, contours, -1, (0,255,255), 1)
    return result


# ---------------------------------------
# MAIN PIPELINE
# ---------------------------------------
def detect_acne_pipeline(img_bgr, debug=False):

    skin = get_skin_mask_ycrcb(img_bgr)
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
