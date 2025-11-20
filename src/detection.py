
# source .venv/bin/activate
# .venv/bin/python acne_detector.py test1.jpg --out result1.png --debug

import cv2
import numpy as np


def get_skin_mask_ycrcb(img_bgr):  # 스킨 마스크 (어디까지가 피부영역인지 판별)
    """
    BGR -> YCrCb로 바꾸고, 피부 색 범위로 마스크 생성
    마스크 범위 예시 -> Y: 0~255, Cr: 133~173, Cb: 77~127
    (볼/턱/이마 bbox 이미지 기준으로 필요하면 수동 튜닝)
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    # 피부색 범위 (필요하면 나중에 조정)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)

    mask = cv2.inRange(ycrcb, lower, upper)

    # 모폴로지 연산 -> (노이즈 제거)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def get_red_candidate_mask_hsv(img_bgr, skin_mask):  # 빨간색 후보 마스크 (피부 마스크 안에서만)
    """
    BGR -> HSV로 바꾸고, 피부 영역 안에서 붉은 계열(Hue)만 추출.
    OpenCV HSV: H ∈ [0,179], 붉은색은 양 끝(0 근처, 170~179)에 모임.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 1구간: H 0~10 근처
    lower_red1 = np.array([0, 50, 50], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

    # 2구간: H 170~179 근처
    lower_red2 = np.array([170, 50, 50], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(mask1, mask2)

    # 피부 마스크랑 AND → 피부영역에 있는 붉은 영역만 추출
    red_mask = cv2.bitwise_and(red_mask, skin_mask)

    return red_mask


def smooth_mask(mask):  # 노이즈 제거 + 경계 매끄럽게
    """
    모폴로지 + Gaussian Blur + Otsu 바이너리 마스크 생성
    (여기에서 너무 세게 closing 하면 서로 다른 여드름이 하나로 붙으므로 적당히)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # 작은 점 제거
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # 살짝 closing
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 블러 후 다시 이진화해서 경계를 조금 매끄럽게
    m_blur = cv2.GaussianBlur(m, (3, 3), 0)
    _, m_bin = cv2.threshold(
        m_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return m_bin


# smooth_mask처럼 붙어 있는 붉은 영역 -> distance transform + watershed로 여러 개의 작은 blob으로 분리
def separate_touching_spots_by_watershed(img_bgr, # 원본 이미지
                                         mask, # 0/255 이진 마스크 
                                         dist_ratio = 0.45, #distanceTransfrom 최대값 대미 임계 비율 (크면 더 많이 쪼개짐)
                                         split_min_area = 800): # 이면적보다 작은 blob은 그대로두고, 얘보다 크면 watershed 진행

    mask_bin = (mask > 0).astype(np.uint8) * 255
    h, w = mask_bin.shape
    separated = np.zeros_like(mask_bin)

    # 전체 마스크에서 blob 단위로 처리
    contours, _ = cv2.findContours(
        mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return mask_bin

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # blob 작으면, watershed x 
        if area < split_min_area:
            cv2.drawContours(separated, [cnt], -1, 255, thickness=-1)
            continue

        # blob이 크면, ROI 잘라서 watershed 적용
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        roi_mask = mask_bin[y:y + h_box, x:x + w_box]
        roi_img = img_bgr[y:y + h_box, x:x + w_box]

        # 거리 변환
        dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)

        # 확실한 foreground
        _, sure_fg = cv2.threshold(dist,
                                   dist_ratio * dist.max(),
                                   255, cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg)

        # 배경
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(roi_mask, kernel, iterations=1)

        # unknown = 배경 - foreground
        unknown = cv2.subtract(sure_bg, sure_fg)

        # marker 생성
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # watershed
        roi_for_ws = roi_img.copy()
        markers = cv2.watershed(roi_for_ws, markers)

        # 라벨별로 다시 마스크 구성
        roi_sep = np.zeros_like(roi_mask)
        for label in np.unique(markers):
            if label <= 1:
                continue  # -1(경계), 1(배경) 제외
            roi_sep[markers == label] = 255

        # 전체 separated에 ROI 결과 반영
        dst_roi = separated[y:y + h_box, x:x + w_box]
        separated[y:y + h_box, x:x + w_box] = cv2.bitwise_or(dst_roi, roi_sep)

    return separated


def filter_contours_by_shape_and_a( img_bgr,
                                   mask,
                                   skin_mask,
                                   min_area=5,#최소면적
                                   max_area=8000, # 최대면적
                                   small_spot_max_area=500,
                                   min_circularity=0.2, # 원형도
                                   min_delta_a=1.0,
                                   cluster_min_delta_a=3.0,
                                   debug_print=False):
    """
    각 컨투어에 대해:
    - 면적 (min_area ~ max_area)
    - 원형도 (circularity)
    - Lab a* 내/외곽 링 대비(Δa*)
    를 이용해 여드름 후보를 선택.

    작은 점(spot) vs 큰 클러스터(cluster)에 서로 다른 기준 적용:
    - area <= small_spot_max_area  → 둥글고(circ) 일정 이상 붉으면(Δa*) 개별 여드름
    - area >  small_spot_max_area → 모양은 자유, 대신 매우 붉으면(Δa*↑) 클러스터 여드름
    """
    
    
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    _, a, _ = cv2.split(lab)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    print("Total candidate contours:", len(contours))

    acne_contours = []
    acne_areas = []

    total_skin_area = int(np.count_nonzero(skin_mask))

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 0.0
        if perimeter > 0:
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)

        # 기본 면적 필터 (너무 작은 점/너무 큰 패치 제거용)
        if area < min_area or area > max_area:
            if debug_print:
                print(f"[{i}] REJECT area range (area={area:.1f})")
            continue

        # 컨투어 마스크
        cnt_mask = np.zeros_like(mask)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=-1)

        # area 기반으로 링 두께 설정
        radius = max(1, int(np.sqrt(max(area, 1)) * 0.2))
        k = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        eroded = cv2.erode(cnt_mask, kernel, iterations=1)
        dilated = cv2.dilate(cnt_mask, kernel, iterations=1)

        inner_ring = cv2.subtract(cnt_mask, eroded)
        outer_ring = cv2.subtract(dilated, cnt_mask)
        outer_ring = cv2.bitwise_and(outer_ring, skin_mask)

        inner_idx = inner_ring > 0
        outer_idx = outer_ring > 0

        if np.any(inner_idx) and np.any(outer_idx):
            inner_a = float(a[inner_idx].mean())
            outer_a = float(a[outer_idx].mean())
            delta_a = inner_a - outer_a  # 안쪽이 더 붉으면 양수
        else:
            delta_a = 0.0

        # 디버그 출력
        if debug_print:
            print(
                f"[{i}] area={area:.1f}, circ={circularity:.3f}, Δa*={delta_a:.2f}",
                end=" "
            )

        # -------------------------
        # 크기별로 다른 기준 적용
        # -------------------------
        if area <= small_spot_max_area:
            # 개별 여드름(점 형태): 둥글고, 어느 정도 붉어야 함
            if circularity < min_circularity or delta_a < min_delta_a:
                if debug_print:
                    print("-> REJECT small-spot rule")
                continue
            if debug_print:
                print("-> ACCEPT small spot")
        else:
            # 밀집된 클러스터: 모양은 자유롭게, 대신 매우 붉어야 함
            if delta_a < cluster_min_delta_a:
                if debug_print:
                    print("-> REJECT cluster rule")
                continue
            if debug_print:
                print("-> ACCEPT cluster")

        acne_contours.append(cnt)
        acne_areas.append(area)

    total_acne_area = float(np.sum(acne_areas)) if acne_areas else 0.0
    area_ratio = (total_acne_area / total_skin_area) if total_skin_area > 0 else 0.0

    print("Selected acne contours:", len(acne_contours))
    return acne_contours, total_acne_area, total_skin_area, area_ratio


def overlay_acne(img_bgr, acne_contours, alpha=0.4):
    """
    여드름으로 판정된 컨투어를 빨간색으로 채우고,
    반투명 오버레이로 원본 위에 합성.
    """
    overlay = img_bgr.copy()
    # 내부를 빨간색으로 채움
    cv2.drawContours(overlay, acne_contours, -1, (0, 0, 255), thickness=-1)
    # 원본과 블렌딩
    result = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    # 윤곽선은 노란색으로 한번 더
    cv2.drawContours(result, acne_contours, -1, (0, 255, 255), thickness=1)
    return result


def detect_acne_pipeline(img_bgr, debug=False):
    """
    전체 파이프라인 실행:
    - 피부 마스크
    - 붉은 후보 마스크
    - 스무딩
    - watershed로 큰 덩어리 쪼개기
    - 컨투어 필터링 (small spot / cluster)
    - 오버레이 + 점수
    (입력 이미지는 볼/턱/이마처럼 잘린 bbox라고 가정)
    """
    skin_mask = get_skin_mask_ycrcb(img_bgr)
    red_mask = get_red_candidate_mask_hsv(img_bgr, skin_mask)
    smooth = smooth_mask(red_mask)

    # 스무딩된 마스크에서 원시 컨투어 개수 확인용
    raw_contours, _ = cv2.findContours(
        smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    print("Raw contours after smooth:", len(raw_contours))

    # distance transform + watershed로 밀집된 큰 blob 쪼개기
    separated = separate_touching_spots_by_watershed(
        img_bgr, smooth, dist_ratio=0.45
    )

    raw_contours_ws, _ = cv2.findContours(
        separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    print("Raw contours after watershed:", len(raw_contours_ws))

    acne_contours, total_acne_area, total_skin_area, area_ratio = \
        filter_contours_by_shape_and_a(
            img_bgr, separated, skin_mask,
            min_area=5,           # bbox 해상도에 따라 3~10 사이에서 튜닝
            max_area=8000,        # 너무 큰 패치(입술 등)는 제외
            small_spot_max_area=500,
            min_circularity=0.2,
            min_delta_a=1.0,
            cluster_min_delta_a=3.0,
            debug_print=debug     # --debug일 때만 상세 로그 출력
        )

    overlay = overlay_acne(img_bgr, acne_contours)

    score = {
        "num_spots": len(acne_contours),
        "total_acne_area": float(total_acne_area),
        "total_skin_area": int(total_skin_area),
        "area_ratio": float(area_ratio),
    }

    if debug:
        return overlay, score, {
            "skin_mask": skin_mask,
            "red_mask": red_mask,
            "smooth_mask": smooth,
            "separated_mask": separated,
        }
    else:
        return overlay, score


#################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Color+shape-based acne detection"
    )
    parser.add_argument(
        "image_path",
        help="입력 여드름/피부 bbox 이미지 경로 (예: test1.jpg)"
    )
    parser.add_argument(
        "--out", default="acne_result.png",
        help="오버레이 결과 저장 파일 이름"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="중간 마스크(skin/red/smooth/separated)도 같이 저장"
    )

    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    if img is None:
        raise SystemExit(f"이미지를 읽을 수 없습니다: {args.image_path}")

    overlay, score, debug_maps = detect_acne_pipeline(
        img, debug=args.debug
    ) if args.debug else (*detect_acne_pipeline(img, debug=False), None)

    print("Detected spots:", score["num_spots"])
    print("Total acne area:", score["total_acne_area"])
    print("Total skin area:", score["total_skin_area"])
    print("Area ratio (acne/skin):", score["area_ratio"])

    cv2.imwrite(args.out, overlay)
    print("Saved overlay to:", args.out)

    if debug_maps is not None:
        cv2.imwrite("debug_skin_mask.png", debug_maps["skin_mask"])
        cv2.imwrite("debug_red_mask.png", debug_maps["red_mask"])
        cv2.imwrite("debug_smooth_mask.png", debug_maps["smooth_mask"])
        cv2.imwrite("debug_separated_mask.png", debug_maps["separated_mask"])
        print(
            "Saved debug masks: "
            "debug_skin_mask.png, debug_red_mask.png, "
            "debug_smooth_mask.png, debug_separated_mask.png"
        )
