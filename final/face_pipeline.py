# face_pipeline.py

import cv2
import numpy as np
import mediapipe as mp

from crop_regions import (
    crop_full,
    crop_side,
    landmarks_to_xy,
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    FULL_FACE_EYE_RATIO_THR,
)
from acne_detect import detect_acne_pipeline


def detect_landmarks_and_mode(img_bgr):
    """
    Mediapipe FaceMesh로 랜드마크 추출 + 정면/측면 모드 자동 판별.
    return: (pts, mode) 또는 (None, None)
    """
    h, w = img_bgr.shape[:2]

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            return None, None

        pts = landmarks_to_xy(res.multi_face_landmarks[0].landmark, w, h)

        # 눈 사이 거리 기반 full / side 판별
        xL = pts[LEFT_EYE_OUTER, 0]
        xR = pts[RIGHT_EYE_OUTER, 0]
        eye_dist = abs(xR - xL)
        ratio = eye_dist / float(w)

        mode = "full" if ratio >= FULL_FACE_EYE_RATIO_THR else "side"
        return pts, mode


def analyze_face_regions(img_bgr, filename=None):
    """
    전체 얼굴 이미지 -> crop -> 여드름 분석

    filename이 "s_" 또는 "f_"로 시작하면 모드를 강제 적용한다.
    그렇지 않으면 자동 모드(눈 간 거리 기반)를 사용한다.
    """

    # ------- 1) 자동 모드 탐지 -------
    pts, auto_mode = detect_landmarks_and_mode(img_bgr)
    if pts is None:
        return {}

    # ------- 2) 파일명 기반 강제 모드 -------
    forced_mode = None
    if filename is not None:
        fname = filename.lower()
        if fname.startswith("s_"):     # 예: s_image1.jpg
            forced_mode = "side"
        elif fname.startswith("f_"):   # 예: f_front.png
            forced_mode = "full"

    # 최종 모드 결정
    mode = forced_mode if forced_mode is not None else auto_mode

    # ------- 3) crop -------
    regions = crop_full(img_bgr, pts) if mode == "full" else crop_side(img_bgr, pts)

    # ------- 4) 각 영역 별 여드름 분석 (debug 포함) -------
    results = {}
    for name, patch in regions.items():
        if patch.size == 0:
            continue

        overlay, score, debug_maps = detect_acne_pipeline(patch, debug=True)

        results[name] = {
            "count": score["num_spots"],
            "ratio": score["area_ratio"],
            "overlay": overlay,
            "crop": patch,
            "debug": debug_maps,   # 팀원 debug 기능 유지
        }

    return results
