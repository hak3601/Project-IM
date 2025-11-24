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
    return: (pts, mode)  또는 (None, None)
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


def analyze_face_regions(img_bgr):
    """
    전체 얼굴 이미지 -> (mode에 따라) crop -> 각 영역에 대해 여드름 분석.
    return:
        {
          "forehead": {
              "count": int,
              "ratio": float,
              "overlay": np.ndarray(BGR),
              "crop": np.ndarray(BGR)
          },
          "left_cheek": {...},
          ...
        }
    얼굴 못 찾으면 {} 리턴
    """
    pts, mode = detect_landmarks_and_mode(img_bgr)
    if pts is None:
        return {}

    if mode == "full":
        regions = crop_full(img_bgr, pts)
    else:
        regions = crop_side(img_bgr, pts)

    results = {}
    for name, patch in regions.items():
        if patch.size == 0:
            continue
        overlay, score, _ = detect_acne_pipeline(patch, debug=False)
        results[name] = {
            "count": score["num_spots"],
            "ratio": score["area_ratio"],
            "overlay": overlay,
            "crop": patch,
        }
    return results
