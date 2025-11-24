# app.py

import streamlit as st
import cv2
import numpy as np

from face_pipeline import analyze_face_regions

st.set_page_config(page_title="여드름 탐지기 (부위별)", layout="wide")
st.title("얼굴 부위 자동 분리 + 여드름 탐지기")
st.write("얼굴 이미지를 업로드하면, 이마/볼/턱 등을 자동으로 잘라서 여드름을 분석합니다.")

uploaded = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded:
    # 파일 읽기
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)

    if img_bgr is None:
        st.error("이미지를 읽을 수 없습니다.")
    else:
        st.subheader("원본 이미지")
        st.image(img_bgr[:, :, ::-1], channels="RGB")

        st.markdown("---")
        st.subheader("부위별 분석 결과")

        results = analyze_face_regions(img_bgr)

        if not results:
            st.error("얼굴을 찾지 못했어요. 정면 또는 측면 얼굴이 잘 보이도록 다시 찍어주세요.")
        else:
            for region_name, res in results.items():
                st.markdown(f"### 🔹 {region_name}")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("📌 Crop된 영역")
                    st.image(res["crop"][:, :, ::-1], channels="RGB")

                with col2:
                    st.write("🔥 여드름 탐지 Overlay")
                    st.image(res["overlay"][:, :, ::-1], channels="RGB")
                    st.write(f"➡️ 탐지된 여드름 개수: **{res['count']} 개**")
                    st.write(f"➡️ 여드름 면적 비율: **{res['ratio']:.5f}**")

else:
    st.info("좌측 또는 위쪽의 '파일 업로드' 버튼을 눌러 이미지를 업로드하세요.")
