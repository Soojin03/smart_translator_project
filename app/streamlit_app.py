"""Streamlit 기반 스마트 번역기 실행 파일입니다."""

import sys
from pathlib import Path

import streamlit as st

# ------------------------------------------------------
# 프로젝트 루트 경로 등록
# ------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# ------------------------------------------------------
# src 모듈 import
# ------------------------------------------------------
from src.predict import load_model, translate
from src.train import train_model


# ------------------------------------------------------
# Streamlit 기본 설정
# ------------------------------------------------------
st.set_page_config(
    page_title="Smart Translator",
    page_icon="🌐",
    layout="centered"
)

st.title("🌐 Smart Translator")
st.write("영어 ↔ 한국어 문자 단위 Seq2Seq 번역기입니다.")


# ------------------------------------------------------
# 모델 로딩 또는 학습 함수
# ------------------------------------------------------
@st.cache_resource
def cached_load_or_train_model():
    """
    학습된 모델과 문자 사전을 불러옵니다.

    중요한 점:
    - model 객체 안에는 char2idx, idx2char 속성이 없습니다.
    - char2idx, idx2char는 meta 파일에서 따로 불러와야 합니다.
    - 따라서 반드시 load_model() 또는 train_model()의 반환값을
      model, char2idx, idx2char 형태로 받아야 합니다.
    """
    try:
        # 이미 학습된 모델 파일이 있으면 모델과 사전을 불러옵니다.
        model, char2idx, idx2char = load_model()
        return model, char2idx, idx2char

    except FileNotFoundError:
        # 모델 파일이 없으면 새로 학습합니다.
        model, char2idx, idx2char = train_model()
        return model, char2idx, idx2char


# ------------------------------------------------------
# 모델, 문자 사전 로딩
# ------------------------------------------------------
model, char2idx, idx2char = cached_load_or_train_model()


# ------------------------------------------------------
# 사용자 입력 UI
# ------------------------------------------------------
input_text = st.text_area(
    "번역할 문장을 입력하세요.",
    placeholder="예: hello / 안녕하세요",
    height=120
)

# ------------------------------------------------------
# 번역 실행
# ------------------------------------------------------
if st.button("번역하기"):
    if not input_text.strip():
        st.warning("번역할 문장을 입력하세요.")
    else:
        result = translate(
            text=input_text,
            model=model,
            char2idx=char2idx,
            idx2char=idx2char
        )

        st.subheader("번역 결과")
        st.success(result)