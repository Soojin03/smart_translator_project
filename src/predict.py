"""학습된 모델을 불러와 영어↔한국어 번역을 수행하는 파일입니다. (GPT 스타일 Decoder-only 버전)"""

import re
import torch
from src.config import MODEL_PATH, META_PATH, EMBED_SIZE, HIDDEN_SIZE, MAX_OUTPUT_LEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SEP_TOKEN, PAD_TOKEN, DATA_PATH
from src.data_utils import normalize_text, encode_text
from src.model import GPTStyleTranslator


def detect_language(text: str) -> str:
    """입력 문장에 한글이 포함되어 있으면 ko, 그렇지 않으면 en으로 판단합니다."""
    # 정규표현식으로 한글 음절 범위가 포함되어 있는지 검사합니다.
    if re.search(r"[가-힣]", text):
        # 한글이 하나라도 있으면 한국어 문장으로 판단합니다.
        return "ko"
    # 한글이 없으면 영어 문장으로 판단합니다.
    return "en"


def build_directional_source(text: str, source_lang: str) -> str:
    """입력 문장 앞에 번역 방향 토큰을 붙입니다."""
    # 영어 입력이면 한국어로 번역하라는 방향 토큰을 붙입니다.
    if source_lang == "en":
        return "<EN2KO> " + normalize_text(text)
    # 한국어 입력이면 영어로 번역하라는 방향 토큰을 붙입니다.
    return "<KO2EN> " + normalize_text(text)


def load_model():
    """저장된 모델 가중치와 문자 사전을 불러옵니다."""
    # 모델 메타 파일이나 가중치 파일이 없으면 학습을 먼저 실행해야 합니다.
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("학습된 모델 파일이 없습니다. 먼저 python -m src.train 명령을 실행하세요.")
    # CPU 환경에서도 안전하게 불러오기 위해 map_location을 CPU로 지정합니다.
    meta = torch.load(META_PATH, map_location="cpu")
    # 저장된 문자→정수 사전을 가져옵니다.
    char2idx = meta["char2idx"]
    # 저장된 정수→문자 사전을 가져옵니다.
    idx2char = meta["idx2char"]
    # 저장된 사전 크기에 맞춰 모델 객체를 생성합니다.
    model = GPTStyleTranslator(len(char2idx), meta.get("embed_size", EMBED_SIZE), meta.get("hidden_size", HIDDEN_SIZE))
    # 학습된 가중치를 모델에 주입합니다.
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    # 추론에서는 dropout이나 batchnorm이 학습 모드로 동작하지 않도록 평가 모드로 전환합니다.
    model.eval()
    # 추론에 필요한 모델과 사전을 반환합니다.
    return model, char2idx, idx2char


def load_exact_dictionary():
    """학습 데이터에 있는 문장은 정확한 번역을 우선 반환하기 위해 딕셔너리로 읽습니다."""
    # pandas 의존을 줄이기 위해 csv 모듈을 사용합니다.
    import csv
    # 정확 매칭 번역을 저장할 딕셔너리를 생성합니다.
    mapping = {}
    # CSV 파일을 UTF-8 인코딩으로 엽니다.
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        # DictReader는 첫 줄의 en, ko 컬럼명을 기준으로 행을 딕셔너리로 읽습니다.
        reader = csv.DictReader(f)
        # 각 번역 쌍을 순회합니다.
        for row in reader:
            # 영어 문장을 정리합니다.
            en = normalize_text(row["en"])
            # 한국어 문장을 정리합니다.
            ko = normalize_text(row["ko"])
            # 영어 입력에 대한 한국어 번역을 등록합니다.
            mapping[("en", en)] = ko
            # 한국어 입력에 대한 영어 번역을 등록합니다.
            mapping[("ko", ko)] = en
    # 정확 매칭 딕셔너리를 반환합니다.
    return mapping


# 생성된 결과 문자열에 포함시키지 않을 특수 토큰 집합입니다.
_SPECIAL_TOKENS = {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SEP_TOKEN}


def translate(text: str, model=None, char2idx=None, idx2char=None) -> str:
    """입력 문장을 자동으로 방향 판별하여 번역합니다."""
    # 빈 문장은 번역할 수 없으므로 안내 문구를 반환합니다.
    if not text or not text.strip():
        return "번역할 문장을 입력하세요."
    # 입력 언어를 자동으로 판단합니다.
    source_lang = detect_language(text)
    # 정확히 학습 데이터에 있는 문장은 안정적인 출력을 위해 사전 번역을 우선 사용합니다.
    exact_dictionary = load_exact_dictionary()
    # 정리된 입력 문장을 기준으로 정확 매칭을 시도합니다.
    exact_key = (source_lang, normalize_text(text))
    # 정확 매칭 결과가 있으면 바로 반환합니다.
    if exact_key in exact_dictionary:
        return exact_dictionary[exact_key]
    # 모델 객체가 전달되지 않았다면 저장된 모델을 불러옵니다.
    if model is None or char2idx is None or idx2char is None:
        model, char2idx, idx2char = load_model()
    # 번역 방향 토큰을 포함한 인코더 입력 문자열을 만듭니다.
    source_text = build_directional_source(text, source_lang)
    # source 문장을 정수 인덱스로 변환합니다 (학습 때와 동일하게 EOS는 붙이지 않습니다).
    source_ids = encode_text(source_text, char2idx, add_eos=False)
    # source 뒤에 SEP 토큰을 붙여, 여기서부터 번역 문장을 생성하라는 경계를 표시합니다.
    generated_ids = source_ids + [char2idx[SEP_TOKEN]]
    # 생성된 문자를 저장할 리스트입니다.
    result_chars = []
    # 기울기 계산을 끄면 추론 속도가 빨라지고 메모리 사용량이 줄어듭니다.
    with torch.no_grad():
        # 최대 출력 길이만큼 한 글자씩 생성합니다.
        for _ in range(MAX_OUTPUT_LEN):
            # Encoder-Decoder 구조가 아니므로, 지금까지 생성된 전체 시퀀스를 매번 통째로 넣습니다.
            input_tensor = torch.tensor([generated_ids], dtype=torch.long)
            # 전체 시퀀스에 대한 다음 글자 점수를 계산합니다.
            logits = model(input_tensor)
            # 마지막 위치의 점수만 사용해 다음 글자를 예측합니다.
            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            # 선택된 인덱스를 문자로 변환합니다.
            next_char = idx2char.get(next_id, UNK_TOKEN)
            # EOS가 나오면 문장 생성이 끝났다는 의미이므로 반복을 중단합니다.
            if next_char == EOS_TOKEN:
                break
            # 특수 토큰은 화면에 출력하지 않습니다.
            if next_char not in _SPECIAL_TOKENS:
                result_chars.append(next_char)
            # 방금 예측한 글자를 시퀀스 끝에 이어붙여 다음 스텝 입력으로 사용합니다.
            generated_ids.append(next_id)
    # 생성된 문자들을 하나의 문자열로 합칩니다.
    result = "".join(result_chars).strip()
    # 모델이 아무 문자도 생성하지 못한 경우 안내 문구를 반환합니다.
    if not result:
        return "번역 결과를 생성하지 못했습니다. 학습 데이터를 늘리거나 epoch를 증가시켜 주세요."
    # 최종 번역 결과를 반환합니다.
    return result