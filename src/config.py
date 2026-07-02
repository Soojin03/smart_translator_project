"""프로젝트 전체에서 공통으로 사용하는 설정값을 관리하는 파일입니다."""

from pathlib import Path

# 현재 config.py 파일의 상위 폴더(src)의 상위 폴더를 프로젝트 루트로 지정합니다.
# 예: smart_translator_project/src/config.py -> smart_translator_project 폴더 저장 처리
BASE_DIR = Path(__file__).resolve().parent.parent

# 학습 데이터 CSV 파일 경로를 지정합니다.
DATA_PATH = BASE_DIR / "data" / "translation_pairs.csv"

# 학습된 PyTorch 모델 파일이 저장될 경로를 지정합니다.
MODEL_PATH = BASE_DIR / "models" / "smart_translator.pt"

# 문자 사전, 문장 최대 길이, 하이퍼파라미터 등 메타 정보를 함께 저장할 경로입니다.
META_PATH = BASE_DIR / "models" / "translator_meta.pt"

# 인코더와 디코더의 임베딩 차원입니다.
# 문자를 정수로 바꾼 뒤, 이 정수를 벡터 공간으로 변환할 때 사용하는 크기입니다.
EMBED_SIZE = 64

# RNN의 은닉 상태 차원입니다.
# 값이 클수록 표현력이 커질 수 있지만 학습 시간이 증가할 수 있습니다.
HIDDEN_SIZE = 128

# 학습 반복 횟수입니다.
# 강의교안의 MY_EPOCH 개념에 해당하며, 데이터 전체를 몇 번 반복 학습할지 결정합니다.
EPOCHS = 120

# 한 번에 학습할 데이터 묶음 크기입니다.
# 작은 데이터셋이므로 16 정도로 설정하여 빠르게 학습되도록 합니다.
BATCH_SIZE = 16

# 학습률입니다.
# 옵티마이저가 가중치를 얼마나 크게 수정할지 결정합니다.
LEARNING_RATE = 0.003

# 번역 결과를 생성할 때 최대 몇 글자까지 만들지 결정합니다.
MAX_OUTPUT_LEN = 60

# 특수 토큰입니다.
# PAD는 길이를 맞추기 위한 빈 칸, SOS는 디코더 시작, EOS는 문장 종료, UNK는 사전에 없는 문자입니다.
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
SEP_TOKEN = "<SEP>"
