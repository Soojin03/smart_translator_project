"""번역 데이터 로딩, 문자 사전 생성, 문장 인코딩 기능을 제공하는 파일입니다. (GPT 스타일 버전)"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from src.config import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SEP_TOKEN

# labels에서 loss를 계산하지 않을 위치를 표시하는 값입니다.
# PyTorch CrossEntropyLoss의 ignore_index 기본 관례를 그대로 따릅니다.
IGNORE_INDEX = -100


def normalize_text(text: str) -> str:
    """입력 문장을 모델이 처리하기 쉬운 형태로 정리합니다."""
    # None이나 결측값이 들어오는 경우를 방지하기 위해 문자열로 변환합니다.
    text = str(text)
    # 앞뒤 공백을 제거하고, 영어 대문자는 소문자로 통일합니다.
    # 한글에는 lower()가 영향을 거의 주지 않으므로 한영 공통으로 사용할 수 있습니다.
    text = text.strip().lower()
    # 여러 개의 공백이 있을 경우 하나의 공백으로 합칩니다.
    text = " ".join(text.split())
    # 정리된 문자열을 반환합니다.
    return text


def load_translation_pairs(csv_path):
    """CSV 파일에서 영어-한국어 번역 쌍을 읽고 양방향 학습 데이터로 확장합니다."""
    # CSV 파일을 pandas DataFrame으로 읽습니다.
    df = pd.read_csv(csv_path)
    # 영어와 한국어 컬럼이 모두 존재하는지 확인합니다.
    required_columns = {"en", "ko"}
    # 필요한 컬럼이 없으면 명확한 오류 메시지를 발생시킵니다.
    if not required_columns.issubset(set(df.columns)):
        raise ValueError("CSV 파일에는 en, ko 컬럼이 반드시 있어야 합니다.")
    # 결측값이 있는 행은 번역 학습에 사용할 수 없으므로 제거합니다.
    df = df.dropna(subset=["en", "ko"])
    # 각 문장을 정리합니다.
    df["en"] = df["en"].map(normalize_text)
    df["ko"] = df["ko"].map(normalize_text)

    # 하나의 모델이 영어->한국어, 한국어->영어를 모두 학습하도록 방향 토큰을 붙입니다.
    pairs = []
    # CSV의 각 행을 순회하며 양방향 데이터를 구성합니다.
    for _, row in df.iterrows():
        # 영어를 한국어로 번역하는 학습 예시입니다.
        pairs.append(("<EN2KO> " + row["en"], row["ko"]))
        # 한국어를 영어로 번역하는 학습 예시입니다.
        pairs.append(("<KO2EN> " + row["ko"], row["en"]))
    # 전체 학습 쌍을 반환합니다.
    return pairs


def build_vocab(pairs):
    """학습 데이터에 등장하는 모든 문자를 기반으로 문자 사전을 생성합니다."""
    # 특수 토큰을 가장 앞에 배치하여 고정된 인덱스를 갖도록 합니다.
    # SEP_TOKEN은 source와 target을 하나의 시퀀스로 이어붙일 때 경계를 표시합니다.
    tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SEP_TOKEN]
    # 모든 입력 문장과 출력 문장에서 문자 단위 집합을 수집합니다.
    charset = set()
    # 학습 쌍을 반복하면서 입력과 출력의 문자를 모두 모읍니다.
    for source, target in pairs:
        # 입력 문장의 각 문자를 집합에 추가합니다.
        charset.update(list(source))
        # 출력 문장의 각 문자를 집합에 추가합니다.
        charset.update(list(target))
    # 재현 가능한 사전을 위해 문자 집합을 정렬합니다.
    sorted_chars = sorted(charset)
    # 특수 토큰 뒤에 실제 문자를 붙여 전체 토큰 목록을 만듭니다.
    vocab = tokens + sorted_chars
    # 문자 또는 특수 토큰을 정수 인덱스로 바꾸는 딕셔너리입니다.
    char2idx = {token: idx for idx, token in enumerate(vocab)}
    # 정수 인덱스를 문자 또는 특수 토큰으로 되돌리는 딕셔너리입니다.
    idx2char = {idx: token for token, idx in char2idx.items()}
    # 두 사전을 반환합니다.
    return char2idx, idx2char


def encode_text(text, char2idx, add_eos=False):
    """문자열을 정수 인덱스 리스트로 변환합니다."""
    # 사전에 없는 문자는 UNK 인덱스로 변환합니다.
    ids = [char2idx.get(ch, char2idx[UNK_TOKEN]) for ch in text]
    # 필요한 경우에만 EOS 토큰을 끝에 추가합니다.
    if add_eos:
        ids.append(char2idx[EOS_TOKEN])
    # 정수 리스트를 반환합니다.
    return ids


class TranslationDataset(Dataset):
    """GPT 스타일 Decoder-only 모델을 위한 번역 데이터셋 클래스입니다.

    source와 target을 [source] + [SEP] + [target] + [EOS] 형태로 하나의 시퀀스로 이어붙인 뒤,
    input_ids는 마지막 글자를 뺀 부분, labels는 첫 글자를 뺀 부분으로 한 칸씩 밀어서(next-token
    prediction) 만듭니다. source+SEP 구간은 예측 대상이 아니므로 IGNORE_INDEX로 마스킹합니다.
    """

    def __init__(self, pairs, char2idx):
        # 원본 문장 쌍을 저장합니다.
        self.pairs = pairs
        # 문자 사전을 저장합니다.
        self.char2idx = char2idx

    def __len__(self):
        # 전체 데이터 개수를 반환합니다.
        return len(self.pairs)

    def __getitem__(self, index):
        # index 위치의 입력 문장과 정답 문장을 가져옵니다.
        source, target = self.pairs[index]

        # source와 target을 각각 문자 인덱스로 변환합니다 (EOS는 마지막에 한 번만 붙일 것이므로 여기선 생략).
        source_ids = encode_text(source, self.char2idx, add_eos=False)
        target_ids = encode_text(target, self.char2idx, add_eos=False)

        # [source] + [SEP] + [target] + [EOS] 형태로 하나의 시퀀스를 만듭니다.
        full_ids = source_ids + [self.char2idx[SEP_TOKEN]] + target_ids + [self.char2idx[EOS_TOKEN]]

        # 모델 입력은 마지막 글자를 뺀 부분입니다.
        input_ids = full_ids[:-1]
        # 정답(labels)은 첫 글자를 뺀 부분으로, input_ids보다 한 칸씩 밀려 있습니다.
        labels = full_ids[1:]

        # source_ids 길이만큼은 "target을 예측하기 이전 구간"이므로 loss에서 제외합니다.
        # (source_ids 뒤에 SEP 하나가 있고, labels[len(source_ids)]가 바로 target의 첫 글자를
        #  예측하는 위치이므로 정확히 len(source_ids)개만 마스킹하면 됩니다.)
        mask_len = len(source_ids)
        labels = [IGNORE_INDEX] * mask_len + labels[mask_len:]

        # 학습에 필요한 두 텐서를 반환합니다.
        return torch.tensor(input_ids), torch.tensor(labels)


def collate_batch(batch):
    """길이가 서로 다른 시퀀스들을 한 배치에서 사용할 수 있도록 길이를 맞춥니다."""
    # 배치에서 input_ids와 labels를 각각 분리합니다.
    input_ids, labels = zip(*batch)
    # input_ids는 PAD 토큰(인덱스 0)으로 패딩합니다.
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    # labels는 IGNORE_INDEX로 패딩해서, 패딩 구간이 loss 계산에 포함되지 않게 합니다.
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    # 패딩이 완료된 배치 텐서를 반환합니다.
    return input_ids, labels