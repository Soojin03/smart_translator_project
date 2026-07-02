"""PyTorch 기반 문자 단위 번역 모델을 정의하는 파일입니다. (GPT 스타일 Decoder-only Transformer 버전)

기존 Encoder-Decoder 구조 대신, source 문장과 target 문장을 하나의 시퀀스로 이어붙여
(예: [source 글자들] [SEP] [target 글자들]) 하나의 Transformer 스택에 넣고
causal self-attention만으로 다음 글자를 예측하는 GPT 스타일 구조입니다.
Cross-Attention은 존재하지 않고, Self-Attention만 사용합니다.
"""

import math

import torch
import torch.nn as nn

from src.config import PAD_TOKEN


class PositionalEncoding(nn.Module):
    """Transformer는 순서 개념이 없기 때문에, 각 위치 정보를 벡터에 더해주는 계층입니다."""

    def __init__(self, embed_size, max_len=1024):
        super().__init__()
        # 위치(0~max_len)와 차원(0~embed_size)에 따라 미리 계산해두는 위치 인코딩 테이블입니다.
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size)
        )
        pe = torch.zeros(max_len, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 학습되는 파라미터가 아니므로 buffer로 등록합니다 (state_dict에는 포함, 학습 X).
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: [배치, 시퀀스 길이, 임베딩 차원]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class GPTStyleTranslator(nn.Module):
    """source와 target을 하나로 이어붙인 시퀀스를 입력받아, 다음 글자를 예측하는 Decoder-only Transformer입니다.

    Encoder-Decoder 구조(Cross-Attention)가 없고, Self-Attention 스택 하나만 존재합니다.
    입력 예시: [source 글자 인덱스들] + [SEP 토큰] + [target 글자 인덱스들]
    """

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        # 문자 인덱스를 밀집 벡터로 변환하는 임베딩 계층입니다. (source/target 공용 하나의 임베딩)
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # 위치 정보를 더해주는 포지셔널 인코딩입니다.
        self.pos_encoding = PositionalEncoding(embed_size)

        # GPT처럼 Self-Attention만 반복하는 블록입니다.
        # nn.TransformerEncoderLayer를 사용하지만, forward에서 causal mask를 넣어
        # 실질적으로는 "미래를 못 보는 self-attention 블록"(=GPT의 Decoder block)으로 동작시킵니다.
        block = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=num_layers)

        # 마지막 은닉 벡터를 문자별 점수(logits)로 변환하는 선형 계층입니다.
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, input_ids):
        # input_ids: [배치, 시퀀스 길이] — source와 target이 이미 하나로 이어붙여진 상태로 들어옵니다.
        seq_len = input_ids.size(1)
        device = input_ids.device

        # 각 위치가 자기 자신과 그 이전 위치만 볼 수 있도록 막는 causal(삼각형) 마스크입니다.
        # 이 마스크 덕분에 Encoder-Decoder 없이도 GPT처럼 다음 글자 예측이 가능합니다.
        # padding_mask(bool)와 타입을 맞추기 위해 float가 아닌 bool로 직접 생성합니다.
        # (대각선 위쪽, 즉 미래 위치가 True = attention에서 가려짐)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )

        # 패딩 위치를 True로 표시하는 마스크입니다 (attention에서 무시하기 위함).
        # PAD_TOKEN은 "<PAD>" 같은 문자열이므로 정수 텐서와 비교할 수 없습니다.
        # 실제 패딩 인덱스는 nn.Embedding(padding_idx=0)과 동일하게 0번입니다.
        padding_mask = input_ids == 0

        # 정수 인덱스 시퀀스를 임베딩 벡터로 변환합니다.
        embedded = self.embedding(input_ids) * math.sqrt(self.embed_size)
        # 위치 정보를 더합니다.
        embedded = self.pos_encoding(embedded)

        # Self-Attention 블록을 causal mask와 함께 통과시킵니다. (Cross-Attention 없음)
        hidden = self.blocks(embedded, mask=causal_mask, src_key_padding_mask=padding_mask)

        # 각 시점의 은닉 벡터를 문자별 점수(logits)로 변환합니다.
        logits = self.fc(hidden)
        return logits