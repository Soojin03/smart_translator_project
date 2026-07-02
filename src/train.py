"""번역 모델을 학습하고 모델 파일을 저장하는 실행 파일입니다. (GPT 스타일 Decoder-only 버전)"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.config import DATA_PATH, MODEL_PATH, META_PATH, EMBED_SIZE, HIDDEN_SIZE, EPOCHS, BATCH_SIZE, LEARNING_RATE
from src.data_utils import load_translation_pairs, build_vocab, TranslationDataset, collate_batch, IGNORE_INDEX
from src.model import GPTStyleTranslator


def train_model(epochs=EPOCHS):
    """CSV 번역 데이터를 사용하여 GPT 스타일 Decoder-only 번역 모델을 학습합니다."""
    # CUDA GPU를 사용할 수 있으면 GPU를 사용하고, 없으면 CPU를 사용합니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CSV 파일에서 양방향 번역 학습 쌍을 읽습니다.
    pairs = load_translation_pairs(DATA_PATH)
    # 학습 데이터에 등장하는 문자 기반 사전을 생성합니다.
    char2idx, idx2char = build_vocab(pairs)
    # 문자 사전 크기를 계산합니다.
    vocab_size = len(char2idx)
    # PyTorch Dataset 객체를 생성합니다.
    dataset = TranslationDataset(pairs, char2idx)
    # DataLoader는 데이터를 배치 단위로 섞어서 모델에 공급합니다.
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    # GPT 스타일 Decoder-only 번역 모델을 생성하고 연산 장치로 이동합니다.
    model = GPTStyleTranslator(vocab_size, EMBED_SIZE, HIDDEN_SIZE).to(device)
    # 패딩 및 source 구간은 IGNORE_INDEX로 마스킹되어 있으므로 손실 계산에서 제외합니다.
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    # Adam 옵티마이저는 학습률을 자동 보정하며 안정적으로 학습되는 편입니다.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 지정한 epoch 수만큼 반복 학습합니다.
    for epoch in range(1, epochs + 1):
        # 모델을 학습 모드로 전환합니다.
        model.train()
        # epoch별 손실 합계를 저장할 변수를 초기화합니다.
        total_loss = 0.0
        # DataLoader에서 미니배치를 하나씩 가져옵니다.
        for input_ids, labels in loader:
            # 입력 텐서를 연산 장치로 이동합니다.
            input_ids = input_ids.to(device)
            # 정답 텐서를 연산 장치로 이동합니다.
            labels = labels.to(device)
            # 이전 배치에서 계산된 기울기를 초기화합니다.
            optimizer.zero_grad()
            # 모델이 각 위치의 다음 문자를 예측하도록 순전파를 실행합니다 (causal self-attention).
            logits = model(input_ids)
            # CrossEntropyLoss는 [배치*시간, 클래스수] 형태의 입력을 기대하므로 형태를 변경합니다.
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            # 손실값을 기준으로 역전파를 수행하여 기울기를 계산합니다.
            loss.backward()
            # 기울기 폭주를 방지하기 위해 기울기 크기를 제한합니다.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 옵티마이저가 모델 가중치를 업데이트합니다.
            optimizer.step()
            # 현재 배치 손실을 누적합니다.
            total_loss += loss.item()
        # 20 epoch마다 학습 손실을 출력하여 학습 상황을 확인합니다.
        if epoch == 1 or epoch % 20 == 0:
            print(f"Epoch {epoch:03d}/{epochs} | loss={total_loss / len(loader):.4f}")
    # 모델 저장 폴더가 없으면 생성합니다.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 학습된 모델 가중치를 저장합니다.
    torch.save(model.state_dict(), MODEL_PATH)
    # 추론에 필요한 문자 사전과 하이퍼파라미터를 저장합니다.
    torch.save({"char2idx": char2idx, "idx2char": idx2char, "embed_size": EMBED_SIZE, "hidden_size": HIDDEN_SIZE}, META_PATH)
    # 저장 완료 메시지를 출력합니다.
    print(f"모델 저장 완료: {MODEL_PATH}")
    # 학습된 모델 객체와 메타 정보를 반환합니다.
    return model, char2idx, idx2char


if __name__ == "__main__":
    # 이 파일을 직접 실행할 때 모델 학습을 시작합니다.
    train_model()