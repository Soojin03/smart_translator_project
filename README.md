# Torch 기반 스마트 번역기 Streamlit 프로젝트

핵심 흐름은 다음과 같습니다.

- 번역 데이터 구축
- 문자 사전 생성
- 인코더 입력, 디코더 입력, 디코더 출력 데이터 구성
- Seq2Seq RNN 모델 구현
- 손실 함수와 옵티마이저를 이용한 지도 학습
- 학습된 모델로 새 문장 번역
- 반복 학습 수, 최적화 함수, 손실 함수, RNN 은닉 차원 등을 조절하는 최적화

## 프로젝트 구조
```text
smart_translator_torch_streamlit_project/
├─ app/
│  └─ streamlit_app.py
├─ data/
│  └─ translation_pairs.csv
├─ models/
├─ src/
│  ├─ config.py
│  ├─ data_utils.py
│  ├─ model.py
│  ├─ predict.py
│  └─ train.py
├─ .gitignore
├─ README.md
└─ requirements.txt
```

## 설치
```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 모델 학습
```bash
python -m src.train
```

학습이 완료되면 다음 파일이 생성됩니다.
```text
models/smart_translator.pt
models/translator_meta.pt
```

## Streamlit 실행
```bash
streamlit run app/streamlit_app.py
```

## 사용 예시
영어 입력:
```text
hello
thank you
i am a student
what are you doing
```

한국어 입력:
```text
안녕하세요
감사합니다
나는 학생입니다
무엇을 하고 있나요
```

## 참고
이 예제는  문자 단위 Seq2Seq 모델입니다. 
실제 상용 번역기 수준의 품질을 원한다면 더 많은 병렬 말뭉치, Transformer 기반 모델, SentencePiece 토크나이저, 
BLEU/chrF 평가 등을 추가해야 합니다.
