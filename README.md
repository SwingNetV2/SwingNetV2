# 🏌️‍♂️ 골프 스윙 이벤트 탐지 시스템 (SwingNet 기반 개선 모델)

본 프로젝트는 [McNally et al.의 GolfDB](https://github.com/wmcnally/golfdb)를 기반으로 하여,  
경량 하이브리드 모델인 **SwingNet**을 개선한 **골프 스윙 이벤트 탐지 시스템**입니다.

우리는 기존 구조에 최신 딥러닝 기법을 반영하여 모델의 **정확도**, **일반화 성능**, **학습 안정성**을 향상시키고,  
**데이터 불균형** 및 **자원 제약 환경**에서의 문제를 해결하고자 다음과 같은 주요 개선을 시도했습니다.

---

## ✅ 핵심 개선 사항

### 1. Backbone Replacement

기존 MobileNetV2 기반의 CNN을 다음과 같은 최신 백본으로 교체하여 성능 향상을 시도했습니다:

- **MobileNetV4**, **EfficientNet (예: EfficientNetV2-L)**
- **ConvLSTM**: 시공간 패턴을 더 효과적으로 학습하기 위해 순환 합성곱 구조 도입

---

### 2. Data Augmentation

- `Albumentations` 라이브러리를 사용해 다양한 증강 기법 적용
- **데이터 편향 및 과적합 문제**를 완화
- 소수 클래스에서도 **일반화 성능 향상**

---

### 3. Learning Rate Scheduler &  Warm-Up 기법

- 초기 학습률을 점진적으로 증가시켜 **급격한 가중치(weight) 업데이트로 인한 불안정한 수렴 완화**
- 특히 **사전학습된 백본을 파인튜닝(fine-tuning)** 할 때 유용

---

### 4. Gradient Accumulation

- 자원 제약 환경에서도 안정적인 학습을 위해 Gradient Accumulation을 적용하여,
- 작은 미니배치로도 큰 배치 효과와 부드러운 수렴을 유도

---

## 📌 기대 효과

- 시퀀스별 골프 스윙 분석 정확도 향상  
- 소규모, 편향된 데이터셋에서도
