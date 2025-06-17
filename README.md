# 🏌 골프 스윙 이벤트 탐지 시스템 (SwingNet 기반 개선 모델)

본 프로젝트는 [McNally et al. GolfDB](https://github.com/wmcnally/golfdb)를 기반으로 하여, 경량 하이브리드 모델인 **SwingNet**을 개선한 **골프 스윙 이벤트 탐지 시스템**입니다.

저희는 기존 SwingNet의 구조에 최신 딥러닝 기법을 반영하여 모델의 **정확도**, **일반화 성능**, **학습 안정성**을 향상시키고, **데이터 불균형** 및 **자원 제약 환경**에서의 문제를 해결하고자 다음과 같은 주요 개선을 시도했습니다. 추가적으로 HPE를 적용하는 객관적 지표를 추가하여 코칭 등 2차 활용에 용이하도록 구현했습니다.

### 주요 기능
- 골프 스윙 8가지 동작 분류
- HPE를 적용하여 활용 지표 제공

### 핵심 개선 사항
- 최신 백본(MobileNetV4, EfficientNet, ConvLSTM)으로 성능 향상
- 학습시 Flip, Rotate 증강 기법을 적용하여 데이터 편향 및 과적합 완화
- Warm-Up과 learning rate scheduler으로 안정적이고 효율적인 학습 유도
- Gradient Accumulation으로 자원 제약 환경에서도 큰 배치 효과 확보
- Accuracy, Precision, Recall, F1-Score 성능지표 반영


> 위와 같은 개선을 통해, 스윙 분석 서비스의 기반이 될 수 있는 시퀀스 분류 중심의 골프 스윙 이벤트 탐지 시스템을 구축하였습니다.


---

## 데이터 분할 & 전처리

- generate_splits.py를 실행하여 golfdb.mat 데이터세트 파일을 데이터프레임으로 변환하고 train과 validation 두 개의 pkl 파일로 분할됩니다. 

## 학습
- 160x160 프레임 크기의 전처리된 비디오 클립을 제공하였습니다. 여기에서 데이터를 다운받아 디렉터리에 ‘videos_160’을 추가하세요. 직접 영상을 수집하여 사용하려면 Youtube 비디오를 다운로드하고 직접 전처리(data/preprocess_videos.py)해야 합니다. 
- 데이터가 준비되면  train.py을 실행하세요.

## 예측 및 테스트
- 위 단계에 따라 직접 모델을 학습시키거나 여기에서 사전 학습된 가중치를 다운로드하세요. ‘models’ 디렉터리가 아직 생성되지 않았다면 생성하고, ‘best_conv_lr.pth’ 파일을 이 디렉터리에 넣으세요.
- predict.py를 실행합니다. 
