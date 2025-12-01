# AI 음원 판별 모델

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)
![Librosa](https://img.shields.io/badge/Librosa-0.9+-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

딥러닝 기반 AI 생성 음원과 원본 음원을 구분하는 이진 분류 모델

## 🎯 Overview

이 프로젝트는 CNN 기반 딥러닝 모델을 활용하여 AI로 생성된 음원과 원본 음원을 자동으로 판별하는 시스템을 제안합니다.

- **Dataset**: 솔로 가수 팝송 음원 (원본 + AI 커버)
- **Model**: CNN (Convolutional Neural Network)
- **Feature Extraction**: MFCC (Mel-Frequency Cepstral Coefficients)
- **Performance**: 91.5% accuracy

## 📄 Research Background

### Problem Statement

AI 음원 생산 사례가 급증하고 있으나 관련 규제가 미비한 상황입니다:
- YouTube 'AI cover' 검색 시 다량의 조회수 발생
- 학습 시 음원/목소리 무단 사용
- 창작자에 대한 공정한 보상 부재
- 현행법상 목소리에 대한 저작권 침해 주장 성립 어려움

### Our Solution

음성 신호의 MFCC 특징 추출과 CNN 딥러닝 모델을 통해 AI 생성 음원을 자동으로 판별합니다.

## 🔬 Method

### Preprocessing Pipeline
```
Raw Audio Files (mp4)
    ↓
WAV 형식 변환
    ↓
Audio Segmentation (10초 단위, 5초 중첩)
    ↓
MFCC Feature Extraction
    ↓
Train/Test Split (75%/25%)
    ↓
CNN Model Training
```

### Audio Segmentation

**세션 분할 기준:**
- **interval**: 10초 단위 분할
- **step**: 5초 단위 중첩

이를 통해 데이터 증강 효과를 얻고 학습 데이터를 확보합니다.

### Feature Extraction

**MFCC (Mel-Frequency Cepstral Coefficients)** 방식 채택:
- 사람이 소리를 듣는 방식과 유사하게 분석
- 주파수 영역에서 오디오 신호를 분석해 멜 스케일로 변환
- 사람의 청각에 더 적합한 주파수 대역 강조
- 음성 인식, 음악 장르 분류 등에 주로 사용

**GFCC 대비 장점:**
- 더 안정적인 학습 곡선
- 음성 특징 추출에 최적화
- 계산 효율성 우수

## 📊 Results

### Model Performance

| Metric | Score |
|--------|-------|
| **Train Accuracy** | 99.60% |
| **Validation Accuracy** | 91.5% |
| **Epochs** | 21 (EarlyStopping) |

### Model Architecture
```python
Model: Sequential
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
conv1d_3 (Conv1D)          (None, 99, 64)            256       
max_pooling1d_3 (MaxPooling)(None, 49, 64)           0         
conv1d_4 (Conv1D)          (None, 47, 128)           24704     
max_pooling1d_4 (MaxPooling)(None, 23, 128)          0         
conv1d_5 (Conv1D)          (None, 21, 256)           98560     
max_pooling1d_5 (MaxPooling)(None, 10, 256)          0         
flatten_1 (Flatten)        (None, 2560)              0         
dense_2 (Dense)            (None, 64)                163904    
dropout (Dropout)          (None, 64)                0         
dense_3 (Dense)            (None, 1)                 65        
=================================================================
Total params: 287,489 (1.10 MB)
Trainable params: 287,489 (1.10 MB)
Non-trainable params: 0 (0.00 Byte)
```

### Training Configuration

- **Optimizer**: Adam
- **Loss**: Binary Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 50 (stopped at 21)
- **Batch Size**: 32
- **EarlyStopping**: monitor='val_accuracy', patience=5

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.x
Librosa 0.9+
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ai-audio-detector.git
cd ai-audio-detector

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

1. 원본 음원과 AI 커버 음원을 수집
2. `data/raw/` 디렉토리에 저장
   - `data/raw/original/` - 원본 음원
   - `data/raw/ai_cover/` - AI 생성 음원

### Audio Segmentation
```python
from pydub import AudioSegment
import os

# 음원 분할
interval = 10 * 1000  # 10초
step = 5 * 1000       # 5초

audio = AudioSegment.from_wav('audio.wav')
length = len(audio)

for i in range(0, length, step):
    start_time = i
    end_time = i + interval
    split_audio = audio[start_time:end_time]
    
    start_sec = start_time // 1000
    end_sec = end_time // 1000
    filename = f"output_{start_sec}-{end_sec}.wav"
    split_audio.export(filename, format="wav")
```

### Feature Extraction & Training
```python
import librosa
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# MFCC 특징 추출
def extract_features(audio_samples, sample_rate):
    extracted_features = []
    
    for sample in audio_samples:
        zero_cross_feat = librosa.feature.zero_crossing_rate(sample).mean()
        mfccs = librosa.feature.mfcc(y=sample, sr=sample_rate, n_mfcc=100)
        mfccsscaled = np.mean(mfccs.T, axis=0)
        mfccsscaled = np.append(mfccsscaled, zero_cross_feat)
        extracted_features.append(mfccsscaled)
    
    return np.array(extracted_features)

# CNN 모델 구축
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(101,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 학습
history = model.fit(X_train, y_train, 
                   epochs=50, 
                   batch_size=32, 
                   validation_data=(X_test, y_test),
                   callbacks=[early_stop])
```

### Inference
```python
# 음원 판별
def predict_audio(audio_path):
    # 오디오 로드
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # 특징 추출
    features = extract_features([audio], sr)
    features = features.reshape(1, -1, 1)
    
    # 예측
    prediction = model.predict(features)
    
    if prediction[0] > 0.5:
        return "AI Generated"
    else:
        return "Original"

result = predict_audio('test_audio.wav')
print(f"Classification: {result}")
```

## 📁 Repository Structure
```
ai-audio-detector/
│
├── data/
│   ├── raw/                    # 원본 오디오 파일
│   │   ├── original/          # 원본 음원
│   │   └── ai_cover/          # AI 커버 음원
│   └── processed/              # 분할된 오디오 파일
│
├── src/
│   ├── preprocessing/
│   │   ├── audio_segment.py   # 오디오 세션 분할
│   │   └── wav_converter.py   # mp4 to wav 변환
│   │
│   ├── feature_extraction/
│   │   ├── mfcc_extractor.py  # MFCC 특징 추출
│   │   └── visualizer.py      # 파형 및 UMAP 시각화
│   │
│   ├── models/
│   │   ├── cnn_model.py        # CNN 모델 정의
│   │   └── dnn_model.py        # DNN 모델 (비교용)
│   │
│   ├── training/
│   │   └── train.py            # 모델 학습
│   │
│   └── inference.py            # 음원 판별
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_comparison.ipynb
│   └── 03_model_evaluation.ipynb
│
├── models/
│   └── best_model.h5           # 학습된 모델
│
├── requirements.txt
└── README.md
```

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow 2.x, Keras
- **Audio Processing**: Librosa, PyDub
- **Feature Extraction**: MFCC, Zero-Crossing Rate
- **Visualization**: Matplotlib, Seaborn, UMAP
- **Data Processing**: NumPy, Pandas

## 📈 Methodology Comparison

### Feature Extraction Methods

| Method | Description | Result |
|--------|-------------|--------|
| **MFCC** ✓ | 멜 스케일 기반 주파수 분석 | Stable, 91.5% accuracy |
| GFCC | 감마톤 필터 기반 분석 | Overfitting 발생 |

### Deep Learning Algorithms

| Algorithm | Description | Result |
|-----------|-------------|--------|
| **CNN** ✓ | 합성곱 신경망, 시계열 패턴 학습 | Best performance |
| DNN | 완전연결 신경망 | 과적합 발생 |

**CNN 선택 이유:**
- 시계열 오디오 데이터의 지역적 패턴 추출에 효과적
- MaxPooling을 통한 차원 축소로 과적합 방지
- 안정적인 학습 곡선

## 🎵 Data Collection

### Collection Criteria

- 사람의 귀로 들었을 때 AI 구분이 어려운 고난이도 음원
- 기계음이 거의 포함되지 않은 솔로 가수 팝송
- WAV 형식 (무손실 압축)

### Audio Format

**WAV vs MP4:**

| Format | Pros | Cons |
|--------|------|------|
| **WAV** ✓ | 원시 오디오 데이터 제공<br>샘플링 레이트, 뎁스 조정 용이 | 파일 크기 큼 |
| MP4 | 파일 크기 작음 | 압축으로 인한 데이터 손실<br>신호 처리에 부적합 |

## 📊 Visualization

### Waveform Comparison

파형 시각화를 통해 원본과 AI 음원의 차이를 확인할 수 있습니다. 육안으로는 구별이 어려우나 MFCC 특징 추출 후 UMAP 시각화 시 명확한 차이가 나타납니다.

### UMAP Projection

고차원 MFCC 특징을 2D로 투영하여 클러스터링 패턴을 확인합니다.

## 🚀 Future Work

### Limitations

- **데이터 한계**: 음성 관련 데이터 확보의 어려움
- **범위 제한**: 튜닝된 음원, 혼성/그룹 음원 사용 불가
- **방법론적 한계**: 시간적 제약으로 추가 실험 제한

### Future Improvements

1. **데이터 확장**
   - 다양한 장르 및 카테고리 추가
   - 한국어 음원 데이터 수집
   - 그룹 및 혼성 음원 지원

2. **모델 고도화**
   - Transformer 기반 모델 적용
   - Ensemble 모델 구축
   - Real-time 판별 시스템 개발

3. **서비스화**
   - AI 워터마크 의무화와 연계
   - 콘텐츠 게시 시 자동 판별 시스템
   - API 서비스 제공

4. **비즈니스 모델**
   - AI 음원 수익 분배 시스템
   - 저작권 보호 솔루션
   - 음원 플랫폼 연동

## 📚 References

1. [Librosa Documentation](https://librosa.org/)
2. [MFCC Tutorial](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
3. [CNN for Audio Classification](https://arxiv.org/abs/1610.00087)
4. [PyDub Documentation](https://pydub.com/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

⭐ **Key Findings**: CNN 기반 MFCC 특징 추출을 통해 91.5%의 정확도로 AI 생성 음원을 판별할 수 있으며, 이는 AI 음원 관련 저작권 및 수익 분배 문제 해결에 기여할 수 있습니다.
