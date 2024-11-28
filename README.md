# 모델 성능 평가 보고서

학습 성능
- 사용된 데이터셋 수 (Train, Validation, Test)
- 전체 클래스 분포
- 적용된 하이퍼파라미터
- epochs, learning rate, confidence, batch size, input size 등
- Validation 결과

## 데이터 EDA 과정

| 전체 이미지 | OK 이미지  | NG 이미지 |
|------------|-----------|-----------|
| 5243       | 4944      | 299       |

### 정의한 결함 유형
| 결함 유형   | 이미지 |
|-------------|--------|
| 기포        | <img src="https://github.com/user-attachments/assets/661b13f3-0cbe-4023-939c-8fb89225b43b" alt="기포" height="80"> |
| 모서리 결함 | <img src="https://github.com/user-attachments/assets/d4f7c8f4-3fa2-4b5f-aa95-485b066d5735" alt="모서리"> |
| 절단면      | <img src="https://github.com/user-attachments/assets/2674136b-52f3-4de5-a352-31d2f6f80b8a" alt="절단면" height="45"> |
| 잔재        | <img src="https://github.com/user-attachments/assets/6c1ba20e-06bf-494f-a2ea-dae48a1af635" alt="잔재" height="60"> |
| 패임        | <img src="https://github.com/user-attachments/assets/1ce6f101-5a89-4768-9d49-3406ecac183b" alt="패임" height="100"> |
| 라인        | <img src="https://github.com/user-attachments/assets/081d2158-4208-4929-a633-e767682b2cca" alt="라인"> |
| 이물질      | <img src="https://github.com/user-attachments/assets/e3988f18-a058-4c70-819c-e4bbb9222cf8" alt="이물질" height="50"> |

### 데이터 변환 시도
- 색공간
<img src="https://github.com/user-attachments/assets/ce75a673-a60b-4b86-9877-2745c26f1a6c" alt="색공간" height="300">
- 대비 조정
- 이진화 및 컨투어 추출

## 데이터 셋 구축 (총 5세트)
- **2 class** (inner defect, outer defect) 640sz, 1389sz
- **7 class** (bubble, chip, cut, debris, dent, line, spot) 640sz, 1389sz
- **7 class** (**bubble**, chip, cut, debris, dent, line, spot) 640sz
하단 두 세트는 기포 라벨링 방식에 차이를 둠. 


## YOLO 모델

<img src="https://raw.githubusercontent.com/ultralytics/assets/refs/heads/main/yolo/performance-comparison.png" alt="모델 비교" width="70%">

**Latency T4 TensorRT10 FP16 (ms/img)**: T4 GPU에서 TensorRT 10 및 FP16 최적화를 적용한 모델이 한 이미지를 처리하는 데 평균적으로 걸리는 ms를 의미합니다.<br>
**T4 TensorRT10**은 NVIDIA **Tesla T4 GPU**를 사용한 딥러닝 추론 최적화 환경입니다.<br>
**Tesla T4 GPU**를 사용하는 **AWS EC2** 인스턴스는 **g4dn 시리즈** 인스턴스입니다.<br>

**제한된 데이터셋 규모**와 **실시간 추론**의 필요성을 고려할 때, 모델의 계산 효율성과 성능 간 최적의 균형점을 찾는 것이 핵심 과제였습니다. 큰 모델은 작은 데이터셋에서 오히려 과적합 위험이 높고 일반화 성능이 떨어질 수 있어, 이러한 요구사항에 가장 적합한 모델로 YOLOv11s를 채택하였습니다. 이 모델은 경량 아키텍처를 통해 빠른 처리 속도와 제한된 데이터 환경에서도 안정적인 성능을 동시에 제공합니다.

## 용어 정리

**IoU(Intersection over Union)** : 두 영역의 겹치는 영역을 두 영역의 합으로 나눈 값<br>
<img src="https://velog.velcdn.com/images%2Failab%2Fpost%2F09ce7d70-b35e-4b5d-bc3a-9367be990c99%2Fimage.png" alt="IoU" height="300">

---
**Recall(재현율)** : 실제 정답값 True 중 True라고 예측한 비율<br>
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcXkk52%2FbtrhA2AnprS%2Fq2Z7HPTQ3hLklC0YpBTio0%2Fimg.png" alt="Recall" height="80">

---
**Precision(정밀도)** : True라고 예측한 것 중 실제 True인 비율<br>
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FeOLCCS%2Fbtrhy2uohzQ%2FMnWjo9wSCidNHaL3kM4ik0%2Fimg.png" alt="Precision" height="80">

---
**P-R Curve(Precision-Recall Curve)** : Recall의 변화에 따른 Precision을 나타낸 곡선(x축이 Recall, y축이 Precision)<br>
이 곡선은 분류기의 양성 클래스 성능을 세밀하게 분석하며, True Positive(TP), False Positive(FP), False Negative(FN) 메트릭을 기반으로 계산됩니다. x축은 Recall, y축은 Precision으로 구성되어 모델의 성능을 시각적으로 표현합니다.<br>
주요 특징은 분류기의 신뢰도(Confidence) 임계값을 변경하면서 Precision과 Recall의 변화를 관찰하는 것입니다. 임계값이 낮아질수록 Recall은 증가하지만 Precision은 감소하는 트레이드오프 관계를 명확하게 보여줍니다. 이는 True Negative(TN)에 의존하지 않고 양성 클래스의 성능만을 집중적으로 평가하기 때문에, 전통적인 정확도(Accuracy) 메트릭보다 더 신뢰성 있는 평가 방법입니다.<br>
곡선 해석에서는 왼쪽 위 모서리(Precision = 1, Recall = 1)에 가까울수록 모델의 성능이 우수함을 의미합니다. 곡선 아래 면적(Area Under Curve, AUC)을 통해 모델의 전반적인 성능을 단일 값으로 요약할 수 있어, 특히 양성 클래스가 희소한 영역에서 매우 유용하게 활용됩니다.

---
**AP(Average Precision)** : Precision-Recall Curve 아래 면적을 계산하여 모델의 전반적인 성능을 0에서 1 사이의 단일 값으로 요약하는 지표<br>

---
**mAP(mean Average Precision)** : AP를 계산한 후 그 평균을 내는 것으로, 모델이 얼마나 다양한 클래스를 정확하게 탐지하는지를 종합적으로 보여주는 지표

---
**NMS(Non-Maximum Suppression)** : 객체 탐지(Object Detection) 알고리즘에서 중복된 경계 상자(Bounding Box)를 제거하고 가장 적합한 경계 상자를 선택하는 후처리 기법
```
주요 작동 원리: 
- 모든 경계 상자를 confidence score 기준으로 정렬합니다.
- 가장 높은 confidence score를 가진 상자를 선택합니다.
- 선택된 상자와 IoU(Intersection over Union)가 특정 임계값 이상인 다른 상자들을 제거합니다.
- 남은 상자들 중 다음으로 높은 confidence score를 가진 상자를 선택하고 과정을 반복합니다.
```

## 성능 지표
### mAP
mAP는 재현율(Recall)과 정밀도(Precision)의 균형을 종합적으로 평가하는 지표입니다. 단순히 한 가지 메트릭에 의존하지 않고, 모델이 얼마나 정확하게(Precision) 그리고 얼마나 빠짐없이(Recall) 객체를 탐지하는지를 동시에 고려합니다. 이를 통해 모델의 성능을 보다 균형 잡히고 신뢰성 있게 측정할 수 있어, 이 지표를 선택하였습니다.

    mAP@50: IoU 0.5 이상일 때의 정확도
    mAP@75: IoU 0.75 이상일 때의 정확도
    mAP@50:95: IoU 0.5에서 0.95까지 0.05 단위로 평균낸 정확도

**mAP@50와 mAP@50:95**를 최우선으로 고려했습니다.
결함의 정확한 위치보다 **존재 여부가 더 중요**하기 때문입니다.

## 학습 결과 분석

### 0. YOLO의 자동 최적화 모드('optimizer=auto') 사용

YOLO의 자동 최적화 모드를 사용해서 데이터셋과 모델 구조에 최적화된 옵티마이저를 선택하도록 하였습니다. 이를 통해 학습률, momentum, 레이어별 weight decay를 자동으로 결정되게 하여 수동 설정 대비 더욱 안정적인 학습이 진행되도록 하였습니다.

### 1. 이미지 크기에 따른 비교 | 640, 원본(1389)

| **2class yolov11s 50epoch img1389** | **2class yolov11s 50epoch img640** |
|-------------------------------|---------------------------------|
|  <img src="https://github.com/user-attachments/assets/45a5cb8f-dd1c-40da-a0f6-c4024a11bbea" alt="2class yolov11s 50epoch img1389" height="300"> | <img src="https://github.com/user-attachments/assets/0c7eaa96-f87e-440b-b731-9adaf5d3b0c7" alt="2class yolov11s 50epoch img640" height="300">  |

> **640**으로 정규화된 이미지로 학습한 결과가 원본 이미지로 학습한 것보다 높은 mAP 값을 갖는 것을 확인하였습니다.

### 2. 클래스 분류 방식에 따른 비교 | 2class, 7class

| **2class yolov11s 50epoch** | **7class yolov11s 50epoch** |
|-------------------------------|---------------------------------|
|  <img src="https://github.com/AI-CV-Tofu/Data-Model/blob/main/seohyeon/yolov11s/2class_6403/PR_curve.png?raw=true" alt="2class" height="300"> | <img src="https://github.com/AI-CV-Tofu/Data-Model/blob/main/minkyoung/yolov11s/7class_640/PR_curve.png?raw=true" alt="7class" height="300">  |

> **7class**로 분류된 라벨로 학습한 결과가 2class로 분류된 라벨로 학습한 것보다 높은 mAP 값을 갖는 것을 확인하였습니다.

### 3. 기포 라벨링 방식에 따른 비교 | 전반적인 분포, 각각의 기포

| **Overall distribution labeling** | **Individual bubble labeling** |
|-------------------------------|---------------------------------|
|  <img src="https://github.com/AI-CV-Tofu/Data-Model/blob/main/minkyoung/yolov11s/7class_640/PR_curve.png?raw=true" alt="Overall distribution labeling" height="300"> | <img src="https://github.com/user-attachments/assets/504a11e0-c75f-4017-94f9-36b388fcc600" alt="Individual bubble labeling" height="300">  |

> **각각의 기포**에 대한 어노테이션으로 학습한 결과가 전반적인 기포에 대한 어노테이션을 학습한 것보다 높은 mAP 값을 갖는 것을 확인하였습니다.

### 4. 대비 조정 | 원본, CLAHE 2.0,  CLAHE 4.0

![image](https://github.com/user-attachments/assets/f9a09d5d-cbf1-4ae3-a533-6a8be80b9d9c)

| original_map_results                                                       | CLAHE2.0_map_results                                                       | CLAHE4.0_map_results                                                       |
|---------------------------------------------------------------------------|---------------------------------------------------------------------------|---------------------------------------------------------------------------|
| ![original_map_results](https://github.com/user-attachments/assets/cddcd2a9-dfd1-46de-8314-c9253773561c) | ![CLAHE2 0_map_results](https://github.com/user-attachments/assets/b271fc00-b580-4e09-b470-01b645c77651) | ![CLAHE4 0_map_results](https://github.com/user-attachments/assets/f47a2d7a-df79-4991-8a29-084a8661b6be) |

| PR_curve_original                                                       | PR_curve_CLAHE2.0                                                       | PR_curve_CLAHE4.0                                                       |
|-------------------------------------------------------------------------|-----------------------------------------------------------------------|-----------------------------------------------------------------------|
| ![PR_curve original](https://github.com/user-attachments/assets/812dad59-6315-4f70-b5b9-b689a85320d3) | ![PR_curve_clahe1](https://github.com/user-attachments/assets/c91f7cac-fb38-4d46-b215-b0c7d04bc08b) | ![PR_curve_clahe2](https://github.com/user-attachments/assets/3ca019ec-a041-48ee-8a43-d188e6a6b1a6) |

분석 결과, CLAHE 2.0과 4.0은 mAP@50에서 각각 0.65, 0.61로 original의 0.60보다 약간 높은 성능을 보였으나, 클래스별 성능의 표준편차(original: 0.284, CLAHE 2.0: 0.292, CLAHE 4.0: 0.293)를 고려할 때 original이 더 균형 잡힌 성능을 보여주었습니다. CLAHE의 대비 강조는 일부 클래스에서는 성능을 향상시키지만 다른 클래스에서는 노이즈로 인한 성능 저하를 초래하기에, 학습에 대비 조정을 하지 않은 original 이미지를 사용하기로 결정했습니다.
```
성능 균형 분석:
Original: 표준편차 0.284 (가장 낮음)
CLAHE2.0: 표준편차 0.292
CLAHE4.0: 표준편차 0.293 (가장 높음)
```

### 5. 7class, Individual bubble labeling, image size 640 에서 최적의 파라미터 탐색 | epochs와 batch

7개 클래스에 대한 개별 기포 라벨링과 640 크기의 이미지를 사용한 실험에서는 최적의 파라미터를 찾기 위해 batch size(8, 16, 32)와 epochs(40, 50, 60, 70, 80)를 다양하게 조정하여 총 15가지 조건에서 실험을 수행했습니다. 각 조건별로 mAP(mean Average Precision) 값을 측정하여 모델의 성능을 평가했으며, 이를 통해 결함 검출에 가장 효과적인 학습 파라미터 조합을 도출했습니다.

**실험 결과, batch size 32와 epoch 40에서 가장 우수한 성능을 보였습니다.**

이 조합에서 mAP@50은 0.74, mAP@75는 0.17, mAP@50:95는 0.35를 기록했습니다. 특히 더 큰 batch size(32)가 일관되게 더 나은 성능을 보였으며, 더 많은 epoch 수가 반드시 성능 향상으로 이어지지는 않았습니다. 결함 탐지에서 중요한 지표인 mAP@50이 최고값을 기록했고, 전체 범위 평균인 mAP@50:95도 가장 높아 안정적인 성능을 보장하는 것으로 판단했습니다.


| Batch Size \ Epoch | 40  | 50  | 60  | 70  | 80  |
|--------------------|-----|-----|-----|-----|-----|
| 8                  |  <img src="https://github.com/AI-CV-Tofu/Data-Model/blob/main/minkyoung/yolov11s_batch_and_epoch/V3_640_batch8_epochs40/PR_curve.png?raw=true" alt="40">   |  <img src="https://github.com/AI-CV-Tofu/Data-Model/blob/main/minkyoung/yolov11s_batch_and_epoch/V3_640_batch8_epochs50/PR_curve.png?raw=true" alt="50">   |  <img src="https://github.com/AI-CV-Tofu/Data-Model/blob/main/minkyoung/yolov11s_batch_and_epoch/V3_640_batch8_epochs60/PR_curve.png?raw=true" alt="60">   |  <img src="https://github.com/AI-CV-Tofu/Data-Model/blob/main/minkyoung/yolov11s_batch_and_epoch/V3_640_batch8_epochs70/PR_curve.png?raw=true" alt="70">   |  <img src="https://github.com/AI-CV-Tofu/Data-Model/blob/main/minkyoung/yolov11s_batch_and_epoch/V3_640_batch8_epochs80/PR_curve.png?raw=true" alt="80">   |
| 16                 |  <img src="https://github.com/AI-CV-Tofu/Data-Model/blob/main/minkyoung/yolov11s_batch_and_epoch/V3_640_batch16_epochs40/PR_curve.png?raw=true" alt="40">   |  <img src="https://github.com/AI-CV-Tofu/Data-Model/blob/main/minkyoung/yolov11s_batch_and_epoch/V3_640_batch16_epochs50/PR_curve.png?raw=true" alt="50">   |  <img src="https://github.com/AI-CV-Tofu/Data-Model/blob/main/minkyoung/yolov11s_batch_and_epoch/V3_640_batch16_epochs60/PR_curve.png?raw=true" alt="60">   |  <img src="https://github.com/AI-CV-Tofu/Data-Model/blob/main/minkyoung/yolov11s_batch_and_epoch/V3_640_batch16_epochs70/PR_curve.png?raw=true" alt="70">   |  <img src="https://github.com/AI-CV-Tofu/Data-Model/blob/main/minkyoung/yolov11s_batch_and_epoch/V3_640_batch32_epochs80/PR_curve.png?raw=true" alt="80">   |
| 32                 |  <img src="https://github.com/AI-CV-Tofu/Data-Model/blob/main/minkyoung/yolov11s_batch_and_epoch/V3_640_batch32_epochs40/PR_curve.png?raw=true" alt="40">   |  <img src="https://github.com/AI-CV-Tofu/Data-Model/blob/main/minkyoung/yolov11s_batch_and_epoch/V3_640_batch32_epochs50/PR_curve.png?raw=true" alt="50">   |  <img src="https://github.com/AI-CV-Tofu/Data-Model/blob/main/minkyoung/yolov11s_batch_and_epoch/V3_640_batch32_epochs60/PR_curve.png?raw=true" alt="60">   |  <img src="https://github.com/AI-CV-Tofu/Data-Model/blob/main/minkyoung/yolov11s_batch_and_epoch/V3_640_batch32_epochs70/PR_curve.png?raw=true" alt="70">   |  <img src="https://github.com/AI-CV-Tofu/Data-Model/blob/main/minkyoung/yolov11s_batch_and_epoch/V3_640_batch32_epochs80/PR_curve.png?raw=true" alt="80">   |


<img src="https://github.com/user-attachments/assets/1cb5b1f7-7301-4cdf-a801-fcd9263d5f92" alt="batch&epochs">
