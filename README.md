# Sewer Binary Classification v9

## 🚀 간단 추론 모듈 (`inference.py`)

`inference.py`는 복잡한 설정 파일(`config.yaml`) 수정 없이, **학습된 모델**(`.pth`)과 **이미지 폴더 경로**를 바탕으로 간편하게 추론하고 결과를 CSV로 저장하는 모듈입니다.

#### 1. 주요 특징
*   **간편한 실행**: `config.yaml` 없이 **CLI 인자만으로도** 실행 가능합니다.
*   **자동 결과 저장**: 추론 결과(파일명, 예측 클래스, 신뢰도)가 **CSV 파일**로 자동 저장됩니다.
*   **독립성**: 학습 로직이 포함된 코드(`main.py`, `baseline.py`)와 별개로 제작해서 추론만 하기에는 더욱 적합합니다.

#### 2. 사용 방법
터미널에서 아래 명령어를 바탕으로 실행합니다.

```bash
python inference.py --img_dir "이미지_폴더_경로" --model "모델_파일_경로"
```

#### 3. 명령어 인자 설명 (Arguments)
*   `--img_dir` (**필수**): 추론을 수행할 이미지들이 저장된 폴더의 경로입니다. 해당 폴더 내의 모든 이미지 파일(jpg, png 등)을 읽어 추론을 진행합니다.
*   `--model` (**필수**): 학습된 모델 가중치 파일(`.pth`)의 경로입니다. `main.py`나 `baseline.py`로 학습하여 저장된 `best_model.pth` 파일 경로를 입력합니다.
*   `--baseline_name` (**선택**): 베이스라인을 사용할 때 지정하는 아키텍처 이름입니다. (예: `efficientnet_b0`, `xie2019`). **제안 모델이 아닌 경우 필수**로 입력해야 합니다.
*   `--pruning_info` (**선택**): 가지치기(Pruning)가 적용된 모델을 추론할 때 필요한 `pruning_info.yaml` 파일의 경로입니다. 이 파일은 Pruning된 모델의 구조를 복원하는 데 사용됩니다. **Pruning된 모델인 경우 필수**입니다.
*   `--config` (**선택 / 생략추천**): 모델 아키텍처 설정이 담긴 YAML 파일 경로입니다. 생략 시 코드 내장 기본 설정이 사용되기 때문에, 학습 시 사용한 설정을 그대로 적용하려면 생략하는 것이 좋습니다.
*   `--use_int8` (**선택**): ONNX Runtime을 사용하여 **INT8 정적 양자화(Static Quantization)** 추론을 수행합니다. CPU 환경에서 추론 속도 향상을 위해 사용합니다. (FP16과 동시 사용 불가)
*   `--use_fp16` (**선택**): ONNX Runtime을 사용하여 **FP16(Half Precision)** 추론을 수행합니다. (INT8과 동시 사용 불가)

---

#### 4. 실행 예시 (Examples)

#### 4.1. 제안 모델
사용 인자 `--img_dir`, `--model`
```bash
python inference.py --img_dir "./data/samples" --model "./pretrained/proposed_model/original/best_model.pth"
```
---
#### 4.2. 원본 베이스라인
사용 인자 `--img_dir`, `--model`, `--baseline_name`
```bash
python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/original/resnet18/best_model.pth" --baseline_name "resnet18"
```
```bash
python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/original/efficientnet_b0/best_model.pth" --baseline_name "efficientnet_b0"
```
```bash
python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/original/mobilenet_v4_s/best_model.pth" --baseline_name "mobilenet_v4_s"
```
```bash
python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/original/xie2019/best_model.pth" --baseline_name "xie2019"
```
```bash
python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/original/deit_tiny/best_model.pth" --baseline_name "deit_tiny"
```
```bash
python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/original/mobile_vit_xxs/best_model.pth" --baseline_name "mobile_vit_xxs"
```
---
#### 4.3. Iso-FLOPs Pruned 베이스라인
사용 인자 `--img_dir`, `--model`, `--baseline_name`, `--pruning_info`
* ##### L1-norm Pruning
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_FLOPs/resnet18_l1/best_model.pth" --baseline_name "resnet18" --pruning_info "./pretrained/baselines/pruned/iso_FLOPs/resnet18_l1/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_FLOPs/efficientnet_b0_l1/best_model.pth" --baseline_name "efficientnet_b0" --pruning_info "./pretrained/baselines/pruned/iso_FLOPs/efficientnet_b0_l1/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_FLOPs/mobilenet_v4_s_l1/best_model.pth" --baseline_name "mobilenet_v4_s" --pruning_info "./pretrained/baselines/pruned/iso_FLOPs/mobilenet_v4_s_l1/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_FLOPs/xie2019_l1/best_model.pth" --baseline_name "xie2019" --pruning_info "./pretrained/baselines/pruned/iso_FLOPs/xie2019_l1/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_FLOPs/deit_tiny_l1/best_model.pth" --baseline_name "deit_tiny" --pruning_info "./pretrained/baselines/pruned/iso_FLOPs/deit_tiny_l1/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_FLOPs/mobile_vit_xxs_l1/best_model.pth" --baseline_name "mobile_vit_xxs" --pruning_info "./pretrained/baselines/pruned/iso_FLOPs/mobile_vit_xxs_l1/pruning_info.yaml"
  ```

* ##### FPGM Pruning
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_FLOPs/resnet18_fpgm/best_model.pth" --baseline_name "resnet18" --pruning_info "./pretrained/baselines/pruned/iso_FLOPs/resnet18_fpgm/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_FLOPs/efficientnet_b0_fpgm/best_model.pth" --baseline_name "efficientnet_b0" --pruning_info "./pretrained/baselines/pruned/iso_FLOPs/efficientnet_b0_fpgm/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_FLOPs/mobilenet_v4_s_fpgm/best_model.pth" --baseline_name "mobilenet_v4_s" --pruning_info "./pretrained/baselines/pruned/iso_FLOPs/mobilenet_v4_s_fpgm/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_FLOPs/xie2019_fpgm/best_model.pth" --baseline_name "xie2019" --pruning_info "./pretrained/baselines/pruned/iso_FLOPs/xie2019_fpgm/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_FLOPs/deit_tiny_fpgm/best_model.pth" --baseline_name "deit_tiny" --pruning_info "./pretrained/baselines/pruned/iso_FLOPs/deit_tiny_fpgm/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_FLOPs/mobile_vit_xxs_fpgm/best_model.pth" --baseline_name "mobile_vit_xxs" --pruning_info "./pretrained/baselines/pruned/iso_FLOPs/mobile_vit_xxs_fpgm/pruning_info.yaml"
  ```
---

#### 4.4. Iso-Params Pruned 베이스라인
사용 인자 `--img_dir`, `--model`, `--baseline_name`, `--pruning_info`
* ##### L1-norm Pruning
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_params/resnet18_l1/best_model.pth" --baseline_name "resnet18" --pruning_info "./pretrained/baselines/pruned/iso_params/resnet18_l1/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_params/efficientnet_b0_l1/best_model.pth" --baseline_name "efficientnet_b0" --pruning_info "./pretrained/baselines/pruned/iso_params/efficientnet_b0_l1/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_params/mobilenet_v4_s_l1/best_model.pth" --baseline_name "mobilenet_v4_s" --pruning_info "./pretrained/baselines/pruned/iso_params/mobilenet_v4_s_l1/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_params/xie2019_l1/best_model.pth" --baseline_name "xie2019" --pruning_info "./pretrained/baselines/pruned/iso_params/xie2019_l1/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_params/deit_tiny_l1/best_model.pth" --baseline_name "deit_tiny" --pruning_info "./pretrained/baselines/pruned/iso_params/deit_tiny_l1/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_params/mobile_vit_xxs_l1/best_model.pth" --baseline_name "mobile_vit_xxs" --pruning_info "./pretrained/baselines/pruned/iso_params/mobile_vit_xxs_l1/pruning_info.yaml"
  ```

* ##### FPGM Pruning
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_params/resnet18_fpgm/best_model.pth" --baseline_name "resnet18" --pruning_info "./pretrained/baselines/pruned/iso_params/resnet18_fpgm/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_params/efficientnet_b0_fpgm/best_model.pth" --baseline_name "efficientnet_b0" --pruning_info "./pretrained/baselines/pruned/iso_params/efficientnet_b0_fpgm/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_params/mobilenet_v4_s_fpgm/best_model.pth" --baseline_name "mobilenet_v4_s" --pruning_info "./pretrained/baselines/pruned/iso_params/mobilenet_v4_s_fpgm/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_params/xie2019_fpgm/best_model.pth" --baseline_name "xie2019" --pruning_info "./pretrained/baselines/pruned/iso_params/xie2019_fpgm/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_params/deit_tiny_fpgm/best_model.pth" --baseline_name "deit_tiny" --pruning_info "./pretrained/baselines/pruned/iso_params/deit_tiny_fpgm/pruning_info.yaml"
  ```
  ```bash
  python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/pruned/iso_params/mobile_vit_xxs_fpgm/best_model.pth" --baseline_name "mobile_vit_xxs" --pruning_info "./pretrained/baselines/pruned/iso_params/mobile_vit_xxs_fpgm/pruning_info.yaml"
  ```

---

#### 4.5. `.pth` → ONNX 변환 후 INT8/FP16 적용 예시 (플래그 사용)
`.pth` 파일을 입력으로 주고 스크립트 내부에서 ONNX로 변환하여 INT8 또는 FP16 추론을 수행하려면 `--use_int8` 또는 `--use_fp16` 플래그를 함께 사용합니다. (두 플래그는 동시에 사용 불가)

- 제안 모델 (`.pth` → INT8 변환 후 추론)
```bash
python inference.py --img_dir "./data/samples" --model "./pretrained/proposed_model/original/best_model.pth" --use_int8
```

- 제안 모델 (`.pth` → FP16 변환 후 추론)
```bash
python inference.py --img_dir "./data/samples" --model "./pretrained/proposed_model/original/best_model.pth" --use_fp16
```

- 베이스라인 모델 예시 (`.pth` → INT8, 모델명 명시)
```bash
python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/original/efficientnet_b0/best_model.pth" --baseline_name "efficientnet_b0" --use_int8
```

- 베이스라인 모델 예시 (`.pth` → FP16, 모델명 명시)
```bash
python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/original/deit_tiny/best_model.pth" --baseline_name "deit_tiny" --use_fp16
```

#### 4.6. ONNX Runtime 직접 사용 예시 (플래그 생략)
이미 FP16/INT8로 변환되어 있는 `.onnx` 파일을 입력으로 주어 ONNX Runtime으로 바로 추론할 때는 `--use_int8`/`--use_fp16` 플래그를 사용하지 마십시오. (`--model`에 `.onnx` 경로만 전달하면 스크립트가 자동으로 ONNX Runtime 세션을 생성합니다.)

- 제안 모델 (이미 변환된 INT8 ONNX)
```bash
python inference.py --img_dir "./data/samples" --model "./pretrained/proposed_model/int8/best_model_int8.onnx"
```

- 제안 모델 (이미 변환된 FP16 ONNX)
```bash
python inference.py --img_dir "./data/samples" --model "./pretrained/proposed_model/fp16/best_model_fp16.onnx"
```

- 베이스라인 모델 예시 (이미 변환된 INT8 ONNX, 모델명 명시)
```bash
python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/int8/efficientnet_b0/best_model_int8.onnx" --baseline_name "efficientnet_b0"
```

- 베이스라인 모델 예시 (이미 변환된 FP16 ONNX, 모델명 명시)
```bash
python inference.py --img_dir "./data/samples" --model "./pretrained/baselines/fp16/deit_tiny/best_model_fp16.onnx" --baseline_name "deit_tiny"
```

#### 5. 테스트용 샘플
- **./data/samples**: 아래 두 종류의 샘플을 합친 500장
  - ./data/samples_defect_350: 제안 모델이 Defect 분류한 이미지 샘플 350장
  - ./data/samples_normal_150: 제안 모델이 Normal로 분류한 이미지 샘플 150장

----

#### 6. ✨ 전체 데이터셋
pretrained 폴더에 저장된 혼동행렬의 성능을 직접 확인하기 위해서는 테스트용 샘플 대신에 **전체 데이터셋** (abnormal: 815, normal: 877)을 --img_dir 경로로 지정해야 합니다.
(용량 문제로 인해 본 압축 파일에는 전체 데이터셋이 존재하지 않습니다.)

## 📝 프로젝트 설명

#### 1. 소개 (Introduction)
이 문서는 `sewer_binary_cls_v7` 프로젝트의 `main.py`와 `baseline.py`를 사용하여 모델을 훈련하고 추론하는 방법을 안내합니다.

#### 2. 실행 환경 설정 (Configuration)
모든 실행 관련 설정은 `config.yaml` 파일을 통해 제어됩니다. 스크립트를 실행하기 전에 이 파일을 사용자의 환경과 목적에 맞게 수정해야 합니다.

#### 3. 사전 훈련된 모델로 추론하기 (Inference with Pre-trained Models)

기존에 훈련 및 저장된 `.pth` 모델 가중치를 사용하여 추론을 수행하고, 이전에 기록된 성능과 동일한 결과를 재현할 수 있습니다.

**3.1. `config.yaml` 파일 수정**:
*   `run.mode`를 `'inference'`로 설정하여 실행 모드를 추론으로 변경합니다.
*   `run.pth_inference_dir`에 추론에 사용할 `.pth` 파일이 위치한 디렉토리 경로를 정확히 입력합니다.
*   `run.pth_best_name`에 불러올 모델의 파일명(예: `best_model.pth`)을 정확히 입력합니다.

    ```yaml

    run:
      mode: 'inference'
      pth_inference_dir: 'path/to/your/pth/directory' 
      pth_best_name: 'best_model.pth'
    ```

**3.2. 스크립트 실행**:
*   설정이 완료되면 `main.py` 또는 `baseline.py`를 실행하여 추론을 시작합니다.
*   두 스크립트는 사용하는 모델 아키텍처에 따라 구분됩니다.

**3.3.  `main.py` vs `baseline.py`**


*   **`main.py` 실행**:
    *   `models.py`에 정의된 커스텀 아키텍처 모델을 실행할 때 사용합니다.
    *   모델의 세부 구조(예: CNN 백본, 트랜스포머 레이어 수 등)는 `config.yaml`의 `model` 섹션에서 상세하게 설정할 수 있습니다.

*   **`baseline.py` 실행**:
    *   ResNet, EfficientNet, ViT, Swin Transformer 등 표준적인 딥러닝 모델을 실행할 때 사용합니다.
    *   사용할 모델은 `config.yaml`의 `baseline` 섹션에 있는 `model_name` 파라미터로 지정합니다.

#### 4. Baseline 모델 추론 가이드 (`baseline.py`)

`baseline.py`를 사용하여 추론할 때는 **원본 모델**인지 **가지치기(pruning)된 모델**인지에 따라 `config.yaml` 설정을 다르게 해야 합니다.

**4.1. 원본 모델 (Unpruned Model) 추론**

가지치기가 적용되지 않은 원본 모델의 성능을 재현하려면, `config.yaml`의 `baseline` 섹션에 있는 모든 가지치기 관련 옵션을 `false`로 설정하고 관련 수치들을 비활성화(0 또는 주석 처리)해야 합니다.

```yaml


baseline:
  model_name: 'xie2019' # 추론하려는 모델 이름

  # --- 모든 경량화 옵션을 false로 설정 ---
  use_l1_pruning: false
  use_fpgm_pruning: false
  # pruning_sparsity: 0.0 # 사용되지 않음
  pruning_flops_target: 0.0 # 0으로 설정하여 비활성화
```

**4.2. 가지치기된 모델 (Pruned Model) 추론**

* 가지치기된 모델의 성능을 재현하려면, 해당 모델을 훈련할 때 생성된 `pruning_info.yaml` 파일을 반드시 참조해야 합니다.

1.  **`pruning_info.yaml` 확인**: 모델 훈련 시 생성된 로그 디렉토리에서 `pruning_info.yaml` 파일을 찾아, 적용되었던 가지치기 종류(`pruning_method`)와 희소도(`pruning_sparsity`) 값을 확인합니다.

2.  **`config.yaml` 수정**:
    *   `pruning_info.yaml`에 명시된 가지치기 방법에 해당하는 `use_..._pruning` 옵션만 `true`로 설정하고, 나머지는 모두 `false`로 설정합니다.
    *   `pruning_sparsity` 값을 `pruning_info.yaml`에서 확인한 값과 **정확히 동일하게** `config.yaml`에 복사하여 붙여넣습니다.

    **예시**: `depgraph` 방식으로 `0.4756...` 만큼 가지치기된 모델을 재현하는 경우

    ```yaml

    baseline:
      model_name: 'xie2019' # 예시 모델

      # --- 경량화 옵션 ---
      use_fpgm_pruning: true # pruning_info.yaml에 명시된 프루닝만 true로 설정
      # ... 나머지 프루닝 관련 use_ 옵션은 모두 false ...

      pruning_sparsity: 0.4756640625 # pruning_info.yaml에서 가져온 값
      pruning_flops_target: 0.0      # sparsity를 직접 지정하므로 0으로 설정
    ```

#### 5. 순수 추론 및 결과 저장 (Pure Inference)

정답 레이블이 없는 데이터에 대해 추론만 수행하고, 각 데이터에 대한 예측 결과를 파일로 저장하고 싶을 때 `only_inference` 옵션을 사용합니다. 이는 실제 배포 환경에서 유용하게 사용될 수 있습니다.

*   **`config.yaml` 파일 수정**:
    *   `run.only_inference`를 `true`로 설정합니다. `false`로 설정하면 정답 레이블과 비교하여 성능 평가를 수행합니다.

    ```yaml
    run:
      mode: 'inference'
      only_inference: true
    
    dataset:
      type: 'image_folder'
      train_split_ratio: 0.0 # 모든 이미지를 추론에 사용하기 위해 0.0으로 설정
    ```

*   **실행 결과**:
    *   스크립트 실행이 완료되면, 실행 로그가 저장되는 디렉토리에 `inference_results_{timestamp}.csv` 파일이 생성됩니다.
    *   이 CSV 파일에는 각 이미지 파일의 경로, 모델이 예측한 클래스, 그리고 해당 예측에 대한 신뢰도(confidence score)가 기록됩니다.

#### 6. ONNX 변환 및 평가 (ONNX Conversion & Evaluation)

`config.yaml` 파일의 `evaluate_onnx` 옵션을 `true`로 설정하면, PyTorch 모델(`.pth`)을 ONNX(Open Neural Network Exchange) 형식으로 변환하고 평가하는 기능을 활성화할 수 있습니다.

*   **동작 시점**:
    *   **`train` 모드**: 훈련이 완료된 후, 최고 성능의 모델을 테스트셋으로 최종 평가하는 과정에서 `.pth` 파일이 `.onnx` 파일로 변환되어 로그 디렉토리에 저장됩니다.
    *   **`inference` 모드**: 지정된 `.pth` 파일을 불러와 추론을 수행한 후, 해당 모델을 `.onnx` 파일로 변환하여 로그 디렉토리에 저장합니다.

*   **설정 방법**:
    `config.yaml` 파일에서 `evaluate_onnx` 값을 `true`로 설정합니다.

    ```yaml
    run:
      # ...
      evaluate_onnx: true
    ```

*   **결과**:
    *   실행 로그가 저장되는 디렉토리(예: `log/DATASET_NAME/RUN_TIMESTAMP/`)에 `model_TIMESTAMP.onnx`와 같은 이름으로 ONNX 파일이 생성됩니다.
    *   변환된 ONNX 모델의 성능(추론 시간, 정확도 등)이 PyTorch 모델과 함께 측정되어 로그에 기록됩니다.