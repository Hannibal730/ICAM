import os
import re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from PIL import Image

def plot_and_save_val_accuracy_graph(log_file_path, save_dir, final_acc, timestamp):
    """
    로그 파일에서 에포크별 [Valid] Accuracy를 추출하여 그래프를 그리고,
    지정된 디렉토리에 이미지 파일로 저장합니다. (Validation Accuracy 전용)

    Args:
        log_file_path (str): 분석할 로그 파일의 전체 경로.
        save_dir (str): 그래프 이미지를 저장할 디렉토리 경로.
        final_acc (float): 그래프 제목에 표시할 최종 추론 정확도.
        timestamp (str): 파일 이름에 추가할 타임스탬프.
    """
    epochs = []
    accuracies = []

    # 그래프를 저장할 전용 폴더 생성
    graph_dir = os.path.join(save_dir, f'graph_{timestamp}')
    os.makedirs(graph_dir, exist_ok=True)

    # 정규 표현식을 사용하여 로그 라인에서 에포크와 정확도 추출
    # 예: [Valid] [1/30] | Val Acc: 85.00% ...
    pattern = re.compile(r"\[Valid\] \[(\d+)/\d+\] \|.*?Val Acc: ([\d\.]+)%")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    epochs.append(int(match.group(1)))
                    accuracies.append(float(match.group(2)))

        if not epochs:
            logging.warning("로그 파일에서 유효한 [Valid] Accuracy 데이터를 찾을 수 없어 그래프를 생성하지 않습니다.")
            return

        plt.figure(figsize=(12, 8))
        plt.plot(epochs, accuracies, marker='.', linestyle='-', color='b')
        plt.title(f'Validation Accuracy per Epoch (Final Test Acc: {final_acc:.2f}%)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100) # Y축 범위를 0부터 100까지로 고정
        plt.yticks(range(0, 101, 10)) # Y축 눈금을 10 단위로 설정
        plt.grid(True)

        png_save_path = os.path.join(graph_dir, 'val_acc.png')
        pdf_save_path = os.path.join(graph_dir, 'val_acc.pdf')
        plt.tight_layout()
        plt.savefig(png_save_path)
        plt.savefig(pdf_save_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Val Acc 그래프 저장 완료: '{png_save_path}' and '{pdf_save_path}'")

    except FileNotFoundError:
        logging.error(f"로그 파일 '{log_file_path}'를 찾을 수 없습니다.")
    except Exception as e:
        logging.error(f"그래프 생성 중 오류 발생: {e}")

def plot_and_save_train_val_accuracy_graph(log_file_path, save_dir, final_acc, timestamp):
    """
    로그 파일에서 Train 및 Valid Accuracy를 추출하여 하나의 그래프에 그리고,
    지정된 디렉토리에 이미지 파일로 저장합니다.

    Args:
        log_file_path (str): 분석할 로그 파일의 전체 경로.
        save_dir (str): 그래프 이미지를 저장할 디렉토리 경로.
        final_acc (float): 그래프 제목에 표시할 최종 추론 정확도.
        timestamp (str): 파일 이름에 추가할 타임스탬프.
    """
    train_epochs, train_accuracies = [], []
    val_epochs, val_accuracies = [], []

    # 그래프를 저장할 전용 폴더 생성
    graph_dir = os.path.join(save_dir, f'graph_{timestamp}')
    os.makedirs(graph_dir, exist_ok=True)

    # Train 및 Valid 정확도 추출을 위한 정규 표현식
    train_pattern = re.compile(r"\[Train\] \[(\d+)/\d+\] \| .* Train Acc: ([\d\.]+)%")
    val_pattern = re.compile(r"\[Valid\] \[(\d+)/\d+\] \|.*?Val Acc: ([\d\.]+)%")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                train_match = train_pattern.search(line)
                val_match = val_pattern.search(line)
                if train_match:
                    train_epochs.append(int(train_match.group(1)))
                    train_accuracies.append(float(train_match.group(2)))
                if val_match:
                    val_epochs.append(int(val_match.group(1)))
                    val_accuracies.append(float(val_match.group(2)))

        if not val_epochs:
            logging.warning("로그 파일에서 유효한 [Valid] Accuracy 데이터를 찾을 수 없어 그래프를 생성하지 않습니다.")
            return

        plt.figure(figsize=(12, 8))
        
        # Train Accuracy (빨간색 점선)와 Valid Accuracy (파란색 실선) 플로팅
        if train_epochs:
            plt.plot(train_epochs, train_accuracies, marker='.', linestyle='-', color='r', label='Train Accuracy')
        plt.plot(val_epochs, val_accuracies, marker='.', linestyle='-', color='b', label='Validation Accuracy')
        
        plt.title(f'Train & Validation Accuracy per Epoch (Final Test Acc: {final_acc:.2f}%)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        plt.yticks(range(0, 101, 10))
        plt.grid(True)
        plt.legend() # 범례 표시

        png_save_path = os.path.join(graph_dir, 'train_val_acc.png')
        pdf_save_path = os.path.join(graph_dir, 'train_val_acc.pdf')
        plt.tight_layout()
        plt.savefig(png_save_path)
        plt.savefig(pdf_save_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Train/Val Acc 그래프 저장 완료: '{png_save_path}' and '{pdf_save_path}'")

    except FileNotFoundError:
        logging.error(f"로그 파일 '{log_file_path}'를 찾을 수 없습니다.")
    except Exception as e:
        logging.error(f"그래프 생성 중 오류 발생: {e}")

def plot_and_save_confusion_matrix(y_true, y_pred, class_names, save_dir, timestamp):
    """
    실제 값과 예측 값을 기반으로 혼동 행렬(Confusion Matrix)을 계산하고,
    이를 시각화하여 지정된 경로에 이미지 파일로 저장합니다.

    Args:
        y_true (list or np.array): 실제 레이블 리스트.
        y_pred (list or np.array): 모델이 예측한 레이블 리스트.
        class_names (list of str): 클래스 이름 리스트 (e.g., ['Normal', 'Defect']).
        save_dir (str): 혼동 행렬 이미지를 저장할 디렉토리 경로.
        timestamp (str): 파일 이름에 추가할 타임스탬프.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 16}) # 숫자 크기 조절
        
        plt.title('Confusion Matrix', fontsize=20)
        plt.ylabel('Actual', fontsize=15)
        plt.xlabel('Predicted', fontsize=15)
        
        png_save_path = os.path.join(save_dir, f'confusion_matrix_{timestamp}.png')
        pdf_save_path = os.path.join(save_dir, f'confusion_matrix_{timestamp}.pdf')
        plt.tight_layout()
        plt.savefig(png_save_path)
        plt.savefig(pdf_save_path, bbox_inches='tight')
        plt.close()
        logging.info("="*50)
        logging.info(f"혼동 행렬 저장 완료. '{png_save_path}' and '{pdf_save_path}'")
    except Exception as e:
        logging.error(f"혼동 행렬 생성 중 오류 발생: {e}")

def plot_and_save_f1_normal_graph(log_file_path, save_dir, timestamp, class_names):
    """
    로그 파일에서 'Normal' 클래스의 F1 점수를 추출하여 그래프를 그리고,
    지정된 디렉토리에 이미지 파일로 저장합니다.

    Args:
        log_file_path (str): 분석할 로그 파일의 전체 경로.
        save_dir (str): 그래프 이미지를 저장할 디렉토리 경로.
        timestamp (str): 파일 이름에 추가할 타임스탬프.
        class_names (list of str): 데이터셋의 클래스 이름 리스트.
    """
    epochs = []
    f1_scores = []

    # 그래프를 저장할 전용 폴더 생성
    graph_dir = os.path.join(save_dir, f'graph_{timestamp}')
    os.makedirs(graph_dir, exist_ok=True)

    # 'normal' 클래스 이름 찾기 (대소문자 구분 없이)
    normal_class_name = None
    for name in class_names:
        if name.lower() == 'normal':
            normal_class_name = name
            break

    if normal_class_name is None:
        logging.warning("클래스 이름 목록에서 'normal' 클래스를 찾을 수 없어 F1 (Normal) 그래프를 생성하지 않습니다.")
        return

    # 에포크와 F1 점수 추출을 위한 정규 표현식
    epoch_pattern = re.compile(r"\[Valid\] \[(\d+)/\d+\]")
    f1_pattern = re.compile(rf"\[Metrics for '{re.escape(normal_class_name)}'\] .* F1: ([\d\.]+)")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            current_epoch = None
            for line in f:
                epoch_match = epoch_pattern.search(line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                
                f1_match = f1_pattern.search(line)
                if f1_match and current_epoch is not None:
                    epochs.append(current_epoch)
                    f1_scores.append(float(f1_match.group(1)))
                    current_epoch = None # 한 에포크에 한 번만 기록

        if not epochs:
            logging.warning(f"로그 파일에서 유효한 F1 ({normal_class_name}) 데이터를 찾을 수 없어 그래프를 생성하지 않습니다.")
            return

        plt.figure(figsize=(12, 8))
        plt.plot(epochs, f1_scores, marker='.', linestyle='-', color='g', label=f'F1 Score ({normal_class_name})')
        plt.title(f'F1 Score ({normal_class_name}) per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.ylim(0, 1.0)
        plt.grid(True)
        plt.legend()

        png_save_path = os.path.join(graph_dir, 'F1_normal.png')
        pdf_save_path = os.path.join(graph_dir, 'F1_normal.pdf')
        plt.tight_layout()
        plt.savefig(png_save_path)
        plt.savefig(pdf_save_path, bbox_inches='tight')
        plt.close()
        logging.info(f"F1 ({normal_class_name}) 그래프 저장 완료: '{png_save_path}' and '{pdf_save_path}'")
    except Exception as e:
        logging.error(f"F1 ({normal_class_name}) 그래프 생성 중 오류 발생: {e}")

def plot_and_save_loss_graph(log_file_path, save_dir, timestamp):
    """
    로그 파일에서 에포크별 [Valid] Loss를 추출하여 그래프를 그리고,
    지정된 디렉토리에 이미지 파일로 저장합니다.

    Args:
        log_file_path (str): 분석할 로그 파일의 전체 경로.
        save_dir (str): 그래프 이미지를 저장할 디렉토리 경로.
        timestamp (str): 파일 이름에 추가할 타임스탬프.
    """
    epochs = []
    losses = []

    graph_dir = os.path.join(save_dir, f'graph_{timestamp}')
    os.makedirs(graph_dir, exist_ok=True)

    # [Valid] [에포크/총에포크] | Loss: 값 | ... 형식의 로그를 찾습니다.
    pattern = re.compile(r"\[Valid\] \[(\d+)/\d+\] \| Loss: ([\d\.]+)")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    epochs.append(int(match.group(1)))
                    losses.append(float(match.group(2)))

        if not epochs:
            logging.warning("로그 파일에서 유효한 [Valid] Loss 데이터를 찾을 수 없어 Validation Loss 그래프를 생성하지 않습니다.")
            return

        plt.figure(figsize=(12, 8))
        plt.plot(epochs, losses, marker='.', linestyle='-', color='orange', label='Validation Loss')
        plt.title('Validation Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        png_save_path = os.path.join(graph_dir, 'val_loss.png')
        pdf_save_path = os.path.join(graph_dir, 'val_loss.pdf')
        plt.tight_layout()
        plt.savefig(png_save_path)
        plt.savefig(pdf_save_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Validation Loss 그래프 저장 완료: '{png_save_path}' and '{pdf_save_path}'")

    except Exception as e:
        logging.error(f"Validation Loss 그래프 생성 중 오류 발생: {e}")

def plot_and_save_lr_graph(log_file_path, save_dir, timestamp):
    """
    로그 파일에서 에포크별 Learning Rate를 추출하여 그래프를 그리고,
    지정된 디렉토리에 이미지 파일로 저장합니다.

    Args:
        log_file_path (str): 분석할 로그 파일의 전체 경로.
        save_dir (str): 그래프 이미지를 저장할 디렉토리 경로.
        timestamp (str): 파일 이름에 추가할 타임스탬프.
    """
    epochs = []
    learning_rates = []

    graph_dir = os.path.join(save_dir, f'graph_{timestamp}')
    os.makedirs(graph_dir, exist_ok=True)

    # [LR] [에포크/총에포크] | Learning Rate: 값 형식의 로그를 찾습니다.
    pattern = re.compile(r"\[LR\]\s+\[(\d+)/\d+\] \| Learning Rate: ([\d\.e\-\+]+)")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    epochs.append(int(match.group(1)))
                    learning_rates.append(float(match.group(2)))

        if not epochs:
            logging.warning("로그 파일에서 유효한 Learning Rate 데이터를 찾을 수 없어 LR 그래프를 생성하지 않습니다.")
            return

        plt.figure(figsize=(12, 8))
        plt.plot(epochs, learning_rates, marker='.', linestyle='-', color='black', label='Learning Rate')
        plt.title('Learning Rate per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.legend()

        png_save_path = os.path.join(graph_dir, 'learning_rate.png')
        pdf_save_path = os.path.join(graph_dir, 'learning_rate.pdf')
        plt.tight_layout()
        plt.savefig(png_save_path)
        plt.savefig(pdf_save_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Learning Rate 그래프 저장 완료: '{png_save_path}' and '{pdf_save_path}'")

    except Exception as e:
        logging.error(f"Learning Rate 그래프 생성 중 오류 발생: {e}")

def plot_and_save_compiled_graph(save_dir, timestamp):
    """
    생성된 4개의 주요 그래프(Train/Val Acc, F1 Normal, Loss, LR)를
    하나의 2x2 이미지로 합쳐 'compile.png'로 저장합니다.

    Args:
        save_dir (str): 개별 그래프가 저장된 부모 디렉토리 경로.
        timestamp (str): 파일 경로 구성에 사용될 타임스탬프.
    """
    graph_dir = os.path.join(save_dir, f'graph_{timestamp}')
    
    # 합칠 그래프 파일들의 경로 정의
    graph_paths = {
        'acc': os.path.join(graph_dir, 'train_val_acc.png'),
        'f1': os.path.join(graph_dir, 'F1_normal.png'),
        'loss': os.path.join(graph_dir, 'val_loss.png'),
        'lr': os.path.join(graph_dir, 'learning_rate.png')
    }

    # 모든 그래프 파일이 존재하는지 확인
    missing_files = [name for name, path in graph_paths.items() if not os.path.exists(path)]
    if missing_files:
        logging.warning(f"다음 그래프 파일이 없어 종합 그래프를 생성할 수 없습니다: {', '.join(missing_files)}")
        return

    try:
        # 2x2 서브플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=(25, 20))
        fig.suptitle(f'Training Summary ({timestamp})', fontsize=24)

        # 각 위치에 그래프 이미지 로드 및 표시
        axes[0, 0].imshow(plt.imread(graph_paths['acc']))
        axes[0, 0].set_title('Train/Validation Accuracy', fontsize=18)

        axes[0, 1].imshow(plt.imread(graph_paths['f1']))
        axes[0, 1].set_title('F1 Score (Normal)', fontsize=18)

        axes[1, 0].imshow(plt.imread(graph_paths['loss']))
        axes[1, 0].set_title('Validation Loss', fontsize=18)

        axes[1, 1].imshow(plt.imread(graph_paths['lr']))
        axes[1, 1].set_title('Learning Rate', fontsize=18)

        # 모든 축의 눈금 및 테두리 제거
        for ax in axes.flat:
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        compile_png_path = os.path.join(graph_dir, 'compile.png')
        compile_pdf_path = os.path.join(graph_dir, 'compile.pdf')
        plt.savefig(compile_png_path)
        plt.savefig(compile_pdf_path, bbox_inches='tight')
        plt.close()
        logging.info(f"종합 그래프 저장 완료: '{compile_png_path}' and '{compile_pdf_path}'")
    except Exception as e:
        logging.error(f"종합 그래프 생성 중 오류 발생: {e}")

def plot_and_save_attention_maps(attention_maps, image_tensor, save_dir, img_size, model_cfg, sample_idx=0, original_filename=None, actual_class=None, predicted_class=None):
    """
    어텐션 맵을 다양한 형식으로 시각화하고, 원본 이미지 이름에 해당하는 폴더 내에
    'compile', 'original', 'head_N' 형태로 분리하여 저장합니다.
    Args:
        attention_maps (torch.Tensor): 모델에서 추출한 어텐션 가중치 텐서.
            Shape: [B, num_heads, num_queries, num_keys]
        image_tensor (torch.Tensor): 원본 이미지 텐서. Shape: [B, C, H, W]
        save_dir (str): 시각화 결과를 저장할 '전용' 디렉토리 경로 (e.g., .../attention_map_timestamp/).
        img_size (int): 원본 이미지의 크기.
        model_cfg (SimpleNamespace): 모델 설정.
        sample_idx (int): 배치에서 시각화할 샘플의 인덱스.
        original_filename (str, optional): 원본 이미지 파일 이름.
        actual_class (str, optional): 실제 클래스 이름.
        predicted_class (str, optional): 예측된 클래스 이름.
    """
    try:
        # 1. 시각화를 위해 배치에서 해당 인덱스(sample_idx)의 데이터만 선택
        if original_filename:
            base_name, _ = os.path.splitext(original_filename)
            # 원본 파일명으로 폴더 생성
            output_folder = os.path.join(save_dir, base_name)
            os.makedirs(output_folder, exist_ok=True)
        else:
            # 파일명이 없는 경우를 대비한 폴백
            output_folder = save_dir
            base_name = f"attention_map_{sample_idx}"

        attention_maps = attention_maps[sample_idx].detach().cpu() # [num_heads, num_queries, num_keys]
        image = image_tensor[sample_idx].detach().cpu()         # [C, H, W]

        # 2. 텐서 정규화 해제 및 이미지로 변환
        # Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])를 역으로 적용
        mean = torch.tensor([0.523, 0.453, 0.345]).view(3, 1, 1)
        std = torch.tensor([0.210, 0.199, 0.154]).view(3, 1, 1)
        image = image * std + mean
        # permute 후 contiguous()를 호출하여 메모리 연속성을 보장한 후 numpy로 변환
        image = image.permute(1, 2, 0).contiguous().numpy().clip(0, 1)

        # 3. 어텐션 맵 차원 정보 추출
        num_heads, num_queries, num_patches = attention_maps.shape
        num_patches_per_side = int(num_patches**0.5)
        if num_patches_per_side * num_patches_per_side != num_patches:
            logging.error("어텐션 맵의 패치 수가 제곱수가 아니므로 2D로 변환할 수 없습니다.")
            return

        # 4. 원본 이미지 저장 (_original.png)
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis('off') # 제목(title)을 제거합니다.
        original_save_path = os.path.join(output_folder, f"{base_name}_original.png")
        plt.savefig(original_save_path, bbox_inches='tight')
        original_pdf_path = os.path.join(output_folder, f"{base_name}_original.pdf")
        Image.open(original_save_path).convert('RGB').save(original_pdf_path, "PDF", resolution=100.0)
        plt.close()

        # 5. 컴파일된 전체 어텐션 맵 저장 (_compile.png)
        # 원본 이미지를 위한 열(1) + 평균 어텐션 맵을 위한 열(1)을 추가하여 (num_heads + 2) 열로 설정
        fig, axes = plt.subplots(num_queries, num_heads + 2, figsize=((num_heads + 2) * 5, num_queries * 5), squeeze=False)
        
        # 전체 그래프 제목 설정
        title = 'Attention Maps per Head and Query'
        filename_info = f'Filename: {original_filename}' if original_filename else ''
        subtitle = f'Actual: {actual_class} / Predicted: {predicted_class}'
        fig.suptitle(f'{title}\n\n{filename_info}\n\n{subtitle}', fontsize=16)

        # 5. 첫 번째 열에 원본 이미지 표시
        # 각 쿼리 행(row)의 첫 번째 열(column)에 원본 이미지를 그립니다.
        for query_idx in range(num_queries):
            ax_orig = axes[query_idx, 0]
            ax_orig.imshow(image)
            # 첫 번째 쿼리에 대해서만 제목을 표시합니다.
            if query_idx == 0:
                ax_orig.set_title('Original Image')
            ax_orig.axis('off')

        # 두 번째 열부터 각 헤드와 쿼리에 대해 반복하며 히트맵 생성
        for head in range(num_heads):
            for query_patch in range(num_queries):
                # 올바른 위치: [해당 쿼리 행, 원본 이미지 열 다음부터]
                ax = axes[query_patch, head + 1]
                
                # 1D 어텐션 맵 -> 2D 그리드로 변환
                attn_map_2d = attention_maps[head, query_patch].view(1, 1, num_patches_per_side, num_patches_per_side)
                
                # 원본 이미지 크기로 업샘플링
                upscaled_map = F.interpolate(attn_map_2d, size=(img_size, img_size), mode='bilinear', align_corners=False)
                upscaled_map = upscaled_map.squeeze().numpy()

                # 원본 이미지와 히트맵 그리기
                ax.imshow(image, extent=(0, img_size, 0, img_size))
                ax.imshow(upscaled_map, cmap='jet', alpha=0.3, extent=(0, img_size, 0, img_size))

                # 변수명을 명확히 하고, 제목에 Query_patch를 사용하여 객관적인 정보를 표시합니다.
                ax.set_title(f'Head {head+1} / Query_patch {query_patch+1}')
                ax.axis('off')

        # 마지막 열에 헤드 평균 어텐션 맵 표시
        for query_patch in range(num_queries):
            ax_avg = axes[query_patch, num_heads + 1]
            
            # 해당 쿼리에 대한 모든 헤드의 평균 계산
            # attention_maps: [num_heads, num_queries, num_patches]
            avg_attn_map = attention_maps[:, query_patch, :].mean(dim=0) # [num_patches]
            
            # 1D -> 2D 그리드로 변환 및 업샘플링
            attn_map_2d = avg_attn_map.view(1, 1, num_patches_per_side, num_patches_per_side)
            upscaled_map = F.interpolate(attn_map_2d, size=(img_size, img_size), mode='bilinear', align_corners=False)
            upscaled_map = upscaled_map.squeeze().numpy()
            
            ax_avg.imshow(image, extent=(0, img_size, 0, img_size))
            ax_avg.imshow(upscaled_map, cmap='jet', alpha=0.3, extent=(0, img_size, 0, img_size))
            
            ax_avg.set_title(f'Average / Query_patch {query_patch+1}')
            ax_avg.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # suptitle 공간 확보를 위해 rect 조정
        
        compile_save_path = os.path.join(output_folder, f"{base_name}_compile.png")
        plt.savefig(compile_save_path)
        compile_pdf_path = os.path.join(output_folder, f"{base_name}_compile.pdf")
        Image.open(compile_save_path).convert('RGB').save(compile_pdf_path, "PDF", resolution=100.0)
        plt.close()

        # 6. 헤드별 어텐션 맵 저장 (_headN.png)
        for head in range(num_heads):
            # 해당 헤드의 모든 쿼리에 대한 어텐션 맵을 평균냅니다.
            head_attn_map = attention_maps[head].mean(dim=0) # [num_patches]
            
            # 1D -> 2D 그리드로 변환 및 업샘플링
            attn_map_2d = head_attn_map.view(1, 1, num_patches_per_side, num_patches_per_side)
            upscaled_map = F.interpolate(attn_map_2d, size=(img_size, img_size), mode='bilinear', align_corners=False)
            upscaled_map = upscaled_map.squeeze().numpy()

            # 시각화
            plt.figure(figsize=(8, 8))
            plt.imshow(image, extent=(0, img_size, 0, img_size))
            plt.imshow(upscaled_map, cmap='jet', alpha=0.3, extent=(0, img_size, 0, img_size))
            plt.axis('off')
            # plt.title(f'Attention Map - Head {head+1}') # 제목(title)을 제거합니다.
            head_save_path = os.path.join(output_folder, f"{base_name}_head{head+1}.png")
            plt.savefig(head_save_path, bbox_inches='tight')
            head_pdf_path = os.path.join(output_folder, f"{base_name}_head{head+1}.pdf")
            Image.open(head_save_path).convert('RGB').save(head_pdf_path, "PDF", resolution=100.0)
            plt.close()

        # 7. 전체 헤드 평균 어텐션 맵 저장 (_avg.png)
        # 모든 헤드와 모든 쿼리에 대해 평균을 냅니다.
        layer_avg_map = attention_maps.mean(dim=(0, 1)) # [num_patches]

        # 1D -> 2D 그리드로 변환 및 업샘플링
        attn_map_2d = layer_avg_map.view(1, 1, num_patches_per_side, num_patches_per_side)
        upscaled_map = F.interpolate(attn_map_2d, size=(img_size, img_size), mode='bilinear', align_corners=False)
        upscaled_map = upscaled_map.squeeze().numpy()

        plt.figure(figsize=(8, 8))
        plt.imshow(image, extent=(0, img_size, 0, img_size))
        plt.imshow(upscaled_map, cmap='jet', alpha=0.3, extent=(0, img_size, 0, img_size))
        plt.axis('off')
        avg_save_path = os.path.join(output_folder, f"{base_name}_avg.png")
        plt.savefig(avg_save_path, bbox_inches='tight')
        avg_pdf_path = os.path.join(output_folder, f"{base_name}_avg.pdf")
        Image.open(avg_save_path).convert('RGB').save(avg_pdf_path, "PDF", resolution=100.0)
        plt.close()

        # 8. 전체 헤드/쿼리 Max 어텐션 맵 저장 (_max.png) [논문용 추천]
        # 평균 대신 가장 강한 신호를 잡기 위해 Max 연산 사용
        # attention_maps: [num_heads, num_queries, num_patches]
        layer_max_map = attention_maps.max(dim=0)[0].max(dim=0)[0] # [num_patches]

        attn_map_2d = layer_max_map.view(1, 1, num_patches_per_side, num_patches_per_side)
        upscaled_map = F.interpolate(attn_map_2d, size=(img_size, img_size), mode='bilinear', align_corners=False)
        upscaled_map = upscaled_map.squeeze().numpy()

        plt.figure(figsize=(8, 8))
        plt.imshow(image, extent=(0, img_size, 0, img_size))
        plt.imshow(upscaled_map, cmap='jet', alpha=0.3, extent=(0, img_size, 0, img_size))
        plt.axis('off')
        max_save_path = os.path.join(output_folder, f"{base_name}_max.png")
        plt.savefig(max_save_path, bbox_inches='tight')
        plt.close()

    except Exception as e:
        logging.error(f"어텐션 맵 시각화 중 오류 발생: {e}")