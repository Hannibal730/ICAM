import logging
import time
import numpy as np
import torch
from torch.utils.data import Subset
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

try:
    from thop import profile
except ImportError:
    profile = None

def evaluate_onnx(run_cfg, onnx_session, data_loader, desc="Evaluating ONNX", class_names=None, log_class_metrics=False):
    """ONNX 모델을 평가하고 정확도, 정밀도, 재현율, F1 점수를 로깅합니다."""
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    show_log = getattr(run_cfg, 'show_log', True)
    progress_bar = tqdm(data_loader, desc=desc, leave=False, disable=not show_log)

    input_name = onnx_session.get_inputs()[0].name

    for images, labels, _ in progress_bar:
        images_np = images.cpu().numpy()
        outputs = onnx_session.run(None, {input_name: images_np})[0]
        predicted = np.argmax(outputs, axis=1)

        total += labels.size(0)
        correct += (predicted == labels.cpu().numpy()).sum()
        all_preds.extend(predicted)
        all_labels.extend(labels.cpu().numpy())

    if total == 0:
        logging.warning("ONNX 평가 데이터가 없습니다. 평가를 건너뜁니다.")
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'labels': [], 'preds': []}

    accuracy = 100 * correct / total
    
    acc_label = "Test Acc (ONNX)"
    log_message = f'{desc} | {acc_label}: {accuracy:.2f}%'
    logging.info(log_message)

    if log_class_metrics and class_names:
        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        for i, class_name in enumerate(class_names):
            log_line = (f"[Metrics for '{class_name}' (ONNX)] | "
                        f"Precision: {precision_per_class[i]:.4f} | "
                        f"Recall: {recall_per_class[i]:.4f} | "
                        f"F1: {f1_per_class[i]:.4f}")
            logging.info(log_line)

    return {
        'accuracy': accuracy,
        'loss': -1,
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_per_class': f1_per_class if log_class_metrics and class_names else None,
        'labels': all_labels,
        'preds': all_preds
    }

def measure_onnx_performance(onnx_session, dummy_input):
    """ONNX 모델의 Forward Pass 시간 및 메모리 사용량을 측정합니다."""
    logging.info("ONNX 런타임의 샘플 당 Forward Pass 시간 측정을 시작합니다...")
    
    input_name = onnx_session.get_inputs()[0].name
    # 배치 입력에서 첫 번째 이미지만을 사용하여 단일 샘플 추론 시간을 측정합니다.
    single_dummy_input_np = dummy_input[0].unsqueeze(0).cpu().numpy()

    # CPU 시간 측정을 위한 예열(warm-up)
    for _ in range(10):
        _ = onnx_session.run(None, {input_name: single_dummy_input_np})

    # 실제 시간 측정
    num_iterations = 100
    iteration_times = []
    for _ in range(num_iterations):
        start_time = time.time()
        _ = onnx_session.run(None, {input_name: single_dummy_input_np})
        end_time = time.time()
        iteration_times.append((end_time - start_time) * 1000) # ms

    # 단일 이미지 추론을 반복했으므로, 총 시간을 반복 횟수로 나누면 샘플 당 평균 시간이 됩니다.
    avg_inference_time_per_sample = np.mean(iteration_times)
    std_inference_time_per_sample = np.std(iteration_times)
    
    # FPS 계산 및 통계
    fps_per_iteration = [1000 / t for t in iteration_times if t > 0]
    avg_fps = np.mean(fps_per_iteration) if fps_per_iteration else 0
    std_fps = np.std(fps_per_iteration) if fps_per_iteration else 0

    logging.info(f"샘플 당 평균 Forward Pass 시간 (ONNX, CPU): {avg_inference_time_per_sample:.2f}ms (std: {std_inference_time_per_sample:.2f}ms)")
    logging.info(f"샘플 당 평균 FPS (ONNX, CPU): {avg_fps:.2f} FPS (std: {std_fps:.2f}) (1개 샘플 x {num_iterations}회 반복)")
    logging.info("ONNX 런타임의 CPU 메모리 사용량 측정은 지원되지 않습니다.")

def measure_model_flops(model, device, data_loader):
    """모델의 연산량(FLOPs)을 측정합니다."""
    gflops_per_sample = 0.0
    try:
        if isinstance(data_loader.dataset, Subset):
            sample_image, _, _ = data_loader.dataset.dataset[0]
        else:
            sample_image, _, _ = data_loader.dataset[0]
        dummy_input = sample_image.unsqueeze(0).to(device)

        if profile:
            model.eval()
            macs, params = profile(model, inputs=(dummy_input,), verbose=False)
            gmacs = macs / 1e9
            gflops_per_sample = (macs * 2) / 1e9
            logging.info(f"연산량 (MACs): {gmacs:.4f} GMACs per sample")
            logging.info(f"연산량 (FLOPs): {gflops_per_sample:.4f} GFLOPs per sample")
        else:
            logging.info("연산량 (FLOPs): N/A (thop 라이브러리 미설치)")
        
        return dummy_input

    except Exception as e:
        logging.error(f"FLOPS 측정 중 오류 발생: {e}")
        return None