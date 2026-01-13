import logging
import os
import time
import numpy as np
import torch
from torch.utils.data import Subset
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import threading
import gc

try:
    import psutil
except ImportError:
    psutil = None

try:
    import onnxruntime
    from onnxruntime.quantization import CalibrationDataReader
except ImportError:
    onnxruntime = None
    CalibrationDataReader = object

try:
    from thop import profile
except ImportError:
    profile = None

# [추가] SCI급 논문용 정밀 메모리 측정 클래스 (Context Manager)
class MemoryMonitor:
    def __init__(self, interval=0.001):
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_memory = 0
        self.start_memory = 0
        self.process = psutil.Process(os.getpid()) if psutil else None

    def __enter__(self):
        if not self.process: return self
        # 시작 시점 메모리 (Baseline)
        self.start_memory = self.process.memory_info().rss / (1024 * 1024)
        self.peak_memory = self.start_memory
        # 모니터링 스레드 시작
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.process: return
        self.stop_event.set()
        self.thread.join()

    def _monitor(self):
        while not self.stop_event.is_set():
            try:
                current_mem = self.process.memory_info().rss / (1024 * 1024)
                if current_mem > self.peak_memory:
                    self.peak_memory = current_mem
            except:
                pass
            time.sleep(self.interval)

def evaluate_onnx(run_cfg, onnx_session, data_loader, desc="Evaluating ONNX", class_names=None, log_class_metrics=False):
    """ONNX 모델을 평가하고 정확도, 정밀도, 재현율, F1 점수를 로깅합니다."""
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    show_log = getattr(run_cfg, 'show_log', True)
    progress_bar = tqdm(data_loader, desc=desc, leave=False, disable=not show_log)

    input_info = onnx_session.get_inputs()[0]
    input_name = input_info.name
    input_type = input_info.type # 예: 'tensor(float)' or 'tensor(float16)'

    for images, labels, _ in progress_bar:
        images_np = images.cpu().numpy()
        # 모델이 FP16 입력을 기대하면 형변환 수행
        if 'float16' in input_type:
            images_np = images_np.astype(np.float16)
            
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
    
    input_info = onnx_session.get_inputs()[0]
    input_name = input_info.name
    input_type = input_info.type

    # 배치 입력에서 첫 번째 이미지만을 사용하여 단일 샘플 추론 시간을 측정합니다.
    single_dummy_input_np = dummy_input[0].unsqueeze(0).cpu().numpy()

    if 'float16' in input_type:
        single_dummy_input_np = single_dummy_input_np.astype(np.float16)

    # [추가] 정확한 메모리 측정을 위해 가비지 컬렉션 수행
    gc.collect()
    
    # [수정] MemoryMonitor를 사용하여 Warm-up 및 추론 루프 전체의 피크 메모리 측정
    with MemoryMonitor(interval=0.0001) as mem_mon:
        # CPU 시간 측정을 위한 예열(warm-up)
        for _ in range(10):
            _ = onnx_session.run(None, {input_name: single_dummy_input_np})

        # 실제 시간 측정
        num_iterations = 100
        iteration_times = []
        
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            _ = onnx_session.run(None, {input_name: single_dummy_input_np})
            end_time = time.perf_counter()
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
    
    # [추가] 추론 중 메모리 사용량 로깅
    peak_mem = 0
    if psutil:
        logging.info(f"[Inference Only] ONNX 런타임 추론 중 최대 CPU 메모리: {mem_mon.peak_memory:.2f} MB (순수 증가량: {mem_mon.peak_memory - mem_mon.start_memory:.2f} MB)")
        peak_mem = mem_mon.peak_memory
    else:
        logging.warning("psutil 모듈이 없어 메모리 측정을 수행하지 못했습니다.")
    return peak_mem

def measure_cpu_peak_memory_during_inference(session, data_loader, device):
    """ONNX 모델 추론 중 CPU 최대 메모리 사용량(RSS)을 단일 샘플 기준으로 측정합니다."""
    if not psutil:
        logging.warning("CPU 메모리 사용량을 측정하려면 'pip install psutil'을 실행해주세요.")
        return

    process = psutil.Process(os.getpid())
    
    try:
        # 데이터 로더에서 단일 샘플을 가져옵니다.
        dummy_input, _, _ = next(iter(data_loader))
        single_dummy_input_np = dummy_input[0].unsqueeze(0).cpu().numpy()
        input_info = session.get_inputs()[0]
        input_name = input_info.name
        input_type = input_info.type
        
        # FP16 입력 대응
        if 'float16' in input_type:
            single_dummy_input_np = single_dummy_input_np.astype(np.float16)
            
    except Exception as e:
        logging.error(f"ONNX 메모리 측정을 위한 더미 데이터 생성 중 오류 발생: {e}")
        return

    # [추가] 정확한 메모리 측정을 위해 가비지 컬렉션 수행
    gc.collect()
    
    with MemoryMonitor(interval=0.0001) as mem_mon:
        # Warm-up (10회)
        logging.info("ONNX CPU 메모리 측정을 위한 예열(warm-up)을 시작합니다 (단일 샘플 x 10회).")
        for _ in range(10):
            session.run(None, {input_name: single_dummy_input_np})

        logging.info("="*50)
        logging.info("ONNX 모델 추론 중 CPU 최대 메모리 사용량 측정을 시작합니다 (단일 샘플 x 100회 반복).")
        
        # 실제 측정 (100회 반복)
        num_iterations = 100
        for _ in range(num_iterations):
            session.run(None, {input_name: single_dummy_input_np})

    logging.info(f"  - 추론 전 기본 CPU 메모리: {mem_mon.start_memory:.2f} MB")
    logging.info(f"  - 추론 중 최대 CPU 메모리 (Peak): {mem_mon.peak_memory:.2f} MB")
    logging.info(f"  - 추론으로 인한 순수 메모리 증가량: {(mem_mon.peak_memory - mem_mon.start_memory):.2f} MB")
    logging.info("="*50)
    return mem_mon.peak_memory

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

# =============================================================================
# [추가] ONNX Static Quantization을 위한 Data Reader 클래스
# =============================================================================
class ONNXCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_loader, input_name):
        self.data_loader = data_loader
        self.iter = iter(data_loader)
        self.input_name = input_name

    def get_next(self):
        try:
            batch = next(self.iter)
            # data_loader가 (images, labels, filenames)를 반환한다고 가정
            images, _, _ = batch
            # ONNX Runtime은 numpy array를 기대합니다.
            return {self.input_name: images.numpy()}
        except StopIteration:
            return None
    
    def rewind(self):
        self.iter = iter(self.data_loader)