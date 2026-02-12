import os
import gc
import platform
import ctypes
import threading
import time
import logging
import torch
from torch.utils.data import Subset

try:
    import psutil
except ImportError:
    psutil = None

try:
    from thop import profile
except ImportError:
    profile = None

def flush_memory():
    """가비지 컬렉션 및 malloc_trim을 수행하여 메모리를 정리합니다."""
    gc.collect()
    if platform.system() == 'Linux':
        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except Exception:
            pass

class MemoryMonitor:
    def __init__(self, interval=0.001):
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_memory = 0
        self.start_memory = 0
        self.process = psutil.Process(os.getpid()) if psutil else None

    def __enter__(self):
        if not self.process: return self
        self.start_memory = self.process.memory_info().rss / (1024 * 1024)
        self.peak_memory = self.start_memory
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

def measure_model_flops(model, device, data_loader):
    """모델의 연산량(FLOPs)을 측정합니다."""
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