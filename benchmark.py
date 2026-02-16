import os
import time
import yaml
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
import psutil
import threading
import csv
from types import SimpleNamespace
from copy import deepcopy

# 기존 모듈 임포트
from model import Model as DecoderBackbone, Encoder, Classifier, HybridModel
# baseline.py에서 필요한 함수 임포트
from baseline import create_baseline_model, patch_timm_model_for_pruning, run_torch_pruning

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MemoryMonitor(threading.Thread):
    """
    CPU 메모리(RSS)를 별도 스레드에서 주기적으로 모니터링하는 클래스.
    SCIE 논문(main.tex)에 언급된 방식대로 psutil을 사용하여 측정.
    """
    def __init__(self, interval=0.001): # 1ms 간격
        super().__init__()
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.peak_rss = 0
        self.running = False
        self.start_rss = 0

    def run(self):
        self.running = True
        self.start_rss = self.process.memory_info().rss
        self.peak_rss = self.start_rss
        while self.running:
            current_rss = self.process.memory_info().rss
            self.peak_rss = max(self.peak_rss, current_rss)
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        return (self.peak_rss - self.start_rss) / 1024 / 1024 # MB 단위 반환

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def prepare_model(args, config, device):
    """설정과 가중치 파일을 기반으로 모델을 생성하고 로드합니다."""
    
    # 1. Config 파싱
    run_cfg = SimpleNamespace(**config['run'])
    model_cfg = SimpleNamespace(**config['model'])
    baseline_cfg = SimpleNamespace(**config.get('baseline', {}))
    
    # 2. 모델 아키텍처 생성
    if args.model_type == 'main':
        logging.info("Creating Proposed HybridModel...")
        # main.py의 모델 생성 로직 복원
        num_patches_per_side = model_cfg.num_patches_per_side
        num_encoder_patches = num_patches_per_side ** 2
        
        # num_labels는 config에 없으므로 데이터셋 정보 기반 추론 혹은 기본값 2 (Binary)
        num_labels = 2 

        decoder_params = {
            'num_encoder_patches': num_encoder_patches,
            'num_patches_h': num_patches_per_side,
            'num_patches_w': num_patches_per_side,
            'num_labels': num_labels,
            'num_decoder_layers': model_cfg.num_decoder_layers,
            'num_decoder_patches': model_cfg.num_decoder_patches,
            'encoder_dim': model_cfg.encoder_dim,
            'adaptive_initial_query': getattr(model_cfg, 'adaptive_initial_query', False),
            'emb_dim': model_cfg.emb_dim,
            'num_heads': model_cfg.num_heads,
            'decoder_ff_ratio': model_cfg.decoder_ff_ratio,
            'dropout': model_cfg.dropout,
            'positional_encoding': model_cfg.positional_encoding,
            'visualize_attention': False,
            'drop_path_ratio': getattr(model_cfg, 'drop_path_ratio', 0.0),
        }
        decoder_args = SimpleNamespace(**decoder_params)

        encoder = Encoder(
            num_patches_per_side=model_cfg.num_patches_per_side,
            encoder_dim=model_cfg.encoder_dim,
            cnn_feature_extractor_name=model_cfg.cnn_feature_extractor['name'],
            pre_trained=False, # 추론 시에는 가중치를 로드하므로 False
        )
        decoder = DecoderBackbone(args=decoder_args)
        classifier = Classifier(
            num_decoder_patches=model_cfg.num_decoder_patches,
            emb_dim=model_cfg.emb_dim,
            num_labels=num_labels,
            dropout=model_cfg.dropout,
        )
        model = HybridModel(encoder, decoder, classifier)

    elif args.model_type == 'baseline':
        model_name = args.baseline_name if args.baseline_name else baseline_cfg.model_name
        logging.info(f"Creating Baseline Model: {model_name}...")
        model = create_baseline_model(model_name, num_labels=2, pretrained=False)
        model = patch_timm_model_for_pruning(model, model_name, device)
        
        # Pruning 정보 확인 및 적용
        model_dir = os.path.dirname(args.model_path)
        pruning_info_path = os.path.join(model_dir, 'pruning_info.yaml')
        
        # pruning_info.yaml이 있거나 config에 pruning 설정이 켜져있으면 적용
        use_pruning = getattr(baseline_cfg, 'use_l1_pruning', False) or \
                      getattr(baseline_cfg, 'use_fpgm_pruning', False) or \
                      getattr(baseline_cfg, 'use_wanda_pruning', False) or \
                      os.path.exists(pruning_info_path)

        if use_pruning:
            logging.info("Applying Pruning structure...")
            # pruning_info.yaml 우선 적용
            if os.path.exists(pruning_info_path):
                with open(pruning_info_path, 'r') as f:
                    p_info = yaml.safe_load(f)
                if 'optimal_sparsity' in p_info:
                    baseline_cfg.pruning_sparsity = p_info['optimal_sparsity']
                    logging.info(f"Loaded sparsity from file: {baseline_cfg.pruning_sparsity}")

            # Pruning 적용 (dummy input 필요)
            criterion = nn.CrossEntropyLoss() # 임시
            model = model.to(device)
            model = run_torch_pruning(model, baseline_cfg, model_cfg, device, train_loader=None, criterion=criterion)
    
    else:
        raise ValueError("Unknown model type. Use 'main' or 'baseline'.")

    # 3. 가중치 로드
    model = model.to(device)
    if args.model_path and os.path.exists(args.model_path):
        logging.info(f"Loading weights from {args.model_path}")
        try:
            state_dict = torch.load(args.model_path, map_location=device)
            # thop 등으로 인해 추가된 불필요한 키 제거
            clean_state_dict = {k: v for k, v in state_dict.items() if not k.endswith(('total_ops', 'total_params'))}
            model.load_state_dict(clean_state_dict, strict=False)
        except Exception as e:
            logging.error(f"Failed to load weights: {e}")
            exit(1)
    else:
        logging.warning(f"Model path not found: {args.model_path}. Using random weights.")

    model.eval()
    
    # CPU 추론 최적화 (Channels Last)
    if device.type == 'cpu':
        model = model.to(memory_format=torch.channels_last)
        
    return model, model_cfg.img_size

def benchmark(args, model, img_size, device, num_warmup=50, num_runs=1000):
    """SCIE급 논문 프로토콜에 따른 벤치마크 수행"""
    
    # 1. Dummy Input 준비
    input_tensor = torch.randn(1, 3, img_size, img_size).to(device)
    if device.type == 'cpu':
        input_tensor = input_tensor.to(memory_format=torch.channels_last)

    logging.info(f"Starting Benchmark: Device={device}, Warmup={num_warmup}, Runs={num_runs}")

    # 2. Warm-up (예열)
    # 캐시 워밍업 및 JIT 컴파일 등을 위해 측정 없이 실행
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 3. Latency 측정 (속도)
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = model(input_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            timings.append((end_time - start_time) * 1000) # ms 단위 변환

    timings = np.array(timings)
    avg_latency = np.mean(timings)
    std_latency = np.std(timings)
    p50 = np.percentile(timings, 50)
    p95 = np.percentile(timings, 95)
    p99 = np.percentile(timings, 99)
    fps = 1000 / avg_latency

    # 4. Peak Memory 측정
    # GPU: torch.cuda.max_memory_allocated 사용
    # CPU: psutil을 이용한 백그라운드 스레드 모니터링 (main.tex 방식)
    peak_mem_mb = 0.0
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(input_tensor)
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        peak_mem_mb = peak_mem_bytes / 1024 / 1024
    else:
        # CPU Memory Measurement
        monitor = MemoryMonitor(interval=0.0001) # 0.1ms 간격으로 타이트하게 측정
        monitor.start()
        with torch.no_grad():
            # 메모리 피크를 잡기 위해 여러 번 실행
            for _ in range(50): 
                _ = model(input_tensor)
        peak_mem_mb = monitor.stop()

    # 결과 출력
    print("="*60)
    print(f"Model Benchmark Results ({device.type.upper()})")
    print("-" * 60)
    print(f"Input Size       : (1, 3, {img_size}, {img_size})")
    print(f"Latency (Mean)   : {avg_latency:.4f} ms ± {std_latency:.4f}")
    print(f"Latency (P50)    : {p50:.4f} ms")
    print(f"Latency (P95)    : {p95:.4f} ms")
    print(f"Latency (P99)    : {p99:.4f} ms")
    print(f"Throughput       : {fps:.2f} FPS")
    print(f"Peak Memory      : {peak_mem_mb:.2f} MB")
    print("="*60)

    # CSV 저장
    csv_file = 'benchmark_results.csv'
    file_exists = os.path.isfile(csv_file)
    
    model_name = "Proposed" if args.model_type == 'main' else args.baseline_name
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Model', 'Device', 'Input Size', 'Latency(ms)', 'Std(ms)', 'P50(ms)', 'P95(ms)', 'P99(ms)', 'FPS', 'Peak Memory(MB)'])
        
        writer.writerow([model_name, device.type, f"{img_size}x{img_size}", 
                         f"{avg_latency:.4f}", f"{std_latency:.4f}", f"{p50:.4f}", f"{p95:.4f}", f"{p99:.4f}", f"{fps:.2f}", f"{peak_mem_mb:.2f}"])
    logging.info(f"Benchmark results saved to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCIE-level Benchmark Tool")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth file')
    parser.add_argument('--model_type', type=str, required=True, choices=['main', 'baseline'], help='Model type: main (Hybrid) or baseline')
    parser.add_argument('--baseline_name', type=str, default=None, help='Name of baseline model (e.g., efficientnet_b0). Required if model_type is baseline.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run benchmark (cuda or cpu)')
    parser.add_argument('--warmup', type=int, default=50, help='Number of warmup runs')
    parser.add_argument('--runs', type=int, default=1000, help='Number of measurement runs')

    args = parser.parse_args()

    # Config 로드
    config = load_config(args.config)
    
    # Device 설정
    device = torch.device(args.device)
    
    # 모델 준비
    model, img_size = prepare_model(args, config, device)
    
    # 벤치마크 실행
    benchmark(args, model, img_size, device, num_warmup=args.warmup, num_runs=args.runs)



# # main 측정 시
# python benchmark.py --config config.yaml --model_path "./log/Sewer-ML/original/main_xxx/best_model.pth" --model_type main

# # baseline 측정 시
# python benchmark.py --config config.yaml --model_path "./pretrained/baselines/original/efficientnet_b0/best_model.pth" --model_type baseline --baseline_name efficientnet_b0

# # CPU 환경 측정 시
# python benchmark.py --config config.yaml --model_path "..." --model_type main --device cpu
# python benchmark.py --config config.yaml --model_path "..." --model_type baseline --device cpu
