import os
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from types import SimpleNamespace
import pandas as pd

import argparse
import yaml
import logging
from datetime import datetime
import random
import time 
import gc
import copy
from models import Model as DecoderBackbone, PatchConvEncoder, Classifier, HybridModel

try:
    import cpuinfo
except ImportError:
    cpuinfo = None

try:
    import psutil
except ImportError:
    psutil = None

from dataloader import prepare_data # 데이터 로딩 함수 임포트

try:
    from thop import profile
except ImportError:
    profile = None

try:
    import onnxruntime
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
    from onnxruntime.quantization.preprocess import quant_pre_process
    import onnx
    from onnxconverter_common import float16
except ImportError:
    onnxruntime = None
    quantize_static = None
    quant_pre_process = None
    CalibrationMethod = None
    onnx = None
    float16 = None

# [추가] Static Quantization을 위한 라이브러리 임포트
try:
    from torch.ao.quantization import quantize_fx, get_default_qconfig_mapping
except ImportError:
    quantize_fx = None
    get_default_qconfig_mapping = None

from onnx_utils import evaluate_onnx, measure_onnx_performance, measure_model_flops, ONNXCalibrationDataReader, MemoryMonitor, flush_memory

from plot import plot_and_save_train_val_accuracy_graph, plot_and_save_val_accuracy_graph, plot_and_save_confusion_matrix, plot_and_save_attention_maps, plot_and_save_f1_normal_graph, plot_and_save_loss_graph, plot_and_save_lr_graph, plot_and_save_compiled_graph

# =============================================================================
# 1. 로깅 설정
# =============================================================================
def setup_logging(run_cfg, data_dir_name, run_name_suffix=None):
    """로그 파일을 log 폴더에 생성하고, 콘솔에도 함께 출력하도록 설정합니다."""
    show_log = getattr(run_cfg, 'show_log', True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not show_log:
        # 로깅을 완전히 비활성화합니다.
        logging.disable(logging.CRITICAL)
        # 임시 디렉토리 경로를 반환하지만, 실제 생성은 하지 않습니다.
        # 훈련 모드에서 모델 저장을 위해 현재 디렉토리('.')를 사용합니다.
        return '.', timestamp

    # 각 실행을 위한 고유한 디렉토리 생성
    run_dir_name = f"main_{timestamp}"
    if run_name_suffix:
        run_dir_name = f"{run_dir_name}_{run_name_suffix}"
    run_dir_path = os.path.join("log", data_dir_name, run_dir_name)
    os.makedirs(run_dir_path, exist_ok=True)
    
    # 로그 파일 경로 설정
    log_filename = os.path.join(run_dir_path, f"log_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True # 핸들러를 다시 설정하기 위해 필요
    )
    logging.info(f"로그 파일이 '{log_filename}'에 저장됩니다.")
    return run_dir_path, timestamp

# =============================================================================
# 2. 훈련 및 평가 함수
# =============================================================================
def log_model_parameters(model):
    """모델의 구간별 및 총 파라미터 수를 계산하고 로깅합니다."""
    
    def count_parameters(m):
        if m is None:
            return 0
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    # Encoder 내부를 세분화하여 파라미터 계산
    # 1. Encoder (PatchConvEncoder) 내부 파라미터 계산
    cnn_feature_extractor = model.encoder.shared_conv[0]
    conv_front_params = count_parameters(cnn_feature_extractor.conv_front)
    conv_1x1_params = count_parameters(cnn_feature_extractor.conv_1x1)
    encoder_norm_params = count_parameters(model.encoder.norm)
    encoder_total_params = conv_front_params + conv_1x1_params + encoder_norm_params

    # 2. Decoder (DecoderBackbone) 내부 파라미터 계산
    embedding_module = model.decoder.embedding4decoder

    # Positional Encoding 파라미터 계산
    pe_params = 0
    if hasattr(embedding_module, 'pos_embed') and isinstance(embedding_module.pos_embed, torch.nn.Parameter) and embedding_module.pos_embed.requires_grad:
        pe_params = embedding_module.pos_embed.numel()
    
    # Learnable Query 파라미터 계산
    query_params = 0
    if hasattr(embedding_module, 'learnable_queries') and embedding_module.learnable_queries.requires_grad:
        query_params = embedding_module.learnable_queries.numel()
    
    w_feat2emb_params = count_parameters(embedding_module.W_feat2emb)
    
    w_k_init_params = 0
    w_v_init_params = 0
    if hasattr(embedding_module, 'W_K_init'):
        w_k_init_params = count_parameters(embedding_module.W_K_init)
    if hasattr(embedding_module, 'W_V_init'):
        w_v_init_params = count_parameters(embedding_module.W_V_init)

    # Embedding4Decoder의 파라미터 총합 (내부 Decoder 레이어 제외)
    embedding4decoder_total_params = w_feat2emb_params + w_k_init_params + w_v_init_params + query_params + pe_params

    # Decoder 내부의 트랜스포머 레이어 파라미터 계산
    decoder_layers_params = count_parameters(model.decoder.embedding4decoder.decoder)
    decoder_total_params = embedding4decoder_total_params + decoder_layers_params

    # 3. Classifier (MLP) 파라미터 계산
    classifier_projection_params = count_parameters(model.classifier.projection)
    classifier_total_params = classifier_projection_params

    total_params = encoder_total_params + decoder_total_params + classifier_total_params

    logging.info("="*50)
    logging.info(f"모델 파라미터 수: {total_params:,} 개")
    logging.info(f"  - Encoder (PatchConvEncoder):         {encoder_total_params:,} 개")
    logging.info(f"    - conv_front (CNN Backbone):        {conv_front_params:,} 개")
    logging.info(f"    - 1x1_conv (Channel Proj):          {conv_1x1_params:,} 개")
    logging.info(f"    - norm (LayerNorm):                 {encoder_norm_params:,} 개")
    logging.info(f"  - Decoder (Cross-Attention-based):    {decoder_total_params:,} 개")
    logging.info(f"    - Embedding Layer (W_feat2emb):     {w_feat2emb_params:,} 개")
    logging.info(f"    - Init Key Proj (W_K_init):         {w_k_init_params:,} 개")
    logging.info(f"    - Init Value Proj (W_V_init):       {w_v_init_params:,} 개")
    logging.info(f"    - Learnable Queries:                {query_params:,} 개")
    # logging.info(f"    - Positional Encoding (learnable):  {pe_params:,} 개")
    logging.info(f"    - Decoder Layers:                   {decoder_layers_params:,} 개")
    logging.info(f"  - Classifier (Projection MLP):        {classifier_total_params:,} 개")

def evaluate(run_cfg, model, data_loader, device, criterion, loss_function_name, desc="Evaluating", class_names=None, log_class_metrics=False):
    """모델을 평가하고 정확도, 정밀도, 재현율, F1 점수를 로깅합니다."""
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    show_log = getattr(run_cfg, 'show_log', True)
    progress_bar = tqdm(data_loader, desc=desc, leave=False, disable=not show_log)
    with torch.no_grad():
        for images, labels, _ in progress_bar: # 파일명은 사용하지 않으므로 _로 받음
            images, labels = images.to(device), labels.to(device)

            outputs = model(images) # [B, num_labels]

            if loss_function_name == 'bcewithlogitsloss':
                loss = criterion(outputs[:, 1].unsqueeze(1), labels.float().unsqueeze(1))
            else: # crossentropyloss
                loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if total == 0:
        logging.warning("테스트 데이터가 없습니다. 평가를 건너뜁니다.")
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'loss': float('inf'), 'labels': [], 'preds': []}

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # desc 내용에 따라 Accuracy 라벨을 동적으로 변경
    if desc.startswith("[Valid]"):
        acc_label = "Val Acc"
        log_message = f'{desc} | Loss: {avg_loss:.4f} | {acc_label}: {accuracy:.2f}%'
    else: # [Test] 또는 [Inference]의 경우
        acc_label = "Test Acc"
        log_message = f'{desc} Loss: {avg_loss:.4f} | {acc_label}: {accuracy:.2f}%'
    logging.info(log_message)

    # 클래스별 상세 지표 로깅
    if log_class_metrics and class_names:
        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        # logging.info("-" * 30)
        for i, class_name in enumerate(class_names):
            log_line = (f"[Metrics for '{class_name}'] | "
                        f"Precision: {precision_per_class[i]:.4f} | "
                        f"Recall: {recall_per_class[i]:.4f} | "
                        f"F1: {f1_per_class[i]:.4f}")
            logging.info(log_line)
        # logging.info("-" * 30)

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'f1_macro': f1, # 평균 F1 점수
        'f1_per_class': f1_per_class if log_class_metrics and class_names else None, # 클래스별 F1 점수
        'labels': all_labels,
        'preds': all_preds
    }

def train(run_cfg, train_cfg, model, optimizer, scheduler, train_loader, valid_loader, device, run_dir_path, class_names, pos_weight):
    """모델 훈련 및 검증을 수행하고 최고 성능 모델을 저장합니다."""
    logging.info("="*50)
    logging.info("train 모드를 시작합니다.")
    
    # 모델 저장 경로를 실행별 디렉토리로 설정
    model_path = os.path.join(run_dir_path, run_cfg.pth_best_name)

    loss_function_name = getattr(train_cfg, 'loss_function', 'CrossEntropyLoss').lower()
    if loss_function_name == 'bcewithlogitsloss':
        # BCEWithLogitsLoss는 [B, 1] 형태의 출력을 기대하므로 모델의 마지막 레이어 수정이 필요할 수 있습니다.
        # 이 코드에서는 num_labels=2를 가정하고, 출력을 [B, 2]에서 [B, 1]로 변환하여 사용합니다.
        if model.classifier.projection[-1].out_features != 2:
            logging.warning(f"BCEWithLogitsLoss는 이진 분류(num_labels=2)에 최적화되어 있습니다. 현재 num_labels={model.classifier.projection[-1].out_features}")

        weight_value = getattr(train_cfg, 'bce_pos_weight', None)
        if weight_value == 'auto':
            final_pos_weight = pos_weight.to(device) if pos_weight is not None else None
        else:
            final_pos_weight = torch.tensor(float(weight_value), dtype=torch.float).to(device) if weight_value is not None else None
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=final_pos_weight)
        logging.info(f"손실 함수: BCEWithLogitsLoss (pos_weight: {final_pos_weight.item() if final_pos_weight is not None else 'None'})")
    elif loss_function_name == 'crossentropyloss':
        label_smoothing = getattr(train_cfg, 'label_smoothing', 0.0)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        logging.info(f"손실 함수: CrossEntropyLoss (label_smoothing: {label_smoothing})")
    else:
        raise ValueError(f"run.py에서 지원하지 않는 손실 함수입니다: {loss_function_name}")

    # --- 손실 함수 설정 ---
    best_model_criterion = getattr(train_cfg, 'best_model_criterion', 'F1_average')
    best_metric = 0.0 if best_model_criterion != 'val_loss' else float('inf')


    for epoch in range(train_cfg.epochs):
        # 에포크 시작 시 구분을 위한 라인 추가
        logging.info("-" * 50)

        # 에포크 시작 시 Learning Rate 로깅
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"[LR]    [{epoch+1}/{train_cfg.epochs}] | Learning Rate: {current_lr:.6f}")

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.epochs} [Training]", leave=False, disable=not getattr(run_cfg, 'show_log', True))


        for images, labels, _ in progress_bar: # 파일명은 사용하지 않으므로 _로 받음
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()

            outputs = model(images)
            if loss_function_name == 'bcewithlogitsloss':
                # BCEWithLogitsLoss는 [B, 1] 형태의 출력을 기대합니다.
                # outputs: [B, 2] -> [B, 1] (Defect 클래스에 대한 로짓만 사용)
                loss = criterion(outputs[:, 1].unsqueeze(1), labels.float().unsqueeze(1))
            else: # crossentropyloss
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # tqdm 프로그레스 바에 현재 loss 표시
            step_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{step_lr:.6f}")

        train_acc = 100 * correct / total
        logging.info(f'[Train] [{epoch+1}/{train_cfg.epochs}] | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%')
        
        # --- 평가 단계 ---
        # 클래스별 F1 점수를 계산하고 로깅하도록 옵션 전달
        eval_results = evaluate(run_cfg, model, valid_loader, device, criterion, loss_function_name, desc=f"[Valid] [{epoch+1}/{train_cfg.epochs}]", class_names=class_names, log_class_metrics=True)

        # --- 최고 성능 모델 저장 기준 선택 ---
        current_metric = 0.0
        if best_model_criterion == 'val_loss':
            current_metric = eval_results['loss']
            is_best = current_metric < best_metric
        else: # F1 score variants
            if best_model_criterion == 'F1_Normal' and eval_results['f1_per_class'] is not None:
                try:
                    normal_idx = class_names.index('Normal')
                    current_metric = eval_results['f1_per_class'][normal_idx]
                except (ValueError, IndexError):
                    logging.warning("best_model_criterion이 'F1_Normal'로 설정되었으나, 'Normal' 클래스를 찾을 수 없습니다. 대신 F1_macro를 사용합니다.")
                    current_metric = eval_results['f1_macro']
            elif best_model_criterion == 'F1_Defect' and eval_results['f1_per_class'] is not None:
                try:
                    defect_idx = class_names.index('Defect')
                    current_metric = eval_results['f1_per_class'][defect_idx]
                except (ValueError, IndexError):
                    logging.warning("best_model_criterion이 'F1_Defect'로 설정되었으나, 'Defect' 클래스를 찾을 수 없습니다. 대신 F1_macro를 사용합니다.")
                    current_metric = eval_results['f1_macro']
            else: # 'F1_average' or default
                current_metric = eval_results['f1_macro']
            is_best = current_metric > best_metric
        
        # 최고 성능 모델 저장
        if is_best:
            best_metric = current_metric
            torch.save(model.state_dict(), model_path)
            # 어떤 기준으로 저장되었는지 명확히 로그에 남깁니다.
            criterion_name = best_model_criterion.replace('_', ' ')
            logging.info(f"[Best Model Saved] ({criterion_name}: {best_metric:.4f}) -> '{model_path}'")
        
        # 스케줄러가 설정된 경우에만 step()을 호출
        if scheduler:
            scheduler.step()

def inference(run_cfg, model_cfg, model, data_loader, device, run_dir_path, timestamp, mode_name="Inference", class_names=None, output_dir=None):
    """저장된 모델을 불러와 추론 시 GPU 메모리 사용량을 측정하고, 테스트셋 성능을 평가합니다."""
    
    # 결과 저장 경로 설정 (output_dir이 주어지면 그곳에, 아니면 run_dir_path에 저장)
    save_dir = output_dir if output_dir else run_dir_path
    
    # --- ONNX 모델 직접 평가 분기 ---
    onnx_inference_path = getattr(run_cfg, 'onnx_inference_path', None)
    if onnx_inference_path and os.path.exists(onnx_inference_path):
        logging.info("="*50)
        logging.info(f"ONNX 모델 직접 평가를 시작합니다: '{onnx_inference_path}'")
        if not onnxruntime:
            logging.error("ONNX Runtime이 설치되지 않았습니다. 'pip install onnxruntime'으로 설치해주세요.")
            return None
        try:
            logging.info(f"ONNX Runtime (v{onnxruntime.__version__})으로 평가를 시작합니다.")
            
            flush_memory() # [수정] 정확한 측정을 위해 메모리 정리
            process = psutil.Process(os.getpid()) if psutil else None
            mem_before_load = process.memory_info().rss / (1024 * 1024) if process else 0

            with MemoryMonitor() as mem_mon:
                onnx_session = onnxruntime.InferenceSession(onnx_inference_path, providers=['CPUExecutionProvider'])
            if psutil:
                logging.info(f"[Model Load] ONNX 모델 로드 중 피크 메모리 - 모델 로드 전 청소 직후 메모리: {mem_mon.peak_memory - mem_mon.start_memory:.2f} MB")

            dummy_input, _, _ = next(iter(data_loader))
            inference_peak = measure_onnx_performance(onnx_session, dummy_input)
            if psutil and inference_peak:
                logging.info(f"[Total] 추론 중 피크 메모리 - 모델 로드 전 청소 직후 메모리: {inference_peak - mem_before_load:.2f} MB")
            evaluate_onnx(run_cfg, onnx_session, data_loader, desc=f"[{mode_name} (ONNX)]", class_names=class_names, log_class_metrics=True)
        except Exception as e:
            logging.error(f"ONNX 모델 평가 중 오류 발생: {e}")
        return None # ONNX 직접 평가 후 종료
    
    logging.info("="*50)
    logging.info(f"{mode_name} 모드를 시작합니다.")
    
    # 훈련 시 사용된 모델 경로를 불러옴
    model_path = os.path.join(run_dir_path, run_cfg.pth_best_name)
    if not os.path.exists(model_path) and mode_name != "Final Evaluation":
        logging.error(f"모델 파일('{model_path}')을 찾을 수 없습니다. 'train' 모드로 먼저 훈련을 실행했는지, 또는 'config.yaml'의 'pth_inference_dir' 설정이 올바른지 확인하세요.")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logging.info(f"'{model_path}' 가중치 로드 완료.") # PyTorch 로드 시점에는 별도 메모리 측정을 하지 않으므로 flush_memory 생략 가능하나, 필요시 추가
    except Exception as e:
        logging.error(f"모델 가중치 로딩 중 오류 발생: {e}")
        return

    model.eval()

    # [Optimization] CPU 추론 시 Channels Last 메모리 포맷 적용 (속도 향상 및 메모리 효율화)
    if device.type == 'cpu':
        model = model.to(memory_format=torch.channels_last)

    # --- PyTorch 모델 성능 지표 측정 (FLOPS 및 더미 입력 생성) ---
    dummy_input = measure_model_flops(model, device, data_loader)
    single_dummy_input = dummy_input[0].unsqueeze(0) if dummy_input.shape[0] > 1 else dummy_input
    
    # [Optimization] 입력 데이터도 Channels Last로 변환
    if device.type == 'cpu':
        single_dummy_input = single_dummy_input.to(memory_format=torch.channels_last)

    # --- 양자화(Quantization) 적용 ---
    use_fp16 = getattr(run_cfg, 'use_fp16_inference', False)
    use_int8 = getattr(run_cfg, 'use_int8_inference', False)

    if use_int8 and use_fp16:
        logging.error("use_int8_inference과 use_fp16_inference이 동시에 설정되었습니다. 둘 중 하나만 True로 설정해주세요.")
        raise ValueError("Conflicting quantization options: use_int8_inference and use_fp16_inference are both True.")

    if use_int8:
        logging.info("="*50)
        logging.info("INT8 Static Quantization (ONNX)을 적용합니다.")
        
        if not onnxruntime:
            logging.error("ONNX Runtime이 설치되지 않아 INT8 양자화를 수행할 수 없습니다.")
            return None

        # 1. FP32 ONNX 변환
        fp32_onnx_path = os.path.join(save_dir, f'model_fp32_for_quant.onnx')
        # Ensure CPU for export stability
        model.to('cpu')
        model.eval()
        
        # Dummy input for export
        dummy_input_cpu = single_dummy_input.to('cpu')
        
        torch.onnx.export(model, dummy_input_cpu, fp32_onnx_path,
                          export_params=True, opset_version=17,
                          do_constant_folding=True,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        
        # [추가] Quantization 전처리 (Graph Optimization & Shape Inference)
        preprocessed_onnx_path = os.path.join(save_dir, f'model_fp32_preprocessed.onnx')
        logging.info("Quantization 전처리를 수행합니다 (Fusion, Shape Inference)...")
        quant_pre_process(fp32_onnx_path, preprocessed_onnx_path)

        # 2. Calibration
        calib_samples = getattr(run_cfg, 'int8_calib_samples', 256)
        logging.info(f"Calibration 진행 ({calib_samples} samples)...")
        
        # Calibration Data Reader Setup
        seed = getattr(run_cfg, 'global_seed', 42)
        g = torch.Generator()
        g.manual_seed(seed if seed is not None else 42)
        
        dataset = data_loader.dataset
        total_len = len(dataset)
        num_samples = min(calib_samples, total_len)
        indices = torch.randperm(total_len, generator=g)[:num_samples].tolist()
        calib_subset = Subset(dataset, indices)
        calib_loader = DataLoader(calib_subset, batch_size=1, shuffle=False, num_workers=0, collate_fn=getattr(data_loader, 'collate_fn', None))
        
        dr = ONNXCalibrationDataReader(calib_loader, input_name='input')

        # 3. Quantize
        int8_onnx_path = os.path.join(save_dir, f'best_model_int8.onnx')

        calib_method_str = getattr(run_cfg, 'int8_calibration_method', 'MinMax').lower()
        
        # [추가] Activation Type 설정
        act_type_str = getattr(run_cfg, 'int8_activation_type', 'QInt8').lower()
        activation_type = QuantType.QUInt8 if act_type_str == 'quint8' else QuantType.QInt8

        # [추가] Extra Options
        extra_options = {}
        if calib_method_str == 'percentile':
            percentile_val = getattr(run_cfg, 'int8_percentile', 99.999)
            extra_options['Percentile'] = percentile_val

        if calib_method_str == 'entropy':
            calib_method = CalibrationMethod.Entropy
        elif calib_method_str == 'percentile':
            calib_method = CalibrationMethod.Percentile
        else:
            calib_method = CalibrationMethod.MinMax
        logging.info(f"Calibration Method: {calib_method_str} ({calib_method})")
        logging.info(f"Activation Type: {activation_type} (Extra Options: {extra_options})")

        quantize_static(
            model_input=preprocessed_onnx_path, # 전처리된 모델 사용
            model_output=int8_onnx_path,
            calibration_data_reader=dr,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=activation_type, # [추가]
            calibrate_method=calib_method,
            extra_options=extra_options # [추가]
        )

        logging.info(f"ONNX INT8 모델 저장 완료: {int8_onnx_path}")

        # 4. Evaluate
        # [수정] 모델 로드 메모리(Static Footprint)와 추론 메모리(Dynamic Overhead) 분리 측정
        flush_memory() # [수정]
        # [추가] 전체 메모리 증가량 계산을 위한 기준점 측정
        process = psutil.Process(os.getpid()) if psutil else None
        mem_before_load = process.memory_info().rss / (1024 * 1024) if process else 0

        with MemoryMonitor() as mem_mon:
            sess_options = onnxruntime.SessionOptions()
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            # [확인용] 최적화된 그래프를 파일로 저장 (Fusion 확인용)
            sess_options.optimized_model_filepath = os.path.join(save_dir, "model_graph_int8.onnx")
            onnx_session = onnxruntime.InferenceSession(int8_onnx_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
        
        if psutil:
            logging.info(f"[Model Load] ONNX 모델(INT8) 로드 중 피크 메모리 - 모델 로드 전 청소 직후 메모리: {mem_mon.peak_memory - mem_mon.start_memory:.2f} MB")

        inference_peak = measure_onnx_performance(onnx_session, dummy_input_cpu)
        if psutil and inference_peak:
             logging.info(f"[Total] (INT8) 추론 중 피크 메모리 - 모델 로드 전 청소 직후 메모리: {inference_peak - mem_before_load:.2f} MB")

        eval_results = evaluate_onnx(run_cfg, onnx_session, data_loader, desc=f"[{mode_name} (ONNX INT8)]", class_names=class_names, log_class_metrics=True)
        
        if eval_results['labels'] and eval_results['preds']:
            plot_and_save_confusion_matrix(eval_results['labels'], eval_results['preds'], class_names, save_dir, timestamp)
        
        return eval_results['accuracy']
    
    elif use_fp16:
        logging.info("="*50)
        logging.info("FP16 Inference (ONNX via onnxconverter-common)을 적용합니다.")

        if not onnxruntime or not onnx or not float16:
            logging.error("ONNX Runtime, onnx, 또는 onnxconverter-common이 설치되지 않았습니다.")
            return None

        # 1. Export FP32 ONNX first
        # Ensure CPU for export stability
        model.to('cpu') 
        model.eval()
        
        fp32_onnx_path = os.path.join(save_dir, f'model_fp32_for_fp16.onnx')
        dummy_input_cpu = single_dummy_input.to('cpu') # FP32 Input
        
        torch.onnx.export(model, dummy_input_cpu, fp32_onnx_path,
                          export_params=True, opset_version=17,
                          do_constant_folding=True,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        
        # 2. Convert to FP16 using onnxconverter-common
        fp16_onnx_path = os.path.join(save_dir, f'best_model_fp16.onnx')
        logging.info("onnxconverter-common을 사용하여 float16 변환을 수행합니다...")
        
        onnx_model = onnx.load(fp32_onnx_path)
        fp16_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
        onnx.save(fp16_model, fp16_onnx_path)
        
        logging.info(f"ONNX FP16 모델 저장 완료: {fp16_onnx_path}")

        # 3. Evaluate
        # [수정] 모델 로드 메모리(Static Footprint)와 추론 메모리(Dynamic Overhead) 분리 측정
        flush_memory() # [수정]
        # [추가] 전체 메모리 증가량 계산을 위한 기준점 측정
        process = psutil.Process(os.getpid()) if psutil else None
        mem_before_load = process.memory_info().rss / (1024 * 1024) if process else 0

        with MemoryMonitor() as mem_mon:
            sess_options = onnxruntime.SessionOptions()
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            # [확인용] 최적화된 그래프를 파일로 저장 (Fusion 확인용)
            sess_options.optimized_model_filepath = os.path.join(save_dir, "model_graph_fp16.onnx")
            onnx_session = onnxruntime.InferenceSession(fp16_onnx_path, sess_options=sess_options, providers=['CPUExecutionProvider'])

        if psutil:
            logging.info(f"[Model Load] ONNX 모델(FP16) 로드 중 피크 메모리 - 모델 로드 전 청소 직후 메모리: {mem_mon.peak_memory - mem_mon.start_memory:.2f} MB")

        inference_peak = measure_onnx_performance(onnx_session, dummy_input_cpu)
        if psutil and inference_peak:
             logging.info(f"[Total] (FP16) 추론 중 피크 메모리 - 모델 로드 전 청소 직후 메모리: {inference_peak - mem_before_load:.2f} MB")

        eval_results = evaluate_onnx(run_cfg, onnx_session, data_loader, desc=f"[{mode_name} (ONNX FP16)]", class_names=class_names, log_class_metrics=True)

        if eval_results['labels'] and eval_results['preds']:
            plot_and_save_confusion_matrix(eval_results['labels'], eval_results['preds'], class_names, save_dir, timestamp)
            
        return eval_results['accuracy']

    # --- 샘플 당 Forward Pass 시간 및 메모리 사용량 측정 ---
    avg_inference_time_per_sample = 0.0
    logging.info("="*50)
    logging.info("GPU 캐시를 비우고 측정을 시작합니다.")
    if device.type == 'cuda' and torch.cuda.is_available():
        # [추가] Python 가비지 컬렉션 및 CUDA 캐시 비우기
        flush_memory() # [수정]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        # 시간 측정을 위한 예열(warm-up)
        with torch.no_grad():
            for _ in range(10):
                _ = model(single_dummy_input)

        # 실제 시간 측정
        # 구간별 시간 측정을 위한 이벤트 생성
        start_event = torch.cuda.Event(enable_timing=True)
        encoder_end_event = torch.cuda.Event(enable_timing=True)
        decoder_end_event = torch.cuda.Event(enable_timing=True)
        classifier_end_event = torch.cuda.Event(enable_timing=True)

        num_iterations = 100
        # 각 반복의 시간을 저장하기 위한 리스트
        iteration_times = {'encoder': [], 'decoder': [], 'classifier': [], 'total': []}

        with torch.no_grad():
            for _ in range(num_iterations):
                start_event.record()
                # 1. Encoder 구간
                encoded_features = model.encoder(single_dummy_input)
                encoder_end_event.record()
                # 2. Decoder 구간
                decoded_features = model.decoder(encoded_features)
                decoder_end_event.record()
                # 3. Classifier 구간
                _ = model.classifier(decoded_features)
                classifier_end_event.record()

                # 모든 이벤트가 기록된 후 동기화
                torch.cuda.synchronize()
                iteration_times['encoder'].append(start_event.elapsed_time(encoder_end_event))
                iteration_times['decoder'].append(encoder_end_event.elapsed_time(decoder_end_event))
                iteration_times['classifier'].append(decoder_end_event.elapsed_time(classifier_end_event))
                iteration_times['total'].append(start_event.elapsed_time(classifier_end_event))
            
        # 평균 및 표준편차 계산
        avg_total_time = np.mean(iteration_times['total'])
        std_total_time = np.std(iteration_times['total'])
        avg_encoder_time = np.mean(iteration_times['encoder'])
        avg_decoder_time = np.mean(iteration_times['decoder'])
        avg_classifier_time = np.mean(iteration_times['classifier'])

        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        logging.info(f"샘플 당 평균 Forward Pass 시간: {avg_total_time:.2f}ms (std: {std_total_time:.2f}ms) (1개 샘플 x {num_iterations}회 반복)")
        logging.info(f"  - Encoder: {avg_encoder_time:.2f}ms")
        logging.info(f"  - Decoder: {avg_decoder_time:.2f}ms")
        logging.info(f"  - Classifier: {avg_classifier_time:.2f}ms")
        logging.info(f"샘플 당 Forward Pass 시 최대 GPU 메모리 사용량: {peak_memory_mb:.2f} MB")
    else:
        logging.info("CUDA를 사용할 수 없어 CPU 추론 시간을 측정합니다.")
        
        # CPU 시간 측정을 위한 예열(warm-up)
        with torch.no_grad():
            for _ in range(10):
                _ = model(single_dummy_input)
        
        flush_memory() # [수정] CPU 메모리 측정 전 GC 수행

        # 실제 시간 측정
        num_iterations = 100
        iteration_times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = model(single_dummy_input)
                end_time = time.perf_counter()
                iteration_times.append((end_time - start_time) * 1000) # ms

        avg_inference_time_per_sample = np.mean(iteration_times)
        std_inference_time_per_sample = np.std(iteration_times)
        
        # FPS 계산 및 통계
        fps_per_iteration = [1000 / t for t in iteration_times if t > 0]
        avg_fps = np.mean(fps_per_iteration) if fps_per_iteration else 0
        std_fps = np.std(fps_per_iteration) if fps_per_iteration else 0

        logging.info(f"샘플 당 평균 Forward Pass 시간 (CPU): {avg_inference_time_per_sample:.2f}ms (std: {std_inference_time_per_sample:.2f}ms)")
        logging.info(f"샘플 당 평균 FPS (CPU): {avg_fps:.2f} FPS (std: {std_fps:.2f}) (1개 샘플 x {num_iterations}회 반복)")

    # 2. 테스트셋 성능 평가
    only_inference_mode = getattr(run_cfg, 'only_inference', False)

    if only_inference_mode:
        # 순수 추론 모드: 예측 결과만 생성하고 CSV로 저장
        all_filenames = []
        all_predictions = []
        all_attention_maps = []
        all_confidences = []
        show_log = getattr(run_cfg, 'show_log', True)
        progress_bar = tqdm(data_loader, desc=f"[{mode_name}]", leave=False, disable=not show_log)
        with torch.no_grad():
            for images, _, filenames in progress_bar:
                images = images.to(device)
                outputs = model(images)
                
                # Softmax를 적용하여 확률 계산
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # 가장 높은 확률(confidence)과 해당 인덱스(예측 클래스)를 가져옴
                confidences, predicted_indices = torch.max(probabilities, 1)
                
                all_filenames.extend(filenames)
                all_predictions.extend([class_names[p] for p in predicted_indices.cpu().numpy()])
                all_confidences.extend(confidences.cpu().numpy())

                # 어텐션 맵 저장 (시각화를 위해)
                if model_cfg.save_attention:
                    all_attention_maps.append(model.decoder.embedding4decoder.decoder.layers[-1].attn.cpu())
        
        # 결과를 DataFrame으로 만들어 CSV 파일로 저장
        results_df = pd.DataFrame({
            'filename': all_filenames,
            'prediction': all_predictions,
            'confidence': all_confidences
        })
        results_df['confidence'] = results_df['confidence'].map('{:.4f}'.format) # 소수점 4자리까지 표시
        result_csv_path = os.path.join(save_dir, f'inference_results_{timestamp}.csv')
        results_df.to_csv(result_csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"추론 결과가 '{result_csv_path}'에 저장되었습니다.")
        final_acc = None # 정확도 없음

    else:
        # 평가 모드: 기존 evaluate 함수 호출
        # 어텐션 맵을 저장하기 위해 evaluate 함수를 직접 호출하는 대신, 루프를 여기서 실행합니다.
        eval_results = evaluate(run_cfg, model, data_loader, device, nn.CrossEntropyLoss(), 'crossentropyloss', desc=f"[{mode_name}]", class_names=class_names, log_class_metrics=True)
        final_acc = eval_results['accuracy']

        # 3. 혼동 행렬 생성 및 저장 (최종 평가 시에만)
        # evaluate 함수가 이미 혼동 행렬에 필요한 labels와 preds를 반환하므로, 이를 사용합니다.
        # 단, 어텐션 맵 시각화를 위해 data_loader를 다시 순회해야 합니다.
        # 여기서는 기존 로직을 유지하고, 어텐션 맵 시각화 부분에서만 수정합니다.
        if eval_results['labels'] and eval_results['preds']:
            plot_and_save_confusion_matrix(eval_results['labels'], eval_results['preds'], class_names, save_dir, timestamp)

    # 4. 어텐션 맵 시각화 (설정이 True인 경우)
    if model_cfg.save_attention:
        try:
            # 1. 어텐션 맵을 저장할 전용 폴더 생성
            attn_save_dir = os.path.join(save_dir, f'attention_map_{timestamp}')
            os.makedirs(attn_save_dir, exist_ok=True)

            num_to_save = min(getattr(model_cfg, 'num_plot_attention', 10), len(data_loader.dataset))
            logging.info(f"어텐션 맵 시각화를 시작합니다 ({num_to_save}개 샘플, 저장 위치: '{attn_save_dir}').")

            saved_count = 0
            # 데이터 로더를 순회하며 num_to_save 개수만큼 시각화
            for sample_images, sample_labels, sample_filenames in data_loader:
                if saved_count >= num_to_save:
                    break

                sample_images = sample_images.to(device)
                batch_size = sample_images.size(0)

                # 모델을 실행하여 어텐션 맵이 저장되도록 함
                with torch.no_grad():
                    outputs = model(sample_images)

                _, predicted_indices = torch.max(outputs.data, 1)
                
                # 모델 실행 후 저장된 어텐션 맵을 가져옵니다.
                # 이 값은 배치 단위의 어텐션 맵입니다.
                batch_attention_maps = model.decoder.embedding4decoder.decoder.layers[-1].attn

                # 현재 배치에서 저장해야 할 샘플 수만큼 반복
                for i in range(batch_size):
                    if saved_count >= num_to_save:
                        break

                    predicted_class = class_names[predicted_indices[i].item()]
                    original_filename = sample_filenames[i]
                    
                    actual_class = "Unknown" if only_inference_mode else class_names[sample_labels[i].item()]

                    plot_and_save_attention_maps(
                        batch_attention_maps, sample_images, attn_save_dir, model_cfg.img_size, model_cfg,
                        sample_idx=i, original_filename=original_filename, actual_class=actual_class, predicted_class=predicted_class
                    )
                    saved_count += 1
            
            logging.info(f"어텐션 맵 {saved_count}개 저장 완료.")
        except Exception as e:
            logging.error(f"어텐션 맵 시각화 중 오류 발생: {e}")
    # --- ONNX 변환 및 평가 (config.yaml 설정에 따라) ---
    evaluate_onnx_flag = getattr(run_cfg, 'evaluate_onnx', False)
    if evaluate_onnx_flag and onnxruntime and dummy_input is not None:
        if use_int8 or use_fp16:
            logging.warning("INT8 양자화 또는 FP16 추론 모드에서는 ONNX 변환 및 평가를 지원하지 않습니다. ONNX 평가를 건너뜁니다.")
        else:
            logging.info("="*50)
            logging.info("ONNX 변환 및 평가를 시작합니다.")
            onnx_path = os.path.join(save_dir, f'model_{timestamp}.onnx')
            try:
                # 모델을 CPU로 이동하여 ONNX로 변환 (일반적으로 더 안정적)
                model.to('cpu')
                # --- ONNX 런타임 세션 옵션 설정 ---
                sess_options = onnxruntime.SessionOptions()
                sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                # [확인용] 최적화된 그래프를 파일로 저장 (Fusion 확인용)
                sess_options.optimized_model_filepath = os.path.join(save_dir, "model_graph_fp32.onnx")

                torch.onnx.export(model, dummy_input.to('cpu'), onnx_path, 
                                    export_params=True, opset_version=17,
                                    do_constant_folding=True,
                                    input_names=['input'], output_names=['output'],
                                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
                model.to(device) # 모델을 원래 장치로 복원

                # ONNX 파일 크기 로깅
                onnx_file_size_bytes = os.path.getsize(onnx_path)
                onnx_file_size_mb = onnx_file_size_bytes / (1024 * 1024)
                logging.info(f"모델이 ONNX 형식으로 변환되어 '{onnx_path}'에 저장되었습니다. (크기: {onnx_file_size_mb:.2f} MB)")

                # ONNX 런타임 세션 생성 및 평가
                # [수정] 모델 로드 메모리(Static Footprint)와 추론 메모리(Dynamic Overhead) 분리 측정
                flush_memory() # [수정]
                # [추가] 전체 메모리 증가량 계산을 위한 기준점 측정
                process = psutil.Process(os.getpid()) if psutil else None
                mem_before_load = process.memory_info().rss / (1024 * 1024) if process else 0

                with MemoryMonitor() as mem_mon:
                    onnx_session = onnxruntime.InferenceSession(onnx_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
                
                if psutil:
                    logging.info(f"[Model Load] ONNX 모델(FP32) 로드 중 피크 메모리 - 모델 로드 전 청소 직후 메모리: {mem_mon.peak_memory - mem_mon.start_memory:.2f} MB")

                inference_peak = measure_onnx_performance(onnx_session, dummy_input)
                if psutil and inference_peak:
                    logging.info(f"[Total] (FP32) 추론 중 피크 메모리 - 모델 로드 전 청소 직후 메모리: {inference_peak - mem_before_load:.2f} MB")

                evaluate_onnx(run_cfg, onnx_session, data_loader, desc=f"[{mode_name} (ONNX)]", class_names=class_names, log_class_metrics=True)

            except Exception as e:
                logging.error(f"ONNX 변환 또는 평가 중 오류 발생: {e}")

    return final_acc


def main():
    """메인 실행 함수"""
    # --- YAML 설정 파일 로드 --- #
    parser = argparse.ArgumentParser(description="YAML 설정을 이용한 이미지 분류기")
    parser.add_argument('--config', type=str, default='config.yaml', help='설정 파일 경로')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # SimpleNamespace를 사용하여 딕셔너리처럼 접근 가능하게 변환
    run_cfg = SimpleNamespace(**config['run'])
    train_cfg = SimpleNamespace(**config['training_main'])
    model_cfg = SimpleNamespace(**config['model'])
    # 중첩된 scheduler_params 딕셔너리를 SimpleNamespace로 변환
    if hasattr(train_cfg, 'scheduler_params') and isinstance(train_cfg.scheduler_params, dict):
        train_cfg.scheduler_params = SimpleNamespace(**train_cfg.scheduler_params)

    # dataset_cfg도 SimpleNamespace로 변환
    run_cfg.dataset = SimpleNamespace(**run_cfg.dataset)
    
    data_dir_name = run_cfg.dataset.name

    # --- 전역 시드 고정 ---
    global_seed = getattr(run_cfg, 'global_seed', None)
    if global_seed is not None:
        random.seed(global_seed)
        os.environ['PYTHONHASHSEED'] = str(global_seed)
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)
        torch.cuda.manual_seed(global_seed)
        logging.info(f"전역 랜덤 시드를 {global_seed}로 고정합니다.")

    # --- 실행 디렉토리 설정 ---
    # 훈련/추론별 실행 디렉토리 및 로깅 설정
    # 폴더 이름에 주요 모델 설정을 포함시킵니다.
    run_suffix = f"grid{model_cfg.grid_size}_patch{model_cfg.num_decoder_patches}_layer{model_cfg.num_decoder_layers}_head{model_cfg.num_heads}"

    if run_cfg.mode == 'train':
        run_dir_path, timestamp = setup_logging(run_cfg, data_dir_name, run_name_suffix=run_suffix)
        log_dir_path = run_dir_path
    else:
        # 추론 모드: pth_inference_dir 또는 onnx_inference_path 사용
        onnx_inference_path = getattr(run_cfg, 'onnx_inference_path', None)
        if not (onnx_inference_path and os.path.exists(onnx_inference_path)):
            run_dir_path = getattr(run_cfg, 'pth_inference_dir', None)
            if getattr(run_cfg, 'show_log', True) and (not run_dir_path or not os.path.isdir(run_dir_path)):
                logging.error("추론 모드에서는 'config.yaml'에 'pth_inference_dir'를 올바르게 설정해야 합니다.")
                return
        else:
            run_dir_path = '.'
        log_dir_path, timestamp = setup_logging(run_cfg, data_dir_name, run_name_suffix=run_suffix)

    # --- 전역 시드 고정 (로그 파일에 남기기 위해 setup_logging 이후 실행) ---
    global_seed = getattr(run_cfg, 'global_seed', None)
    if global_seed is not None:
        random.seed(global_seed)
        os.environ['PYTHONHASHSEED'] = str(global_seed)
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)
        torch.cuda.manual_seed(global_seed)
        logging.info(f"전역 랜덤 시드를 {global_seed}로 고정합니다.")

    # --- 설정 파일 내용 로깅 ---
    config_str = yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False)
    logging.info("="*50)
    logging.info("config.yaml:")
    logging.info("\n" + config_str)
    logging.info("="*50)

    # --- 공통 파라미터 설정 ---
    use_cuda_if_available = getattr(run_cfg, 'cuda', True)
    device = torch.device("cuda" if use_cuda_if_available and torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        logging.info(f"CUDA 사용 가능. GPU 사용을 시작합니다. (Device: {torch.cuda.get_device_name(0)})")
    else:
        if use_cuda_if_available:
            logging.warning("config.yaml에서 CUDA 사용이 활성화되었지만, 사용 가능한 CUDA 장치를 찾을 수 없습니다. CPU를 사용합니다.")
        if cpuinfo:
            cpu_info_str = cpuinfo.get_cpu_info().get('brand_raw', 'N/A')
            logging.info(f"CPU 사용을 시작합니다. (Device: {cpu_info_str})")
        else:
            logging.info("CPU 사용을 시작합니다.")
            logging.info("CPU 정보를 표시하려면 'pip install py-cpuinfo'를 실행하세요.")

    # --- 데이터 준비 ---
    train_loader, valid_loader, test_loader, num_labels, class_names, pos_weight = prepare_data(run_cfg, train_cfg, model_cfg)

    # --- 모델 구성 ---
    num_patches_h = model_cfg.grid_size
    num_patches_w = model_cfg.grid_size
    num_encoder_patches = num_patches_h * num_patches_w
    logging.info(f"이미지 크기: {model_cfg.img_size}, 그리드 크기: {model_cfg.grid_size}x{model_cfg.grid_size} -> 인코더 패치 수: {num_encoder_patches}개")

    decoder_params = {
        'num_encoder_patches': num_encoder_patches,
        'grid_size_h': num_patches_h,
        'grid_size_w': num_patches_w,
        'num_labels': num_labels,
        'num_decoder_layers': model_cfg.num_decoder_layers,
        'num_decoder_patches': model_cfg.num_decoder_patches,
        'featured_patch_dim': model_cfg.featured_patch_dim,
        'adaptive_initial_query': getattr(model_cfg, 'adaptive_initial_query', False),
        'emb_dim': model_cfg.emb_dim,
        'num_heads': model_cfg.num_heads,
        'decoder_ff_ratio': model_cfg.decoder_ff_ratio,
        'dropout': model_cfg.dropout,
        'positional_encoding': model_cfg.positional_encoding,
        'res_attention': model_cfg.res_attention,
        'save_attention': model_cfg.save_attention,
        'drop_path_ratio': getattr(model_cfg, 'drop_path_ratio', 0.0), # [수정] drop_path_ratio 설정 전달
    }
    decoder_args = SimpleNamespace(**decoder_params)

    encoder = PatchConvEncoder(
        grid_size=model_cfg.grid_size,
        featured_patch_dim=model_cfg.featured_patch_dim,
        cnn_feature_extractor_name=model_cfg.cnn_feature_extractor['name'],
        pre_trained=train_cfg.pre_trained,
    )
    decoder = DecoderBackbone(args=decoder_args)

    # Classifier는 num_decoder_patches와 emb_dim을 기반으로 입력을 처리합니다.
    classifier = Classifier(
        num_decoder_patches=model_cfg.num_decoder_patches,
        emb_dim=model_cfg.emb_dim,
        num_labels=num_labels,
        dropout=model_cfg.dropout,
    )
    model = HybridModel(encoder, decoder, classifier).to(device)

    # 모델 생성 후 파라미터 수 로깅
    log_model_parameters(model)
    
    # --- 모드에 따라 실행 ---
    if run_cfg.mode == 'train':
        # --- 옵티마이저 및 스케줄러 설정 ---
        optimizer, scheduler = None, None
        optimizer_name = getattr(train_cfg, 'optimizer', 'adamw').lower()
        logging.info("="*50)
        if optimizer_name == 'sgd':
            # SGD 옵티마이저에 필요한 파라미터들을 train_cfg에서 가져옵니다.
            momentum = getattr(train_cfg, 'momentum', 0.9)
            weight_decay = getattr(train_cfg, 'weight_decay', 0.0001)
            logging.info(f"옵티마이저: SGD (lr={train_cfg.lr}, momentum={momentum}, weight_decay={weight_decay})")
            optimizer = optim.SGD(model.parameters(), lr=train_cfg.lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == 'nadam':
            weight_decay = getattr(train_cfg, 'weight_decay', 0.0)
            logging.info(f"옵티마이저: NAdam (lr={train_cfg.lr}, weight_decay={weight_decay})")
            optimizer = optim.NAdam(model.parameters(), lr=train_cfg.lr, weight_decay=weight_decay)
        elif optimizer_name == 'radam':
            weight_decay = getattr(train_cfg, 'weight_decay', 0.0)
            logging.info(f"옵티마이저: RAdam (lr={train_cfg.lr}, weight_decay={weight_decay})")
            optimizer = optim.RAdam(model.parameters(), lr=train_cfg.lr, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            weight_decay = getattr(train_cfg, 'weight_decay', 0.0)
            momentum = getattr(train_cfg, 'momentum', 0.0)
            logging.info(f"옵티마이저: RMSprop (lr={train_cfg.lr}, weight_decay={weight_decay}, momentum={momentum})")
            optimizer = optim.RMSprop(model.parameters(), lr=train_cfg.lr, weight_decay=weight_decay, momentum=momentum)
        else:
            # 기본값 또는 'adamw'로 설정된 경우
            weight_decay = getattr(train_cfg, 'weight_decay', 0.01)
            logging.info(f"옵티마이저: AdamW (lr={train_cfg.lr}, weight_decay={weight_decay})")
            optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=weight_decay)

        # scheduler_params가 없으면 빈 객체로 초기화
        scheduler_params = getattr(train_cfg, 'scheduler_params', SimpleNamespace())

        scheduler_name = getattr(train_cfg, 'scheduler', 'none').lower()
        if scheduler_name == 'multisteplr':
            milestones = getattr(train_cfg, 'milestones', [])
            gamma = getattr(train_cfg, 'gamma', 0.1)
            logging.info(f"스케줄러: MultiStepLR (milestones={milestones}, gamma={gamma})")
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        elif scheduler_name == 'cosineannealinglr':
            T_max = getattr(scheduler_params, 'T_max', train_cfg.epochs)
            eta_min = getattr(scheduler_params, 'eta_min', 0.0)
            logging.info(f"스케줄러: CosineAnnealingLR (T_max={T_max}, eta_min={eta_min})")
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        else:
            logging.info("스케줄러를 사용하지 않습니다.")

        # 훈련 시에는 train_loader와 valid_loader 사용
        train(run_cfg, train_cfg, model, optimizer, scheduler, train_loader, valid_loader, device, run_dir_path, class_names, pos_weight)
        final_acc = inference(run_cfg, model_cfg, model, test_loader, device, run_dir_path, timestamp, mode_name="Test", class_names=class_names)

        # --- 그래프 생성 ---
        # 로그 파일 이름은 setup_logging에서 생성된 패턴을 기반으로 함
        log_filename = f"log_{timestamp}.log"
        log_file_path = os.path.join(run_dir_path, log_filename)
        if final_acc is not None:
            plot_and_save_val_accuracy_graph(log_file_path, run_dir_path, final_acc, timestamp)
            plot_and_save_train_val_accuracy_graph(log_file_path, run_dir_path, final_acc, timestamp)
            plot_and_save_f1_normal_graph(log_file_path, run_dir_path, timestamp, class_names)
            plot_and_save_loss_graph(log_file_path, run_dir_path, timestamp)
            plot_and_save_lr_graph(log_file_path, run_dir_path, timestamp)
            plot_and_save_compiled_graph(run_dir_path, timestamp)

    elif run_cfg.mode == 'inference':
        # 추론 모드에서는 test_loader를 사용해 성능 평가
        # onnx_inference_path가 지정된 경우, model 객체는 필요 없으므로 None을 전달합니다.
        onnx_inference_path = getattr(run_cfg, 'onnx_inference_path', None)
        if onnx_inference_path and os.path.exists(onnx_inference_path):
            logging.info(f"'{onnx_inference_path}' ONNX 파일 평가를 위해 PyTorch 모델 생성을 건너뜁니다.")
            inference(run_cfg, model_cfg, None, test_loader, device, run_dir_path, timestamp, mode_name="Inference", class_names=class_names)
        else:
            # 모델 생성 후 파라미터 수 로깅
            inference(run_cfg, model_cfg, model, test_loader, device, run_dir_path, timestamp, mode_name="Inference", class_names=class_names, output_dir=log_dir_path)


# =============================================================================
# 5. 메인 실행 블록
# =============================================================================
if __name__ == '__main__':
    main()
