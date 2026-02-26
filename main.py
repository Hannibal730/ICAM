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
from model import Model as DecoderBackbone, Encoder, Classifier, HybridModel
from baseline import run_torch_pruning, find_sparsity_for_target_flops, find_sparsity_for_target_params

try:
    import cpuinfo
except ImportError:
    cpuinfo = None

from dataloader import prepare_data # 데이터 로딩 함수 임포트
from utils import measure_model_complexity, log_hybrid_model_parameters

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

def build_hybrid_model(model_cfg, pre_trained, num_labels, device):
    """config 기반으로 ICAM HybridModel을 생성합니다."""
    num_patches_h = model_cfg.num_patches_per_side
    num_patches_w = model_cfg.num_patches_per_side
    num_encoder_patches = num_patches_h * num_patches_w

    decoder_params = {
        'num_encoder_patches': num_encoder_patches,
        'num_patches_h': num_patches_h,
        'num_patches_w': num_patches_w,
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
        'visualize_attention': model_cfg.visualize_attention,
        'drop_path_ratio': getattr(model_cfg, 'drop_path_ratio', 0.0),
    }
    decoder_args = SimpleNamespace(**decoder_params)

    encoder = Encoder(
        num_patches_per_side=model_cfg.num_patches_per_side,
        encoder_dim=model_cfg.encoder_dim,
        cnn_feature_extractor_name=model_cfg.cnn_feature_extractor['name'],
        pre_trained=pre_trained,
    )
    decoder = DecoderBackbone(args=decoder_args)
    classifier = Classifier(
        num_decoder_patches=model_cfg.num_decoder_patches,
        emb_dim=model_cfg.emb_dim,
        num_labels=num_labels,
        dropout=model_cfg.dropout,
    )
    model = HybridModel(encoder, decoder, classifier).to(device)
    return model

def create_optimizer_and_scheduler(cfg, model):
    """설정에 맞춰 옵티마이저/스케줄러를 생성합니다."""
    optimizer_name = getattr(cfg, 'optimizer', 'adamw').lower()
    optimizer = None
    scheduler = None

    if optimizer_name == 'sgd':
        momentum = getattr(cfg, 'momentum', 0.9)
        weight_decay = getattr(cfg, 'weight_decay', 0.0001)
        logging.info(f"옵티마이저: SGD (lr={cfg.lr}, momentum={momentum}, weight_decay={weight_decay})")
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'nadam':
        weight_decay = getattr(cfg, 'weight_decay', 0.0)
        logging.info(f"옵티마이저: NAdam (lr={cfg.lr}, weight_decay={weight_decay})")
        optimizer = optim.NAdam(model.parameters(), lr=cfg.lr, weight_decay=weight_decay)
    elif optimizer_name == 'radam':
        weight_decay = getattr(cfg, 'weight_decay', 0.0)
        logging.info(f"옵티마이저: RAdam (lr={cfg.lr}, weight_decay={weight_decay})")
        optimizer = optim.RAdam(model.parameters(), lr=cfg.lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        weight_decay = getattr(cfg, 'weight_decay', 0.0)
        momentum = getattr(cfg, 'momentum', 0.0)
        logging.info(f"옵티마이저: RMSprop (lr={cfg.lr}, weight_decay={weight_decay}, momentum={momentum})")
        optimizer = optim.RMSprop(model.parameters(), lr=cfg.lr, weight_decay=weight_decay, momentum=momentum)
    else:
        weight_decay = getattr(cfg, 'weight_decay', 0.01)
        logging.info(f"옵티마이저: AdamW (lr={cfg.lr}, weight_decay={weight_decay})")
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=weight_decay)

    scheduler_params = getattr(cfg, 'scheduler_params', SimpleNamespace())
    scheduler_name = getattr(cfg, 'scheduler', 'none').lower()
    if scheduler_name == 'multisteplr':
        milestones = getattr(cfg, 'milestones', [])
        gamma = getattr(cfg, 'gamma', 0.1)
        logging.info(f"스케줄러: MultiStepLR (milestones={milestones}, gamma={gamma})")
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif scheduler_name == 'cosineannealinglr':
        T_max = getattr(scheduler_params, 'T_max', cfg.epochs)
        eta_min = getattr(scheduler_params, 'eta_min', 0.0)
        logging.info(f"스케줄러: CosineAnnealingLR (T_max={T_max}, eta_min={eta_min})")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        logging.info("스케줄러를 사용하지 않습니다.")

    return optimizer, scheduler

def is_pruning_enabled(baseline_cfg):
    pruning_flags = [
        'use_l1_pruning', 'use_l2_pruning', 'use_fpgm_pruning', 'use_lamp_pruning',
        'use_slimming_pruning', 'use_taylor_pruning', 'use_wanda_pruning'
    ]
    return any(getattr(baseline_cfg, flag, False) for flag in pruning_flags)

def get_icam_pruning_ignored_layers(model, baseline_cfg):
    """ICAM Pruning 제외 레이어를 구성합니다."""
    prunable_types = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm)
    ignored_layers = []
    pruning_scope = getattr(baseline_cfg, 'pruning_scope', 'full_model').lower()

    # 마지막 분류층은 baseline과 동일하게 제외
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'projection'):
        projection = model.classifier.projection
        if isinstance(projection, nn.Sequential) and len(projection) > 0 and isinstance(projection[-1], nn.Linear):
            ignored_layers.append(projection[-1])

    # baseline의 ViT qkv 제외 정책과 동일 취지로, ICAM의 핵심 attention projection은 기본 제외
    # (원하면 config에서 false로 변경 가능)
    exclude_attention_qkv = getattr(baseline_cfg, 'exclude_attention_qkv_pruning', True)
    if exclude_attention_qkv:
        attention_proj_keywords = ('q_proj', 'k_proj', 'v_proj', 'W_K_init', 'W_V_init')
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(keyword in name for keyword in attention_proj_keywords):
                ignored_layers.append(module)

    # Embedding bridge(feat_to_emb)가 줄어들면 pos_embed와 adaptive query 경로에서
    # 차원 불일치가 발생할 수 있어 기본적으로 제외합니다.
    exclude_embedding_bridge = getattr(baseline_cfg, 'exclude_embedding_bridge_pruning', True)
    if exclude_embedding_bridge:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'feat_to_emb' in name:
                ignored_layers.append(module)

    # Learnable query는 module에 속하지 않는 nn.Parameter이므로 별도 제외
    # (torch-pruning 기본 동작의 unwrapped auto-pruning 방지)
    exclude_learnable_queries = getattr(baseline_cfg, 'exclude_learnable_queries_pruning', True)
    if exclude_learnable_queries and hasattr(model, 'decoder') and hasattr(model.decoder, 'embedding4decoder'):
        learnable_queries = getattr(model.decoder.embedding4decoder, 'learnable_queries', None)
        if isinstance(learnable_queries, nn.Parameter):
            ignored_layers.append(learnable_queries)

    if pruning_scope == 'encoder_only':
        encoder_prunable_prefix = "encoder.shared_conv.0.conv_front"
        for name, module in model.named_modules():
            if isinstance(module, prunable_types) and not name.startswith(encoder_prunable_prefix):
                ignored_layers.append(module)
        logging.info(
            "ICAM Pruning 정책: encoder_only (decoder/classifier 제외). "
            f"ignored_layers={len(ignored_layers)}"
        )
    else:
        logging.info(
            "ICAM Pruning 정책: full_model (encoder+decoder+classifier 대상, 마지막 분류층/attention qkv/embedding bridge/learnable queries 제외). "
            f"ignored_layers={len(ignored_layers)}"
        )

    return ignored_layers

def create_loss_for_pruning(train_cfg, pos_weight, device):
    """Pruning 전처리(예: Taylor 중요도 계산)에 사용할 손실 함수를 생성합니다."""
    loss_function_name = getattr(train_cfg, 'loss_function', 'CrossEntropyLoss').lower()
    if loss_function_name == 'bcewithlogitsloss':
        weight_value = getattr(train_cfg, 'bce_pos_weight', None)
        if weight_value == 'auto':
            final_pos_weight = pos_weight.to(device) if pos_weight is not None else None
        else:
            final_pos_weight = torch.tensor(float(weight_value), dtype=torch.float).to(device) if weight_value is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=final_pos_weight)
    label_smoothing = getattr(train_cfg, 'label_smoothing', 0.0)
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

def save_pruning_info(run_dir_path, baseline_cfg, target_type, target_value, optimal_sparsity):
    pruning_method = 'unknown'
    for method_name, cfg_name in (
        ('l1', 'use_l1_pruning'),
        ('l2', 'use_l2_pruning'),
        ('fpgm', 'use_fpgm_pruning'),
        ('lamp', 'use_lamp_pruning'),
        ('slimming', 'use_slimming_pruning'),
        ('taylor', 'use_taylor_pruning'),
        ('wanda', 'use_wanda_pruning'),
    ):
        if getattr(baseline_cfg, cfg_name, False):
            pruning_method = method_name
            break

    pruning_info = {
        'model_name': 'icam',
        'pruning_method': pruning_method,
        'target_type': target_type,
        'target_value': float(target_value),
        'optimal_sparsity': float(optimal_sparsity),
    }
    pruning_info_path = os.path.join(run_dir_path, 'pruning_info.yaml')
    with open(pruning_info_path, 'w', encoding='utf-8') as f:
        yaml.dump(pruning_info, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    logging.info(
        f"계산된 Pruning 정보(희소도: {optimal_sparsity:.4f})를 '{pruning_info_path}'에 저장했습니다."
    )

def merge_namespace_with_defaults(primary_cfg, default_cfg):
    """primary_cfg에 없는 필드는 default_cfg로 채운 새 SimpleNamespace를 반환합니다."""
    merged = SimpleNamespace(**vars(default_cfg))
    for key, value in vars(primary_cfg).items():
        setattr(merged, key, value)
    return merged

# =============================================================================
# 2. 훈련 및 평가 함수
# =============================================================================
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

    # [추가] FLOPs 및 MACs 측정
    measure_model_complexity(model, model_cfg.img_size, device)

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
                if model_cfg.visualize_attention:
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
    if model_cfg.visualize_attention:
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
    finetune_cfg_dict = config.get('finetuning_pruned', {})
    finetune_cfg = SimpleNamespace(**finetune_cfg_dict)

    # Pruning 관련 설정은 finetuning_pruned 섹션을 우선 사용합니다.
    baseline_section_dict = config.get('baseline', {}) or {}
    pruning_keys = [
        'use_l1_pruning', 'use_l2_pruning', 'use_fpgm_pruning', 'use_lamp_pruning',
        'use_slimming_pruning', 'use_taylor_pruning', 'use_wanda_pruning',
        'use_depgraph_pruning', 'use_isomorphic_pruning',
        'num_wanda_calib_samples',
        'pruning_sparsity', 'pruning_flops_target', 'pruning_params_target',
        'pruning_scope', 'exclude_attention_qkv_pruning', 'exclude_embedding_bridge_pruning',
        'exclude_learnable_queries_pruning'
    ]
    baseline_cfg_dict = dict(baseline_section_dict)
    for key in pruning_keys:
        if key in finetune_cfg_dict:
            baseline_cfg_dict[key] = finetune_cfg_dict[key]
    baseline_cfg = SimpleNamespace(**baseline_cfg_dict)
    # 중첩된 scheduler_params 딕셔너리를 SimpleNamespace로 변환
    if hasattr(train_cfg, 'scheduler_params') and isinstance(train_cfg.scheduler_params, dict):
        train_cfg.scheduler_params = SimpleNamespace(**train_cfg.scheduler_params)
    if hasattr(finetune_cfg, 'scheduler_params') and isinstance(finetune_cfg.scheduler_params, dict):
        finetune_cfg.scheduler_params = SimpleNamespace(**finetune_cfg.scheduler_params)

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
    run_suffix = f"grid{model_cfg.num_patches_per_side}_patch{model_cfg.num_decoder_patches}_layer{model_cfg.num_decoder_layers}_head{model_cfg.num_heads}"

    if run_cfg.mode == 'train':
        run_dir_path, timestamp = setup_logging(run_cfg, data_dir_name, run_name_suffix=run_suffix)
        log_dir_path = run_dir_path
    else:
        run_dir_path = getattr(run_cfg, 'pth_inference_dir', None)
        if getattr(run_cfg, 'show_log', True) and (not run_dir_path or not os.path.isdir(run_dir_path)):
            logging.error("추론 모드에서는 'config.yaml'에 'pth_inference_dir'를 올바르게 설정해야 합니다.")
            return
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
    num_encoder_patches = model_cfg.num_patches_per_side * model_cfg.num_patches_per_side
    logging.info(
        f"이미지 크기: {model_cfg.img_size}, 그리드 크기: {model_cfg.num_patches_per_side}x{model_cfg.num_patches_per_side} "
        f"-> 인코더 패치 수: {num_encoder_patches}개"
    )
    model = build_hybrid_model(model_cfg, train_cfg.pre_trained, num_labels, device)

    # 모델 생성 후 파라미터 수 로깅
    log_hybrid_model_parameters(model)
    
    # --- 모드에 따라 실행 ---
    if run_cfg.mode == 'train':
        use_pruning = is_pruning_enabled(baseline_cfg)
        if use_pruning:
            if len(vars(finetune_cfg)) == 0:
                logging.warning("finetuning_pruned 설정이 없어 training_main 설정을 재사용합니다.")
            finetune_cfg = merge_namespace_with_defaults(finetune_cfg, train_cfg)

        logging.info("="*80)
        logging.info("단계 1/2: training_main 기반 사전 훈련(Pre-training)을 시작합니다.")
        logging.info("="*80)
        optimizer, scheduler = create_optimizer_and_scheduler(train_cfg, model)
        train(run_cfg, train_cfg, model, optimizer, scheduler, train_loader, valid_loader, device, run_dir_path, class_names, pos_weight)

        if use_pruning:
            logging.info("="*80)
            logging.info("단계 2/2: Pruning 및 finetuning_pruned 기반 미세 조정을 시작합니다.")
            logging.info("="*80)

            best_model_path = os.path.join(run_dir_path, run_cfg.pth_best_name)
            if not os.path.exists(best_model_path):
                logging.error(f"사전 훈련된 모델 '{best_model_path}'을 찾을 수 없어 Pruning 단계를 중단합니다.")
            else:
                model.load_state_dict(torch.load(best_model_path, map_location=device))
                logging.info(f"사전 훈련된 모델 '{best_model_path}'을(를) 불러왔습니다.")

                ignored_layers = get_icam_pruning_ignored_layers(model, baseline_cfg)
                pruning_run_kwargs = {'extra_ignored_layers': ignored_layers}
                pruning_criterion = create_loss_for_pruning(train_cfg, pos_weight, device)

                if getattr(baseline_cfg, 'pruning_flops_target', 0.0) > 0:
                    optimal_sparsity = find_sparsity_for_target_flops(
                        model, baseline_cfg, model_cfg, device, train_loader, pruning_criterion,
                        pruning_run_kwargs=pruning_run_kwargs
                    )
                    baseline_cfg.pruning_sparsity = optimal_sparsity
                    save_pruning_info(
                        run_dir_path,
                        baseline_cfg,
                        target_type='flops',
                        target_value=getattr(baseline_cfg, 'pruning_flops_target'),
                        optimal_sparsity=optimal_sparsity
                    )
                elif getattr(baseline_cfg, 'pruning_params_target', 0.0) > 0:
                    optimal_sparsity = find_sparsity_for_target_params(
                        model, baseline_cfg, model_cfg, device, train_loader, pruning_criterion,
                        pruning_run_kwargs=pruning_run_kwargs
                    )
                    baseline_cfg.pruning_sparsity = optimal_sparsity
                    save_pruning_info(
                        run_dir_path,
                        baseline_cfg,
                        target_type='params',
                        target_value=getattr(baseline_cfg, 'pruning_params_target'),
                        optimal_sparsity=optimal_sparsity
                    )

                model = run_torch_pruning(
                    model,
                    baseline_cfg,
                    model_cfg,
                    device,
                    train_loader=train_loader,
                    criterion=pruning_criterion,
                    **pruning_run_kwargs
                )
                log_hybrid_model_parameters(model)

                finetune_train_loader = train_loader
                finetune_valid_loader = valid_loader
                finetune_pos_weight = pos_weight
                finetune_batch_size = getattr(finetune_cfg, 'batch_size', train_cfg.batch_size)
                if finetune_batch_size != train_cfg.batch_size:
                    logging.info(
                        f"finetuning_pruned.batch_size={finetune_batch_size}를 적용하기 위해 DataLoader를 재생성합니다."
                    )
                    finetune_train_loader, finetune_valid_loader, _, _, _, finetune_pos_weight = prepare_data(
                        run_cfg, finetune_cfg, model_cfg
                    )

                logging.info("Pruning 이후 finetuning_pruned 설정으로 미세 조정을 수행합니다.")
                finetune_optimizer, finetune_scheduler = create_optimizer_and_scheduler(finetune_cfg, model)
                train(
                    run_cfg,
                    finetune_cfg,
                    model,
                    finetune_optimizer,
                    finetune_scheduler,
                    finetune_train_loader,
                    finetune_valid_loader,
                    device,
                    run_dir_path,
                    class_names,
                    finetune_pos_weight
                )

        # 훈련 종료 후 Best Model 평가를 위해 모델 객체 재생성
        best_model_path = os.path.join(run_dir_path, run_cfg.pth_best_name)
        if os.path.exists(best_model_path):
            logging.info(f"훈련 완료. 최고 성능 모델 '{best_model_path}'을(를) 불러와 테스트 세트로 최종 평가합니다.")
            final_model = build_hybrid_model(model_cfg, pre_trained=False, num_labels=num_labels, device=device)

            if use_pruning:
                pruning_info_path = os.path.join(run_dir_path, 'pruning_info.yaml')
                if os.path.exists(pruning_info_path):
                    with open(pruning_info_path, 'r', encoding='utf-8') as f:
                        pruning_info = yaml.safe_load(f) or {}
                    if 'optimal_sparsity' in pruning_info:
                        baseline_cfg.pruning_sparsity = pruning_info['optimal_sparsity']

                final_ignored_layers = get_icam_pruning_ignored_layers(final_model, baseline_cfg)
                final_pruning_criterion = create_loss_for_pruning(train_cfg, pos_weight, device)
                final_model = run_torch_pruning(
                    final_model,
                    baseline_cfg,
                    model_cfg,
                    device,
                    train_loader=train_loader,
                    criterion=final_pruning_criterion,
                    extra_ignored_layers=final_ignored_layers
                )

            final_model.load_state_dict(torch.load(best_model_path, map_location=device))
            model = final_model
        else:
            logging.warning("최고 성능 모델 파일을 찾을 수 없습니다. 마지막 에포크 모델로 평가를 진행합니다.")

        final_acc = inference(run_cfg, model_cfg, model, test_loader, device, run_dir_path, timestamp, mode_name="Test", class_names=class_names)

        # --- 그래프 생성 ---
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
        use_pruning_in_inference = is_pruning_enabled(baseline_cfg)
        if use_pruning_in_inference:
            pruning_info_path = os.path.join(run_dir_path, 'pruning_info.yaml')
            if os.path.exists(pruning_info_path):
                with open(pruning_info_path, 'r', encoding='utf-8') as f:
                    pruning_info = yaml.safe_load(f) or {}
                if 'optimal_sparsity' in pruning_info:
                    baseline_cfg.pruning_sparsity = pruning_info['optimal_sparsity']
                    logging.info(
                        f"'{pruning_info_path}'에서 Pruning 희소도({baseline_cfg.pruning_sparsity:.4f})를 불러왔습니다."
                    )

            inference_ignored_layers = get_icam_pruning_ignored_layers(model, baseline_cfg)
            inference_pruning_criterion = create_loss_for_pruning(train_cfg, pos_weight, device)
            model = run_torch_pruning(
                model,
                baseline_cfg,
                model_cfg,
                device,
                train_loader=train_loader,
                criterion=inference_pruning_criterion,
                extra_ignored_layers=inference_ignored_layers
            )
            log_hybrid_model_parameters(model)

        inference(run_cfg, model_cfg, model, test_loader, device, run_dir_path, timestamp, mode_name="Inference", class_names=class_names, output_dir=log_dir_path)

# =============================================================================
# 5. 메인 실행 블록
# =============================================================================
if __name__ == '__main__':
    main()
