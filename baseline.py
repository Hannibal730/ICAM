import os
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score # type: ignore
from types import SimpleNamespace
import pandas as pd
from torchvision import models as torchvision_models
import timm

import argparse
import yaml
import logging
from datetime import datetime # type: ignore
import copy
import random
import time
import types
import gc
import torch.nn.functional as F
try:
    from timm.layers import Attention
except ImportError:
    Attention = None

try:
    from thop import profile
except ImportError:
    profile = None

from dataloader import prepare_data 
from utils import measure_model_complexity, log_model_parameters

# [수정] torch_pruning을 전역 스코프에서 import하여 모든 함수에서 접근 가능하게 함
try:
    from timm.layers import BatchNormAct2d
    import torch_pruning as tp
except ImportError:
    BatchNormAct2d = None
    tp = None

from plot import plot_and_save_train_val_accuracy_graph, plot_and_save_val_accuracy_graph, plot_and_save_confusion_matrix, plot_and_save_f1_normal_graph, plot_and_save_loss_graph, plot_and_save_lr_graph, plot_and_save_compiled_graph

# =============================================================================
# 1. 로깅 및 모델 설정
# =============================================================================
def setup_logging(run_cfg, data_dir_name, baseline_model_name, baseline_cfg):
    """로그 파일을 log 폴더에 생성하고, 콘솔에도 함께 출력하도록 설정합니다."""
    show_log = getattr(run_cfg, 'show_log', True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- [수정] 경량화 옵션 이름을 폴더명에 추가 ---
    lightweight_option_names = []
    pruning_options = [
        'use_l1_pruning', 'use_l2_pruning', 'use_fpgm_pruning',
        'use_lamp_pruning', 'use_slimming_pruning', 'use_taylor_pruning',
        'use_wanda_pruning'
    ]
    for option in pruning_options:
        if getattr(baseline_cfg, option, False):
            # 'use_'와 '_pruning'을 제거하여 이름 생성 (예: 'l1')
            name = option.replace('use_', '').replace('_pruning', '')
            lightweight_option_names.append(name)

    if not show_log:
        logging.disable(logging.CRITICAL)
        return '.', timestamp, lightweight_option_names

    lightweight_str = ""
    if lightweight_option_names:
        lightweight_str = "_" + "-".join(lightweight_option_names)

    run_dir_name = f"baseline_{baseline_model_name}{lightweight_str}_{timestamp}"
    run_dir_path = os.path.join("log", data_dir_name, run_dir_name)
    os.makedirs(run_dir_path, exist_ok=True)
    
    log_filename = os.path.join(run_dir_path, f"log_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    logging.info(f"로그 파일이 '{log_filename}'에 저장됩니다.")
    return run_dir_path, timestamp, lightweight_option_names

def patch_timm_model_for_pruning(model, model_name, device):
    """timm 모델의 Pruning 호환성을 위해 모델 구조를 수정(monkey-patch)합니다."""
    # =============================================================================
    # 1. Attention.forward 몽키 패치
    # =============================================================================
    def patched_attention_forward(self, x, attn_mask=None):
        """
        timm의 Attention.forward를 수정한 버전.
        Pruning으로 인해 입력 채널(C)과 내부 헤드 차원(head_dim * num_heads)이 달라졌을 때 발생하는
        reshape 오류를 해결하기 위해 마지막 reshape에서 C 대신 -1을 사용합니다.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if hasattr(self, 'q_norm'):
            q, k = self.q_norm(q), self.k_norm(k)

        scale = getattr(self, 'scale', 1.0 / (self.head_dim ** 0.5))
        q = q * scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    is_vit_family = model_name in ['vit', 'swin_tiny', 'deit_tiny', 'mobile_vit_s', 'mobile_vit_xs', 'mobile_vit_xxs']
    if is_vit_family and Attention is not None and F is not None:
        logging.info(f"timm 모델({model_name})의 Attention.forward를 Pruning 호환성을 위해 Monkey-patching 합니다.")
        for module in model.modules():
            if isinstance(module, Attention):
                module.forward = types.MethodType(patched_attention_forward, module)

    # =============================================================================
    # 2. 복합 레이어(BatchNormAct2d) 변환
    # =============================================================================
    try:
        from timm.layers import BatchNormAct2d
    except ImportError:
        BatchNormAct2d = None

    def fuse_timm_norm_act_layers(module):
        if BatchNormAct2d is None: return
        for name, child in module.named_children():
            if isinstance(child, BatchNormAct2d):
                bn = nn.BatchNorm2d(child.num_features, child.eps, child.momentum, child.affine, child.track_running_stats)
                bn.load_state_dict(child.state_dict())
                new_module = nn.Sequential(bn, child.act).to(device)
                setattr(module, name, new_module)
            else:
                fuse_timm_norm_act_layers(child)

    fuse_timm_norm_act_layers(model)
    return model

class Xie2019(nn.Module):
    def __init__(self, num_classes, dropout_rate = 0.6):
        super(Xie2019, self).__init__()
        self.dropout_rate = dropout_rate

        self.features = nn.Sequential(
            nn.Conv2d(3,64, 11, padding = 5, stride = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding = 1, stride = 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 128, 3, padding = 1, stride = 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        
        self.classifier = nn.Sequential(
            nn.Linear(128*8*8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_baseline_model(model_name, num_labels, pretrained):
    """지정된 이름의 torchvision 모델을 생성하고 마지막 레이어를 수정합니다."""
    logging.info(f"Baseline 모델 '{model_name}'을(를) 생성합니다 (사전 훈련 가중치: {'사용' if pretrained else '미사용'}).")
    
    # Pure CNN
    if model_name == 'mobilenet_v4_s':
        model = timm.create_model('mobilenetv4_conv_small', pretrained=pretrained, num_classes=num_labels)

    # Full self-attn
    elif model_name == 'deit_tiny':
        model = timm.create_model('deit_tiny_patch16_224', pretrained=pretrained, num_classes=num_labels)

    # Hybrid
    elif model_name == 'mobile_vit_xxs':
        model = timm.create_model('mobilevit_xxs', pretrained=pretrained, num_classes=num_labels)
    
    # Efficient Transformer
    elif model_name == 'efficientvit_b0':
        model = timm.create_model('efficientvit_b0', pretrained=pretrained, num_classes=num_labels)

    # Domain-specific
    elif model_name == 'xie2019':
        model = Xie2019(num_classes=num_labels)
    
    # elif model_name == 'shufflenet_v2_x0_5':
    #     model = torchvision_models.shufflenet_v2_x0_5(weights=torchvision_models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pretrained else None)
    #     num_ftrs = model.fc.in_features
    #     model.fc = nn.Linear(num_ftrs, num_labels)

    # elif model_name == 'deit_tiny_distilled' :
    #     model = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=pretrained, num_classes=num_labels)
    
    # elif model_name == 'edgenext_xxs':
    #     model = timm.create_model('edgenext_xx_small', pretrained=pretrained, num_classes=num_labels)
    
    else:
        raise ValueError(f"지원하지 않는 baseline 모델 이름입니다: {model_name}")    
    return model

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
        for images, labels, _ in progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            
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
        logging.warning("평가 데이터가 없습니다. 평가를 건너뜁니다.")
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'loss': float('inf'), 'labels': [], 'preds': []}

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    
    if desc.startswith("[Valid]"):
        acc_label = "Val Acc"
        log_message = f'{desc} | Loss: {avg_loss:.4f} | {acc_label}: {accuracy:.2f}%'
    else:
        acc_label = "Test Acc"
        log_message = f'{desc} Loss: {avg_loss:.4f} | {acc_label}: {accuracy:.2f}%'
    logging.info(log_message)

    if log_class_metrics and class_names:
        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        for i, class_name in enumerate(class_names):
            log_line = (f"[Metrics for '{class_name}'] | "
                        f"Precision: {precision_per_class[i]:.4f} | "
                        f"Recall: {recall_per_class[i]:.4f} | "
                        f"F1: {f1_per_class[i]:.4f}")
            logging.info(log_line)

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_per_class': f1_per_class if log_class_metrics and class_names else None,
        'labels': all_labels,
        'preds': all_preds
    }

def train(run_cfg, train_cfg, baseline_cfg, config, model, optimizer, scheduler, train_loader, valid_loader, device, run_dir_path, class_names, pos_weight, epoch_offset=0, best_metric=None):
    """모델 훈련 및 검증을 수행하고 최고 성능 모델을 저장합니다."""
    logging.info("="*50)
    logging.info("train 모드를 시작합니다.")
    model_path = os.path.join(run_dir_path, run_cfg.pth_best_name)

    # --- 손실 함수 설정 ---
    loss_function_name = getattr(train_cfg, 'loss_function', 'CrossEntropyLoss').lower()
    if loss_function_name == 'bcewithlogitsloss':
        # BCEWithLogitsLoss는 [B, 1] 형태의 출력을 기대하므로 모델의 마지막 레이어 수정이 필요할 수 있습니다.
        # 이 코드에서는 num_labels=2를 가정하고, 출력을 [B, 2]에서 [B, 1]로 변환하여 사용합니다.
        
        # --- 모델 아키텍처에 따라 마지막 분류 레이어를 동적으로 찾기 ---
        last_layer = None
        if hasattr(model, 'fc'): # ResNet 계열
            last_layer = model.fc
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential) and isinstance(model.classifier[-1], nn.Linear): # EfficientNet 계열
            last_layer = model.classifier[-1]
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear): # timm으로 생성된 MobileNetV4 계열
            last_layer = model.classifier
        elif hasattr(model, 'head'): # timm으로 생성된 ViT 계열
            last_layer = model.head
        
        if last_layer is None:
            logging.warning("모델의 마지막 분류 레이어를 자동으로 찾을 수 없습니다. BCE 손실 함수 사용 시 num_labels 확인을 건너뜁니다.")
        elif last_layer.out_features != 2:
            logging.warning(f"BCE 손실 함수는 이진 분류(num_labels=2)에 최적화되어 있습니다. 현재 num_labels={last_layer.out_features}")
        
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
    else: # 'crossentropyloss' 또는 기본값
        raise ValueError(f"baseline.py에서 지원하지 않는 손실 함수입니다: {loss_function_name}")

    best_model_criterion = getattr(train_cfg, 'best_model_criterion', 'F1_average')
    # best_metric이 인자로 전달되지 않은 경우에만 초기화합니다.
    if best_metric is None:
        best_metric = 0.0 if best_model_criterion != 'val_loss' else float('inf')
        logging.info(f"Best metric을 초기값({best_metric})으로 설정합니다.")
    is_best_saved = False # 베스트 모델이 저장되었는지 확인하는 플래그


    for epoch in range(train_cfg.epochs):
        logging.info("-" * 50)
        model.train()
        if optimizer and hasattr(optimizer, 'train'):
            optimizer.train()

        running_loss = 0.0
        correct = 0
        total = 0
        
        # 에포크 시작 시 Learning Rate 로깅
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"[LR]    [{epoch + 1 + epoch_offset}/{train_cfg.epochs + epoch_offset}] | Learning Rate: {current_lr:.6f}")

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1 + epoch_offset}/{train_cfg.epochs + epoch_offset} [Training]", leave=False, disable=not getattr(run_cfg, 'show_log', True))
        for images, labels, _ in progress_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            outputs = model(images)
            if loss_function_name == 'bcewithlogitsloss':
                loss = criterion(outputs[:, 1].unsqueeze(1), labels.float().unsqueeze(1))
            else: # crossentropyloss
                loss = criterion(outputs, labels)
            loss.backward()

            # Network Slimming을 위한 L1 정규화 손실
            is_slimming_pretrain = getattr(baseline_cfg, 'use_slimming_pruning', False) and \
                                    train_cfg.epochs == config.get('training_baseline', {}).get('epochs')

            if is_slimming_pretrain:
                l1_loss = torch.tensor(0., device=device)
                slimming_l1_strength = 1e-4
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        l1_loss += torch.norm(module.weight, 1)
                loss += slimming_l1_strength * l1_loss

            optimizer.step()

            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            running_loss += loss_val

            _, predicted = torch.max(outputs.data, 1) # outputs는 로짓이므로 그대로 사용
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            step_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{step_lr:.6f}")

        train_acc = 100 * correct / total
        logging.info(f'[Train] [{epoch + 1 + epoch_offset}/{train_cfg.epochs + epoch_offset}] | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%')
        
        eval_results = evaluate(run_cfg, model, valid_loader, device, criterion, loss_function_name, desc=f"[Valid] [{epoch + 1 + epoch_offset}/{train_cfg.epochs + epoch_offset}]", class_names=class_names, log_class_metrics=True)

        current_metric = 0.0
        if best_model_criterion == 'val_loss':
            current_metric = eval_results['loss']
            is_best = current_metric < best_metric
        else: # F1 score variants
            if best_model_criterion == 'F1_Normal' and eval_results['f1_per_class'] is not None:
                current_metric = eval_results['f1_per_class'][0] # Assuming 'Normal' is the first class
            elif best_model_criterion == 'F1_Defect' and eval_results['f1_per_class'] is not None and len(eval_results['f1_per_class']) > 1:
                current_metric = eval_results['f1_per_class'][1] # Assuming 'Defect' is the second class
            else: # 'F1_average' or default
                current_metric = eval_results['f1_macro']
            is_best = current_metric > best_metric
        
        # 최고 성능 모델 저장
        if is_best:
            best_metric = current_metric
            # thop.profile에 의해 오염될 수 있는 state_dict를 저장하기 전 정리합니다.
            # 'total_ops' 또는 'total_params'로 끝나는 키를 제거하여 순수한 가중치만 저장합니다.
            clean_state_dict = {k: v for k, v in model.state_dict().items() if not k.endswith(('total_ops', 'total_params'))}
            torch.save(clean_state_dict, model_path)
            criterion_name = best_model_criterion.replace('_', ' ')
            is_best_saved = True
            logging.info(f"[Best Model Saved] ({criterion_name}: {best_metric:.4f}) -> '{model_path}'")
        
        if scheduler:
            scheduler.step()
    
    # 만약 훈련 동안 한 번도 best model이 저장되지 않았다면(e.g., loss가 계속 nan), 마지막 모델이라도 저장합니다.
    if not is_best_saved:
        # 여기에서도 state_dict를 정리하여 저장합니다.
        clean_state_dict = {k: v for k, v in model.state_dict().items() if not k.endswith(('total_ops', 'total_params'))}
        torch.save(clean_state_dict, model_path)
        logging.warning(f"훈련 동안 성능 개선이 없어 Best 모델이 저장되지 않았습니다. 마지막 에포크의 모델을 '{model_path}'에 저장합니다.")
    return best_metric

def inference(run_cfg, model_cfg, model, data_loader, device, run_dir_path, timestamp, mode_name="Inference", class_names=None, output_dir=None):
    """저장된 모델로 추론 및 성능 평가를 수행합니다."""
    
    # 결과 저장 경로 설정 (output_dir이 주어지면 그곳에, 아니면 run_dir_path에 저장)
    save_dir = output_dir if output_dir else run_dir_path
    
    logging.info("="*50)
    logging.info(f"{mode_name} 모드를 시작합니다.")

    model_path = os.path.join(run_dir_path, run_cfg.pth_best_name)

    # 'inference' 모드로 단독 실행될 때만 가중치를 불러옵니다.
    # 'train' 모드 후 'Test'로 호출될 때는 이미 최적 가중치가 적용된 모델 객체가 전달됩니다.
    if run_cfg.mode == 'inference':
        if not os.path.exists(model_path):
            logging.error(f"모델 파일('{model_path}')을 찾을 수 없습니다. 'pth_inference_dir' 경로를 확인하세요.")
            return None # final_acc가 None이 되도록 반환
        try:
            # [수정] main 함수에서 Pruning이 재적용된 모델에 가중치를 로드합니다.
            model.load_state_dict(torch.load(model_path, map_location=device))
            logging.info(f"'{model_path}' 가중치 로드 완료.")
        except RuntimeError as e:
            logging.error(f"모델 가중치 로딩 중 런타임 오류 발생: {e}")
            logging.error("이 오류는 보통 Pruning된 모델 구조와 불러오려는 가중치가 일치하지 않을 때 발생합니다.")
            logging.error("훈련 시 사용된 Pruning 옵션이 현재 config.yaml에 동일하게 설정되어 있는지 확인하세요.")
            return None
        except Exception as e:
            logging.error(f"모델 가중치 로딩 중 예상치 못한 오류 발생: {e}")
            return None

    model.eval()

    # [추가] FLOPs 및 MACs 측정
    measure_model_complexity(model, model_cfg.img_size, device)

    # --- 평가 또는 순수 추론 ---
    logging.info("테스트 데이터셋에 대한 추론을 시작합니다.")
    only_inference_mode = getattr(run_cfg, 'only_inference', False)

    if only_inference_mode:
        all_filenames, all_predictions, all_confidences = [], [], []
        progress_bar = tqdm(data_loader, desc=f"[{mode_name}]", leave=False, disable=not getattr(run_cfg, 'show_log', True))
        with torch.no_grad():
            for images, _, filenames in progress_bar:
                images = images.to(device)
                outputs = model(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predicted_indices = torch.max(probabilities, 1)
                all_filenames.extend(filenames)
                all_predictions.extend([class_names[p] for p in predicted_indices.cpu().numpy()])
                all_confidences.extend(confidences.cpu().numpy())
        
        results_df = pd.DataFrame({'filename': all_filenames, 'prediction': all_predictions, 'confidence': all_confidences})
        results_df['confidence'] = results_df['confidence'].map('{:.4f}'.format)
        result_csv_path = os.path.join(save_dir, f'inference_results_{timestamp}.csv')
        results_df.to_csv(result_csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"추론 결과가 '{result_csv_path}'에 저장되었습니다.")
        final_acc = None
    else:
        # 추론 시에는 간단한 손실 함수를 임시로 생성하여 전달합니다.
        loss_function_name = getattr(SimpleNamespace(**yaml.safe_load(open('config.yaml', 'r', encoding='utf-8'))['training_baseline']), 'loss_function', 'CrossEntropyLoss').lower()
        if loss_function_name == 'bcewithlogitsloss':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        eval_results = evaluate(run_cfg, model, data_loader, device, criterion, loss_function_name, desc=f"[{mode_name}]", class_names=class_names, log_class_metrics=True)
        final_acc = eval_results['accuracy']

        if eval_results['labels'] and eval_results['preds']:
            plot_and_save_confusion_matrix(eval_results['labels'], eval_results['preds'], class_names, save_dir, timestamp)
    
    return final_acc

# [수정] inference 함수 등 외부에서 호출할 수 있도록 Pruning 관련 함수들을 전역 스코프로 이동
def run_torch_pruning(model, baseline_cfg, model_cfg, device, train_loader=None, criterion=None, extra_ignored_layers=None):
    """torch-pruning 라이브러리를 사용하여 DepGraph 기반 Pruning을 수행합니다."""
    if tp is None:
        logging.error("DepGraph Pruning을 사용하려면 'torch-pruning' 라이브러리를 설치해야 합니다. (pip install torch-pruning)")
        return model
    
    # [추가] Wanda (Pruning by Weights and Activations, ICLR 2024) Importance 구현
    # Wanda는 가중치 크기(|W|)와 입력 활성화 값의 노름(||X||)의 곱을 중요도로 사용합니다.
    # tp가 None이 아님을 확인한 후 정의합니다.
    class WandaImportance(tp.importance.Importance):
        def __init__(self, group_reduction="mean", normalizer=None):
            self.group_reduction = group_reduction
            self.normalizer = normalizer
            self.input_norms = {}

        def _hook(self, module, input, output):
            # 입력 텐서의 채널별 L2 Norm을 계산하여 저장합니다.
            x = input[0]
            # 데이터 배치에 대한 각 뉴런의 활성화 값의 제곱합을 누적하여 전체 데이터셋의 통계를 반영합니다.
            # [동작 원리]
            # 전체 데이터셋에 대한 L2 Norm을 구해야 하지만, 메모리 문제로 배치를 나누어 처리합니다.
            # L2 Norm 공식은 sqrt(sum(x^2))이므로, 각 배치에서 x^2의 합(squared sum)을 계산하여 계속 더해줍니다(Accumulate).
            # 마지막에 __call__ 함수에서 sqrt를 취하면, 전체 데이터를 한 번에 계산한 것과 수학적으로 동일한 결과를 얻습니다.
            if isinstance(module, nn.Linear):
                # x: [Batch, In] or [Batch, Tokens, In]
                if x.dim() == 2:
                    sq_sum = x.pow(2).sum(dim=0)
                else:
                    sq_sum = x.flatten(0, 1).pow(2).sum(dim=0)
                
                if module in self.input_norms:
                    self.input_norms[module] += sq_sum
                else:
                    self.input_norms[module] = sq_sum
            elif isinstance(module, nn.Conv2d):
                # x: [Batch, In, H, W] -> 제곱합 계산
                sq_sum = x.pow(2).sum(dim=(0, 2, 3))
                
                if module in self.input_norms:
                    self.input_norms[module] += sq_sum
                else:
                    self.input_norms[module] = sq_sum

        def __call__(self, group, ch_groups=1):
            group_imp = []
            for dep, idxs in group:
                module = dep.target.module
                pruning_fn = dep.handler
                if module not in self.input_norms: continue
                
                # 누적된 제곱합에 sqrt를 취해 최종 L2 Norm 계산
                # 위 _hook 함수에서 누적한 제곱합(Sum of Squares)에 루트를 씌워
                # 전체 Calibration 데이터셋에 대한 정확한 Global L2 Norm을 구합니다.
                inp_norm = self.input_norms[module].sqrt() # [In]
                layer_weight = module.weight # [Out, In, ...]

                # Pruning Output Channels (dim 0) 인 경우에만 Wanda Score 계산
                if pruning_fn in [tp.pruner.function.prune_conv_out_channels, tp.pruner.function.prune_linear_out_channels]:
                    if isinstance(module, nn.Linear):
                        w_weighted = layer_weight.abs() * inp_norm.view(1, -1)
                    elif isinstance(module, nn.Conv2d):
                        w_weighted = layer_weight.abs() * inp_norm.view(1, -1, 1, 1)
                    
                    # 해당 필터/뉴런의 전체 중요도 합 (L1-like aggregation)
                    imp = w_weighted.flatten(1).sum(dim=1)
                    group_imp.append(imp[idxs])
            
            if len(group_imp) == 0: return None
            return torch.stack(group_imp).sum(dim=0)

    pruning_sparsity = getattr(baseline_cfg, 'pruning_sparsity', 0.5)
    dummy_input = torch.randn(1, 3, model_cfg.img_size, model_cfg.img_size).to(device)

    # --- Pruning 전략 선택 ---
    pruner_class = None
    pruner_kwargs = {}
    if getattr(baseline_cfg, 'use_l1_pruning', False):
        logging.info("torch-pruning을 사용한 L1 Norm Pruning을 시작합니다.")
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_class = tp.pruner.MagnitudePruner
        pruner_kwargs = {'importance': imp, 'pruning_ratio': pruning_sparsity}
    elif getattr(baseline_cfg, 'use_l2_pruning', False):
        logging.info("torch-pruning을 사용한 L2 Norm Pruning을 시작합니다.")
        imp = tp.importance.MagnitudeImportance(p=2)
        pruner_class = tp.pruner.MagnitudePruner
        pruner_kwargs = {'importance': imp, 'pruning_ratio': pruning_sparsity}
    elif getattr(baseline_cfg, 'use_lamp_pruning', False):
        logging.info("LAMP Pruning을 시작합니다.")
        imp = tp.importance.LAMPImportance(p=2, group_reduction="mean")
        pruner_class = tp.pruner.MagnitudePruner
        pruner_kwargs = {'importance': imp, 'pruning_ratio': pruning_sparsity}
    elif getattr(baseline_cfg, 'use_slimming_pruning', False):
        logging.info("Network Slimming (BN-Scale) Pruning을 시작합니다.")
        imp = tp.importance.BNScaleImportance()
        pruner_class = tp.pruner.BNScalePruner
        pruner_kwargs = {'importance': imp, 'pruning_ratio': pruning_sparsity}
    elif getattr(baseline_cfg, 'use_taylor_pruning', False):
        logging.info("Group Taylor Pruning을 시작합니다.")
        imp = tp.importance.TaylorImportance()
        if train_loader is None or criterion is None:
            logging.error("Taylor Pruning을 사용하려면 train_loader와 criterion이 필요합니다.")
            return model
        model.zero_grad()
        logging.info("Taylor 중요도 계산을 위해 샘플 배치에 대한 그래디언트를 계산합니다.")
        images, labels, _ = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        pruner_class = tp.pruner.MagnitudePruner
        pruner_kwargs = {'importance': imp, 'pruning_ratio': pruning_sparsity}
    elif getattr(baseline_cfg, 'use_fpgm_pruning', False):
        logging.info("FPGM Pruning을 시작합니다.")
        imp = tp.importance.FPGMImportance(p=2)
        pruner_class = tp.pruner.MagnitudePruner
        pruner_kwargs = {'importance': imp, 'pruning_ratio': pruning_sparsity}
    elif getattr(baseline_cfg, 'use_wanda_pruning', False):
        logging.info("Wanda Pruning을 시작합니다.")
        if train_loader is None:
            logging.error("Wanda Pruning을 사용하려면 Calibration을 위한 train_loader가 필요합니다.")
            return model
        imp = WandaImportance()
        logging.info("Wanda 중요도 계산을 위해 Calibration을 수행합니다.")
        model.eval()
        hooks = []
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                hooks.append(m.register_forward_hook(imp._hook))
        
        # Calibration을 위해 여러 배치를 사용하여 샘플 수를 확보 (예: 128개 샘플 목표)
        num_wanda_calib_samples = getattr(baseline_cfg, 'num_wanda_calib_samples', 128)
        num_batches = (num_wanda_calib_samples + train_loader.batch_size - 1) // train_loader.batch_size
        logging.info(f"  - Calibration Samples: {num_wanda_calib_samples} (approx {num_batches} batches)")
        
        with torch.no_grad():
            for i, (images, _, _) in enumerate(train_loader):
                if i >= num_batches: break
                model(images.to(device))
                
        for h in hooks: h.remove()
        pruner_class = tp.pruner.MagnitudePruner
        pruner_kwargs = {'importance': imp, 'pruning_ratio': pruning_sparsity}
    else:
        logging.warning("활성화된 torch-pruning 기법이 없습니다.")
        return model

    ignored_layers = []
    last_layer = None
    if hasattr(model, 'fc'):
        last_layer = model.fc
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        last_layer = model.classifier[-1]
    elif hasattr(model, 'classifier'):
        classifier_module = model.classifier
        # ICAM Classifier처럼 projection 내부 마지막 Linear만 head인 경우를 우선 처리
        if hasattr(classifier_module, 'projection') and isinstance(classifier_module.projection, nn.Sequential):
            projection = classifier_module.projection
            if len(projection) > 0 and isinstance(projection[-1], nn.Linear):
                last_layer = projection[-1]
            else:
                last_layer = classifier_module
        elif isinstance(classifier_module, nn.Linear):
            last_layer = classifier_module
        else:
            last_layer = classifier_module
    elif hasattr(model, 'head'):
        last_layer = model.head
    
    if last_layer:
        ignored_layers.append(last_layer)
        logging.info(f"분류 레이어 '{type(last_layer).__name__}'을(를) Pruning 대상에서 제외합니다.")

    is_vit_family = 'vit' in getattr(baseline_cfg, 'model_name', '') or 'deit' in getattr(baseline_cfg, 'model_name', '') or 'swin' in getattr(baseline_cfg, 'model_name', '')
    if is_vit_family:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'qkv' in name:
                ignored_layers.append(module)
        logging.info(f"ViT 계열 모델의 모든 qkv 레이어를 Pruning 대상에서 제외합니다.")

    if extra_ignored_layers:
        ignored_layers.extend([layer for layer in extra_ignored_layers if layer is not None])

    # 중복 제거 (동일 모듈/파라미터가 여러 규칙에 의해 추가될 수 있음)
    unique_ignored_layers = []
    seen = set()
    for layer in ignored_layers:
        layer_id = id(layer)
        if layer_id not in seen:
            unique_ignored_layers.append(layer)
            seen.add(layer_id)
    ignored_layers = unique_ignored_layers

    # unwrapped nn.Parameter(예: learnable_queries)는 pruning dim을 명시해
    # torch-pruning의 자동 추론/경고를 제거합니다.
    unwrapped_parameters = []
    for layer in ignored_layers:
        if isinstance(layer, nn.Parameter):
            non_singleton_dims = [i for i, s in enumerate(layer.shape) if s > 1]
            if len(non_singleton_dims) > 0:
                pruning_dim = non_singleton_dims[-1]
                unwrapped_parameters.append((layer, pruning_dim))

    pruner = pruner_class(
        model,
        dummy_input,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters if len(unwrapped_parameters) > 0 else None,
        **pruner_kwargs
    )

    if getattr(baseline_cfg, 'use_taylor_pruning', False):
        pruner.step(interactive=False)
    else:
        pruner.step()
    logging.info(f"torch-pruning 완료. 모델 구조가 희소도({pruning_sparsity})에 맞춰 변경되었습니다.")
    logging.info("="*50)
    return model

def find_sparsity_for_target_flops(model, baseline_cfg, model_cfg, device, train_loader, criterion, pruning_run_kwargs=None):
    """이진 탐색을 사용하여 목표 FLOPs에 가장 가까운 pruning_sparsity를 찾습니다."""
    target_gflops = getattr(baseline_cfg, 'pruning_flops_target', 0.0)
    if target_gflops <= 0:
        return getattr(baseline_cfg, 'pruning_sparsity', 0.5)

    logging.info("="*80)
    logging.info(f"목표 FLOPs ({target_gflops:.4f} GFLOPs)에 맞는 최적의 Pruning 희소도를 탐색합니다.")

    original_macs, _ = profile(model, inputs=(torch.randn(1, 3, model_cfg.img_size, model_cfg.img_size).to(device),), verbose=False)
    original_gflops = original_macs * 2 / 1e9
    logging.info(f"원본 모델 FLOPs: {original_gflops:.4f} GFLOPs")

    low_sparsity, high_sparsity = 0.0, 0.99
    best_sparsity = 0.0
    min_flops_diff = float('inf')
    original_model = copy.deepcopy(model)
    dummy_input = torch.randn(1, 3, model_cfg.img_size, model_cfg.img_size).to(device)

    for i in range(100):
        current_sparsity = (low_sparsity + high_sparsity) / 2
        temp_model = copy.deepcopy(original_model)
        temp_baseline_cfg = copy.deepcopy(baseline_cfg)
        temp_baseline_cfg.pruning_sparsity = current_sparsity
        
        try:
            pruned_temp_model = run_torch_pruning(
                temp_model,
                temp_baseline_cfg,
                model_cfg,
                device,
                train_loader=train_loader,
                criterion=criterion,
                **(pruning_run_kwargs or {})
            )
            macs, _ = profile(pruned_temp_model, inputs=(dummy_input,), verbose=False)
            current_gflops = macs * 2 / 1e9
            flops_diff = abs(current_gflops - target_gflops)
            reduction_ratio = (1 - current_gflops / original_gflops) * 100
            logging.info(f"  [탐색 {i+1:2d}] 희소도: {current_sparsity:.4f} -> FLOPs: {current_gflops:.4f} GFLOPs (감소율: {reduction_ratio:.2f}%)")

            if flops_diff < min_flops_diff:
                min_flops_diff = flops_diff
                best_sparsity = current_sparsity

            if current_gflops > target_gflops:
                low_sparsity = current_sparsity
            else:
                high_sparsity = current_sparsity
        except Exception as e:
            logging.warning(f"  [탐색 {i+1:2d}] 희소도 {current_sparsity:.4f} 적용 중 오류 발생: {e}. 이 희소도는 건너뜁니다.")
            low_sparsity = current_sparsity

    logging.info(f"탐색 완료. 목표 FLOPs({target_gflops:.4f})에 가장 근접한 최적 희소도는 {best_sparsity:.4f} 입니다.")
    logging.info("="*80)
    return best_sparsity

def find_sparsity_for_target_params(model, baseline_cfg, model_cfg, device, train_loader, criterion, pruning_run_kwargs=None):
    """이진 탐색을 사용하여 목표 파라미터 수에 가장 가까운 pruning_sparsity를 찾습니다."""
    target_m_params = getattr(baseline_cfg, 'pruning_params_target', 0.0)
    if target_m_params <= 0:
        return getattr(baseline_cfg, 'pruning_sparsity', 0.5)

    logging.info("="*80)
    logging.info(f"목표 파라미터 수 ({target_m_params:.4f}M)에 맞는 최적의 Pruning 희소도를 탐색합니다.")

    original_params = sum(p.numel() for p in model.parameters())
    original_m_params = original_params / 1e6
    logging.info(f"원본 모델 파라미터: {original_m_params:.4f}M")

    low_sparsity, high_sparsity = 0.0, 0.99
    best_sparsity = 0.0
    min_params_diff = float('inf')
    original_model = copy.deepcopy(model)

    for i in range(100):
        current_sparsity = (low_sparsity + high_sparsity) / 2
        temp_model = copy.deepcopy(original_model)
        temp_baseline_cfg = copy.deepcopy(baseline_cfg)
        temp_baseline_cfg.pruning_sparsity = current_sparsity
        
        try:
            pruned_temp_model = run_torch_pruning(
                temp_model,
                temp_baseline_cfg,
                model_cfg,
                device,
                train_loader=train_loader,
                criterion=criterion,
                **(pruning_run_kwargs or {})
            )
            current_params = sum(p.numel() for p in pruned_temp_model.parameters())
            current_m_params = current_params / 1e6
            params_diff = abs(current_m_params - target_m_params)
            reduction_ratio = (1 - current_params / original_params) * 100
            logging.info(f"  [탐색 {i+1:2d}] 희소도: {current_sparsity:.4f} -> 파라미터: {current_m_params:.4f}M (감소율: {reduction_ratio:.2f}%)")

            if params_diff < min_params_diff:
                min_params_diff = params_diff
                best_sparsity = current_sparsity

            if current_m_params > target_m_params:
                low_sparsity = current_sparsity
            else:
                high_sparsity = current_sparsity
        except Exception as e:
            logging.warning(f"  [탐색 {i+1:2d}] 희소도 {current_sparsity:.4f} 적용 중 오류 발생: {e}. 이 희소도는 건너뜁니다.")
            low_sparsity = current_sparsity

    logging.info(f"탐색 완료. 목표 파라미터 수({target_m_params:.4f}M)에 가장 근접한 최적 희소도는 {best_sparsity:.4f} 입니다.")
    logging.info("="*80)
    return best_sparsity

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="YAML 설정을 이용한 Baseline 모델 분류기")
    parser.add_argument('--config', type=str, default='config.yaml', help="설정 파일 경로. 기본값: 'config.yaml'")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    run_cfg = SimpleNamespace(**config['run'])
    train_cfg_dict = config['training_baseline']
    train_cfg = SimpleNamespace(**train_cfg_dict)
    model_cfg = SimpleNamespace(**config['model'])
    finetune_cfg_dict = config.get('finetuning_pruned', {})
    finetune_cfg = SimpleNamespace(**finetune_cfg_dict) # Fine-tuning 설정 로드

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

    # Baseline 모델 지정은 training_baseline 섹션을 우선 사용합니다.
    if 'model_name' in train_cfg_dict:
        baseline_cfg_dict['model_name'] = train_cfg_dict['model_name']
    elif 'model_name' in baseline_section_dict:
        baseline_cfg_dict['model_name'] = baseline_section_dict['model_name']

    # Pruning 옵션은 finetuning_pruned 섹션을 우선 사용합니다.
    for key in pruning_keys:
        if key in finetune_cfg_dict:
            baseline_cfg_dict[key] = finetune_cfg_dict[key]
        elif key in baseline_section_dict:
            baseline_cfg_dict[key] = baseline_section_dict[key]

    baseline_cfg = SimpleNamespace(**baseline_cfg_dict)
    # 중첩된 scheduler_params 딕셔너리를 SimpleNamespace로 변환
    if hasattr(train_cfg, 'scheduler_params') and isinstance(train_cfg.scheduler_params, dict):
        train_cfg.scheduler_params = SimpleNamespace(**train_cfg.scheduler_params)
    run_cfg.dataset = SimpleNamespace(**run_cfg.dataset)
    
    # Fine-tuning 설정도 동일하게 변환
    if hasattr(finetune_cfg, 'scheduler_params') and isinstance(finetune_cfg.scheduler_params, dict):
        finetune_cfg.scheduler_params = SimpleNamespace(**finetune_cfg.scheduler_params)
    # --- 전역 시드 고정 ---
    global_seed = getattr(run_cfg, 'global_seed', None)
    if global_seed is not None:
        random.seed(global_seed)
        os.environ['PYTHONHASHSEED'] = str(global_seed)
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)
        torch.cuda.manual_seed(global_seed)
        logging.info(f"전역 랜덤 시드를 {global_seed}로 고정합니다.")

    # Baseline 모델 이름 확인
    baseline_model_name = getattr(baseline_cfg, 'model_name', 'resnet18')

    # --- 로깅 및 디렉토리 설정 ---
    data_dir_name = run_cfg.dataset.name
    if run_cfg.mode == 'train':
        run_dir_path, timestamp, lightweight_option_names = setup_logging(run_cfg, data_dir_name, baseline_model_name, baseline_cfg)
    elif run_cfg.mode == 'inference':
        run_dir_path = getattr(run_cfg, 'pth_inference_dir', None)
        if getattr(run_cfg, 'show_log', True) and (not run_dir_path or not os.path.isdir(run_dir_path)):
            logging.error("추론 모드에서는 'config.yaml'에 'pth_inference_dir'를 올바르게 설정해야 합니다.")
            exit()
        log_dir_path, timestamp, lightweight_option_names = setup_logging(run_cfg, data_dir_name, baseline_model_name, baseline_cfg)
    
    config_str = yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False)
    logging.info("="*50)
    logging.info("config.yaml:")
    logging.info("\n" + config_str)
    logging.info("="*50)
    
    use_cuda_if_available = getattr(run_cfg, 'cuda', True)
    device = torch.device("cuda" if use_cuda_if_available and torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        logging.info(f"CUDA 사용 가능. GPU 사용을 시작합니다. (Device: {torch.cuda.get_device_name(0)})")
    else:
        if use_cuda_if_available:
            logging.warning("config.yaml에서 CUDA 사용이 활성화되었지만, 사용 가능한 CUDA 장치를 찾을 수 없습니다. CPU를 사용합니다.")
        logging.info("CPU 사용을 시작합니다.")

    # --- 데이터 준비 ---
    train_loader, valid_loader, test_loader, num_labels, class_names, pos_weight = prepare_data(run_cfg, train_cfg, model_cfg)

    # --- Baseline 모델 생성 ---
    model = create_baseline_model(baseline_model_name, num_labels, pretrained=train_cfg.pre_trained).to(device)

    # Pruning 호환성을 위해 timm 모델 구조를 수정합니다.
    model = patch_timm_model_for_pruning(model, baseline_model_name, device)

    log_model_parameters(model)
    pruner = None # pruner를 main 함수 스코프에서 정의
    masks = None  # masks를 main 함수 스코프에서 정의
    # --- Pruning 적용 여부 확인 ---
    use_pruning = getattr(baseline_cfg, 'use_l1_pruning', False) or \
                    getattr(baseline_cfg, 'use_l2_pruning', False) or \
                    getattr(baseline_cfg, 'use_fpgm_pruning', False) or \
                    getattr(baseline_cfg, 'use_lamp_pruning', False) or \
                    getattr(baseline_cfg, 'use_depgraph_pruning', False) or \
                    getattr(baseline_cfg, 'use_taylor_pruning', False) or \
                    getattr(baseline_cfg, 'use_slimming_pruning', False) or \
                    getattr(baseline_cfg, 'use_slimming_pruning', False) or \
                    getattr(baseline_cfg, 'use_wanda_pruning', False)
    # --- 옵티마이저 및 스케줄러 생성 함수 ---
    def create_optimizer_and_scheduler(cfg, model):
        optimizer_name = getattr(cfg, 'optimizer', 'adamw').lower()
        if optimizer_name == 'sgd':
            momentum = getattr(cfg, 'momentum', 0.9)
            weight_decay = getattr(cfg, 'weight_decay', 0.0001)
            logging.info(f"옵티마이저: SGD (lr={cfg.lr}, momentum={momentum}, weight_decay={weight_decay})")
            optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=momentum, weight_decay=weight_decay)
        # ... (다른 옵티마이저들 추가) ...
        else: # adamw
            weight_decay = getattr(cfg, 'weight_decay', 0.01)
            logging.info(f"옵티마이저: AdamW (lr={cfg.lr}, weight_decay={weight_decay})")
            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=weight_decay)

        scheduler_params = getattr(cfg, 'scheduler_params', SimpleNamespace())
        scheduler_name = getattr(cfg, 'scheduler', 'none').lower()
        scheduler = None
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

    # --- 모드에 따라 실행 ---
    if run_cfg.mode == 'train':
        # --- 1단계: 사전 훈련 (Pruning 없이) ---
        logging.info("="*80)
        logging.info("단계 1/2: 사전 훈련(Pre-training)을 시작합니다.")
        logging.info("="*80)
        
        optimizer, scheduler = create_optimizer_and_scheduler(train_cfg, model)
        best_metric_from_pretrain = train(run_cfg, train_cfg, baseline_cfg, config, model, optimizer, scheduler, train_loader, valid_loader, device, run_dir_path, class_names, pos_weight)
        
        # --- 2단계: Pruning 및 미세 조정 (Fine-tuning) ---
        if use_pruning:
            logging.info("="*80)
            logging.info("단계 2/2: Pruning 및 미세 조정(Fine-tuning)을 시작합니다.")
            logging.info("="*80)

            # 사전 훈련된 최고 성능 모델 불러오기
            best_model_path = os.path.join(run_dir_path, run_cfg.pth_best_name)
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            logging.info(f"사전 훈련된 모델 '{best_model_path}'을(를) 불러왔습니다.")

            # 목표 FLOPs가 설정된 경우, 최적의 희소도를 자동으로 계산
            if getattr(baseline_cfg, 'pruning_flops_target', 0.0) > 0:
                # 이진 탐색으로 최적의 희소도 찾기
                # Taylor Pruning은 criterion이 필요하므로 임시로 생성
                loss_function_name = getattr(train_cfg, 'loss_function', 'CrossEntropyLoss').lower()
                criterion = nn.CrossEntropyLoss() if loss_function_name != 'bcewithlogitsloss' else nn.BCEWithLogitsLoss() # type: ignore
                optimal_sparsity = find_sparsity_for_target_flops(model, baseline_cfg, model_cfg, device, train_loader, criterion) # type: ignore
                # 찾은 희소도를 설정에 반영
                baseline_cfg.pruning_sparsity = optimal_sparsity
                # 계산된 Pruning 정보를 파일에 저장하여 추론 시 재사용
                pruning_info = {
                    'model_name': getattr(baseline_cfg, 'model_name', 'unknown'),
                    'pruning_method': lightweight_option_names[0] if lightweight_option_names else 'unknown',
                    'target_type': 'flops',
                    'target_value': getattr(baseline_cfg, 'pruning_flops_target'),
                    'optimal_sparsity': optimal_sparsity
                }
                pruning_info_path = os.path.join(run_dir_path, 'pruning_info.yaml')
                with open(pruning_info_path, 'w') as f:
                    yaml.dump(pruning_info, f, default_flow_style=False, sort_keys=False)
                logging.info(f"계산된 Pruning 정보(희소도: {optimal_sparsity:.4f})를 '{pruning_info_path}'에 저장했습니다.")
            # 목표 파라미터 수가 설정된 경우, 최적의 희소도를 자동으로 계산
            elif getattr(baseline_cfg, 'pruning_params_target', 0.0) > 0:
                # 이진 탐색으로 최적의 희소도 찾기
                loss_function_name = getattr(train_cfg, 'loss_function', 'CrossEntropyLoss').lower()
                criterion = nn.CrossEntropyLoss() if loss_function_name != 'bcewithlogitsloss' else nn.BCEWithLogitsLoss() # type: ignore
                optimal_sparsity = find_sparsity_for_target_params(model, baseline_cfg, model_cfg, device, train_loader, criterion) # type: ignore
                # 찾은 희소도를 설정에 반영
                baseline_cfg.pruning_sparsity = optimal_sparsity
                # 계산된 Pruning 정보를 파일에 저장하여 추론 시 재사용
                pruning_info = {
                    'model_name': getattr(baseline_cfg, 'model_name', 'unknown'),
                    'pruning_method': lightweight_option_names[0] if lightweight_option_names else 'unknown',
                    'target_type': 'params',
                    'target_value': getattr(baseline_cfg, 'pruning_params_target'),
                    'optimal_sparsity': optimal_sparsity
                }
                pruning_info_path = os.path.join(run_dir_path, 'pruning_info.yaml')
                with open(pruning_info_path, 'w') as f:
                    yaml.dump(pruning_info, f, default_flow_style=False, sort_keys=False)
                logging.info(f"계산된 Pruning 정보(희소도: {optimal_sparsity:.4f})를 '{pruning_info_path}'에 저장했습니다.")

            # Pruning 적용
            # L1, L2 Pruning도 torch-pruning으로 통합
            use_torch_pruning = getattr(baseline_cfg, 'use_l1_pruning', False) or \
                                getattr(baseline_cfg, 'use_l2_pruning', False) or \
                                getattr(baseline_cfg, 'use_fpgm_pruning', False) or \
                                getattr(baseline_cfg, 'use_taylor_pruning', False) or \
                                getattr(baseline_cfg, 'use_lamp_pruning', False) or \
                                getattr(baseline_cfg, 'use_slimming_pruning', False) or \
                                getattr(baseline_cfg, 'use_wanda_pruning', False) or \
                                getattr(baseline_cfg, 'use_isomorphic_pruning', False)

            # Pruning 전 원본 모델의 FLOPs 측정
            original_macs, _ = profile(model, inputs=(torch.randn(1, 3, model_cfg.img_size, model_cfg.img_size).to(device),), verbose=False)
            # [수정] Pruning 전 원본 모델의 FLOPs 측정 (모델 오염 방지를 위해 복사본 사용)
            logging.info("Pruning 전 원본 모델의 FLOPs를 측정합니다.")
            model_for_profiling = copy.deepcopy(model)
            original_macs, _ = profile(model_for_profiling, inputs=(torch.randn(1, 3, model_cfg.img_size, model_cfg.img_size).to(device),), verbose=False)
            del model_for_profiling # 메모리에서 복사본 해제
            original_gflops = original_macs * 2 / 1e9
            if use_torch_pruning:
                # torch-pruning 계열의 기법은 모델을 직접 수정합니다.
                # Taylor Pruning은 그래디언트 계산을 위해 추가 정보가 필요합니다.
                if getattr(baseline_cfg, 'use_taylor_pruning', False) or getattr(baseline_cfg, 'use_wanda_pruning', False):
                    # 임시 손실 함수 생성
                    loss_function_name = getattr(train_cfg, 'loss_function', 'CrossEntropyLoss').lower()
                    criterion = nn.CrossEntropyLoss() if loss_function_name != 'bcewithlogitsloss' else nn.BCEWithLogitsLoss() # type: ignore
                    model = run_torch_pruning(model, baseline_cfg, model_cfg, device, train_loader=train_loader, criterion=criterion) # type: ignore
                else:
                    model = run_torch_pruning(model, baseline_cfg, model_cfg, device)

                log_model_parameters(model) # Pruning 후 파라미터 수 확인

                # Pruning 후 FLOPs 측정 및 감소율 로깅
                pruned_macs, _ = profile(model, inputs=(torch.randn(1, 3, model_cfg.img_size, model_cfg.img_size).to(device),), verbose=False)
                # Pruning 후 FLOPs 측정 및 감소율 로깅 (모델 오염 방지를 위해 복사본 사용)
                logging.info("Pruning 후 모델의 FLOPs를 측정합니다.")
                model_for_profiling = copy.deepcopy(model)
                pruned_macs, _ = profile(model_for_profiling, inputs=(torch.randn(1, 3, model_cfg.img_size, model_cfg.img_size).to(device),), verbose=False)
                del model_for_profiling # 메모리에서 복사본 해제
                pruned_gflops = pruned_macs * 2 / 1e9
                flops_reduction_ratio = (1 - pruned_gflops / original_gflops) * 100
                logging.info(f"FLOPs가 {original_gflops:.4f} GFLOPs에서 {pruned_gflops:.4f} GFLOPs로 감소했습니다 (감소율: {flops_reduction_ratio:.2f}%).")

                pruner = None # torch-pruning은 NNI pruner 객체를 사용하지 않음.
            
            if use_torch_pruning: # [수정] torch-pruning이 사용된 경우에만 미세조정 실행
                # 미세 조정을 위한 새로운 옵티마이저 및 스케줄러 생성
                logging.info("미세 조정을 위한 새로운 옵티마이저와 스케줄러를 생성합니다.")
                finetune_optimizer, finetune_scheduler = create_optimizer_and_scheduler(finetune_cfg, model)
                
                # 미세 조정 훈련 실행
                # best_metric을 None으로 전달하여 미세 조정 시점에서 초기화하도록 합니다.
                train(run_cfg, finetune_cfg, baseline_cfg, config, model, finetune_optimizer, finetune_scheduler, train_loader, valid_loader, device, run_dir_path, class_names, pos_weight, epoch_offset=train_cfg.epochs, best_metric=None)
            else:
                logging.info("활성화된 Pruning 방법이 없어 미세 조정을 건너뜁니다.")

        logging.info("="*50)
        logging.info("훈련 완료. 최고 성능 모델을 불러와 테스트 세트로 최종 평가합니다.")
        # --- 최종 평가를 위해 깨끗한 모델 객체를 새로 생성하고 가중치를 로드합니다. ---
        # 훈련 과정에서 사용된 model 객체는 FLOPs 측정 등으로 오염되었을 수 있습니다.
        # 따라서, 가장 안정적인 방법은 새로운 모델을 만들고, Pruning 구조를 재현한 뒤, 저장된 가중치를 불러오는 것입니다.
        best_model_path = os.path.join(run_dir_path, run_cfg.pth_best_name)
        if os.path.exists(best_model_path):
            # 1. 깨끗한 모델 골격을 새로 생성합니다.
            logging.info("최종 평가를 위해 새로운 모델 객체를 생성합니다.")
            final_model = create_baseline_model(baseline_model_name, num_labels, pretrained=False).to(device)

            # 새로 생성된 모델에도 Pruning 호환성 패치를 적용합니다.
            final_model = patch_timm_model_for_pruning(final_model, baseline_model_name, device)
            
            # 2. Pruning이 사용되었다면, 새로운 모델에도 동일한 Pruning 구조를 적용합니다.
            # config의 pruning_sparsity 값은 훈련 중 자동 계산된 값을 유지하고 있습니다.
            if use_pruning:
                logging.info("최종 평가 모델에 Pruning 구조를 재적용합니다.")
                # Pruning에 필요한 임시 criterion 생성
                criterion = nn.CrossEntropyLoss()
                final_model = run_torch_pruning(final_model, baseline_cfg, model_cfg, device, train_loader=train_loader, criterion=criterion)

            # 3. 깨끗하게 Pruning된 모델에 저장된 가중치를 불러옵니다.
            # 이 시점에는 모델 객체와 state_dict 모두 thop 오염이 없으므로 오류가 발생하지 않습니다.
            final_model.load_state_dict(torch.load(best_model_path, map_location=device))
            logging.info(f"최고 성능 모델 가중치를 새로 생성된 모델 객체에 로드 완료: '{best_model_path}'")
            
            # 최종 평가에 사용할 모델을 교체합니다.
            model_to_test = final_model
        else:
            logging.warning(f"최고 성능 모델 '{best_model_path}'을(를) 찾을 수 없습니다. 마지막 에포크의 모델로 평가를 진행합니다.")
            # fallback으로 훈련 마지막 상태의 모델을 사용합니다.
            model_to_test = model

        final_acc = inference(run_cfg, model_cfg, model_to_test, test_loader, device, run_dir_path, timestamp, mode_name="Test", class_names=class_names)

        if final_acc is not None:
            log_filename = f"log_{timestamp}.log"
            log_file_path = os.path.join(run_dir_path, log_filename)
            plot_and_save_val_accuracy_graph(log_file_path, run_dir_path, final_acc, timestamp)
            plot_and_save_train_val_accuracy_graph(log_file_path, run_dir_path, final_acc, timestamp)
            plot_and_save_f1_normal_graph(log_file_path, run_dir_path, timestamp, class_names)
            plot_and_save_loss_graph(log_file_path, run_dir_path, timestamp)
            plot_and_save_lr_graph(log_file_path, run_dir_path, timestamp)
            plot_and_save_compiled_graph(run_dir_path, timestamp)

    elif run_cfg.mode == 'inference':
        # --- 추론 모드에서 Pruning 재현 로직 ---
        # 훈련 시 Pruning이 사용되었는지 확인하고 모델 구조를 동일하게 재구성합니다.
        # 1. 훈련 로그 디렉토리에서 pruning_info.yaml을 읽어 희소도를 가져옵니다.
        pruning_info_path = os.path.join(run_dir_path, 'pruning_info.yaml')
        if os.path.exists(pruning_info_path):
            with open(pruning_info_path, 'r') as f:
                pruning_info = yaml.safe_load(f)
            # config.yaml의 값보다 pruning_info.yaml의 값을 우선 적용
            if 'optimal_sparsity' in pruning_info:
                baseline_cfg.pruning_sparsity = pruning_info['optimal_sparsity']
                logging.info(f"'{pruning_info_path}'에서 Pruning 희소도({baseline_cfg.pruning_sparsity:.4f})를 불러왔습니다.")

        # 2. config.yaml 또는 run_info.yaml의 설정을 바탕으로 Pruning 적용
        use_pruning_in_inference = getattr(baseline_cfg, 'use_l1_pruning', False) or \
                                    getattr(baseline_cfg, 'use_l2_pruning', False) or \
                                    getattr(baseline_cfg, 'use_fpgm_pruning', False) or \
                                    getattr(baseline_cfg, 'use_taylor_pruning', False) or \
                                    getattr(baseline_cfg, 'use_lamp_pruning', False) or \
                                    getattr(baseline_cfg, 'use_slimming_pruning', False) or \
                                    getattr(baseline_cfg, 'use_wanda_pruning', False)
        
        if use_pruning_in_inference:
            logging.info("추론 모드: 훈련된 모델의 Pruning 구조를 재현합니다.")
            # Taylor Pruning은 그래디언트 계산을 위해 train_loader와 criterion이 필요합니다.
            # 추론 시에는 정확한 그래디언트 계산이 불가능하므로, L1/L2 등 다른 Pruning 기법 사용을 권장합니다.
            # 여기서는 실행을 위해 임시 criterion을 생성합니다.
            criterion = nn.CrossEntropyLoss()
            model = run_torch_pruning(model, baseline_cfg, model_cfg, device, train_loader=train_loader, criterion=criterion)
            log_model_parameters(model)

        inference(run_cfg, model_cfg, model, test_loader, device, run_dir_path, timestamp, mode_name="Inference", class_names=class_names, output_dir=log_dir_path)


if __name__ == '__main__':
    main()
