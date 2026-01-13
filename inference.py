import os
import argparse
import yaml
import logging
from datetime import datetime
import torch
import pandas as pd
from tqdm import tqdm
from types import SimpleNamespace
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import timm
from torchvision import models as torchvision_models
import types
import torch.nn.functional as F

# timm layers
try:
    from timm.layers import Attention, BatchNormAct2d
except ImportError:
    Attention = None
    BatchNormAct2d = None

# Pruning 라이브러리
try:
    import torch_pruning as tp
except ImportError:
    tp = None

# ONNX 관련 라이브러리
try:
    import onnxruntime
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormatt_pre_process
    import onnx
    from onnxconverter_common import float16
except ImportError:
    onnxruntime = None
    quantize_static = None
    quant_pre_process = None
    CalibrationMethod = None
    onnx = None
    float16 = None

# =============================================================================
# 제안 모델 임포트
# =============================================================================
from models import Model as DecoderBackbone, PatchConvEncoder, Classifier, HybridModel
from dataloader import prepare_data, InferenceImageDataset
from onnx_utils import ONNXCalibrationDataReader, measure_cpu_peak_memory_during_inference

def setup_logging(run_cfg, data_dir_name):
    """
    로그 파일을 log 폴더에 생성합니다.
    """
    show_log = getattr(run_cfg, 'show_log', True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not show_log:
        logging.disable(logging.CRITICAL)
        return '.', timestamp

    run_dir_name = f"inference_{timestamp}"
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
    return run_dir_path, timestamp

def dict_to_namespace(d):
    """Dictionary를 재귀적으로 SimpleNamespace로 변환합니다."""
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = dict_to_namespace(v)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    else:
        return d

def get_default_config():
    """config.yaml이 없을 경우 사용할 기본 설정값을 반환합니다."""
    return {
        'run': {
            'show_log': True,
            'cuda': True,
            'num_workers': 4,
            'use_fp16_inference': False,
            'use_int8_inference': False,
            'dataset': {
                'name': 'Inference_Run',
            }
        },
        'training_run': {},
        'model': {
            'img_size': 224,
            'patch_size': 56,
            'stride': 56,
            'cnn_feature_extractor': {'name': 'efficientnet_b0_feat2'},
            'featured_patch_dim': 24,
            'emb_dim': 24,
            'num_heads': 2,
            'num_decoder_layers': 2,
            'num_decoder_patches': 1,
            'adaptive_initial_query': True,
            'decoder_ff_ratio': 2,
            'dropout': 0.1,
            'positional_encoding': True,
            'res_attention': True,
            'save_attention': True,
            'num_plot_attention': 0
        }
    }

# =============================================================================
# Baseline 모델 관련 클래스 및 함수 (baseline.py에서 이식)
# =============================================================================
class Xie2019(nn.Module):
    def __init__(self, num_classes, dropout_rate = 0.6):
        super(Xie2019, self).__init__()
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
        self.avgpool = nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False)
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

def create_baseline_model(model_name, num_labels, pretrained=False):
    """지정된 이름의 Baseline 모델을 생성합니다."""
    logging.info(f"Baseline 모델 '{model_name}'을(를) 생성합니다.")

    if model_name == 'resnet18':
        model = torchvision_models.resnet18(weights=torchvision_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_labels)    
    elif model_name == 'efficientnet_b0':
        model = torchvision_models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_labels)
    elif model_name == 'mobilenet_v4_s':
        model = timm.create_model('mobilenetv4_conv_small', pretrained=pretrained, num_classes=num_labels)
    elif model_name == 'xie2019':
        model = Xie2019(num_classes=num_labels)
    elif model_name in ['vit', 'deit_tiny', 'mobile_vit_s', 'mobile_vit_xs', 'mobile_vit_xxs']:
        # timm 모델 매핑
        timm_map = {
            'vit': 'vit_base_patch16_224',
            'deit_tiny': 'deit_tiny_patch16_224',
            'mobile_vit_s': 'mobilevit_s',
            'mobile_vit_xs': 'mobilevit_xs',
            'mobile_vit_xxs': 'mobilevit_xxs'
        }
        model = timm.create_model(timm_map[model_name], pretrained=pretrained, num_classes=num_labels)
    else:
        raise ValueError(f"지원하지 않는 baseline 모델 이름입니다: {model_name}")
    return model

def patch_timm_model_for_pruning(model, model_name, device):
    """timm 모델의 Pruning 호환성을 위해 모델 구조를 수정(monkey-patch)합니다."""
    def patched_attention_forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if hasattr(self, 'q_norm'): q, k = self.q_norm(q), self.k_norm(k)
        if hasattr(F, 'scaled_dot_product_attention') and getattr(self, 'fused_attn', False):
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.)
        else:
            scale = getattr(self, 'scale', 1.0 / (self.head_dim ** 0.5))
            q = q * scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    is_vit_family = model_name in ['vit', 'deit_tiny', 'mobile_vit_s', 'mobile_vit_xs', 'mobile_vit_xxs']
    if is_vit_family and Attention is not None:
        for module in model.modules():
            if isinstance(module, Attention):
                module.forward = types.MethodType(patched_attention_forward, module)

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

def run_torch_pruning(model, baseline_cfg, model_cfg, device):
    """torch-pruning을 사용하여 모델 구조를 변경합니다."""
    if tp is None:
        logging.error("Pruning을 적용하려면 'torch-pruning' 라이브러리가 필요합니다.")
        return model
    
    pruning_sparsity = getattr(baseline_cfg, 'pruning_sparsity', 0.0)
    dummy_input = torch.randn(1, 3, model_cfg.img_size, model_cfg.img_size).to(device)

    # Pruning 전략 선택
    if getattr(baseline_cfg, 'use_l1_pruning', False):
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_class = tp.pruner.MagnitudePruner
    elif getattr(baseline_cfg, 'use_fpgm_pruning', False):
        imp = tp.importance.FPGMImportance(p=2)
        pruner_class = tp.pruner.MagnitudePruner
    else:
        logging.warning("지원하지 않는 Pruning 방식이거나 설정되지 않았습니다.")
        return model

    # 무시할 레이어 설정
    ignored_layers = []
    last_layer = None
    if hasattr(model, 'fc'): last_layer = model.fc
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential): last_layer = model.classifier[-1]
    elif hasattr(model, 'classifier'): last_layer = model.classifier
    elif hasattr(model, 'head'): last_layer = model.head
    
    if last_layer: ignored_layers.append(last_layer)

    # ViT 계열 qkv 제외
    model_name = getattr(baseline_cfg, 'model_name', '')
    is_vit_family = 'vit' in model_name or 'deit' in model_name
    if is_vit_family:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'qkv' in name:
                ignored_layers.append(module)

    pruner = pruner_class(
        model,
        dummy_input,
        importance=imp,
        pruning_ratio=pruning_sparsity,
        ignored_layers=ignored_layers,
    )
    
    pruner.step()
    logging.info(f"Pruning 적용 완료 (Method: {'L1' if getattr(baseline_cfg, 'use_l1_pruning', False) else 'FPGM'}, Sparsity: {pruning_sparsity})")
    return model

def main():
    """Inference Only 모듈 메인 함수"""
    # 1. 설정 파일 로드
    parser = argparse.ArgumentParser(description="Inference Only Module (CSV Save)")
    parser.add_argument('--img_dir', type=str, required=True, help='이미지 폴더 경로 (필수)')
    parser.add_argument('--model', type=str, required=True, help='모델 가중치(.pth) 파일 경로 (필수)')
    parser.add_argument('--config', type=str, default=None, help='설정 파일 경로 (선택)')
    
    # Baseline & Pruning 관련 인자 추가
    parser.add_argument('--pruning_info', type=str, default=None, help='pruning_info.yaml 파일 경로 (선택)')
    parser.add_argument('--baseline_name', type=str, default=None, help='Baseline 모델 이름 (예: efficientnet_b0). pruning_info가 없을 때 필수.')
    parser.add_argument('--use_int8', action='store_true', help='INT8 양자화 추론 사용 (ONNX)')
    parser.add_argument('--use_fp16', action='store_true', help='FP16 양자화 추론 사용 (ONNX)')
    
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        logging.info("설정 파일(config.yaml)이 제공되지 않았거나 찾을 수 없습니다. 기본 설정을 사용합니다.")
        config = get_default_config()

    cfg = dict_to_namespace(config)
    run_cfg = cfg.run
    model_cfg = cfg.model

    # CLI 인자로 양자화 설정 오버라이드
    if args.use_int8:
        run_cfg.use_int8_inference = True
    if args.use_fp16:
        run_cfg.use_fp16_inference = True

    # 2. 로깅 및 디렉토리 설정
    # img_dir의 폴더명을 데이터셋 이름으로 사용
    data_dir_name = os.path.basename(os.path.normpath(args.img_dir))
    run_dir_path, timestamp = setup_logging(run_cfg, data_dir_name)
    
    logging.info("="*50)
    logging.info(f"Inference.py Started (Timestamp: {timestamp})")
    logging.info("Configured for ONLY INFERENCE mode (No label comparison).")
    logging.info("="*50)

    # 3. 디바이스 설정
    use_cuda = getattr(run_cfg, 'cuda', True)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # 4. 데이터 준비
    logging.info(f"Loading data from '{args.img_dir}'...")
    
    # 독립 실행을 위한 Transform 정의 (Sewer-ML 표준 전처리)
    transform = transforms.Compose([
        transforms.Resize((model_cfg.img_size, model_cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    ])

    dataset = InferenceImageDataset(img_dir=args.img_dir, transform=transform)
    if len(dataset) == 0:
        logging.error(f"'{args.img_dir}' 폴더에 이미지가 없습니다.")
        return

    test_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=getattr(run_cfg, 'num_workers', 4))
    num_labels = 2
    class_names = ['Defect', 'Normal']

    # 5. 모델 구성 (Baseline vs Hybrid 분기)
    # pruning_info가 있거나 model_name이 명시되면 Baseline 모드로 동작
    is_baseline_mode = (args.pruning_info is not None) or (args.baseline_name is not None)

    if is_baseline_mode:
        logging.info("Building BASELINE model...")
        
        # Baseline 설정 초기화
        baseline_cfg = SimpleNamespace()
        
        # 1. pruning_info.yaml 로드 및 설정
        if args.pruning_info and os.path.exists(args.pruning_info):
            with open(args.pruning_info, 'r') as f:
                pruning_info = yaml.safe_load(f)
            
            # YAML 정보 파싱
            baseline_cfg.model_name = pruning_info.get('model_name', args.baseline_name)
            baseline_cfg.pruning_sparsity = pruning_info.get('optimal_sparsity', 0.0)
            pruning_method = pruning_info.get('pruning_method', 'unknown')
            
            # Pruning Method 매핑
            baseline_cfg.use_l1_pruning = (pruning_method == 'l1')
            baseline_cfg.use_fpgm_pruning = (pruning_method == 'fpgm')
            
            logging.info(f"Loaded pruning info from '{args.pruning_info}': Model={baseline_cfg.model_name}, Method={pruning_method}, Sparsity={baseline_cfg.pruning_sparsity}")
        else:
            # pruning_info가 없으면 순수 Baseline 모델 (Unpruned)
            if not args.baseline_name:
                logging.error("Baseline 모드 실행 시 '--pruning_info'가 없으면 '--baseline_name'을 반드시 지정해야 합니다.")
                return
            baseline_cfg.model_name = args.baseline_name
            baseline_cfg.pruning_sparsity = 0.0
            baseline_cfg.use_l1_pruning = False
            baseline_cfg.use_fpgm_pruning = False
            logging.info(f"No pruning info provided. Using unpruned baseline model: {baseline_cfg.model_name}")

        # 2. 모델 생성
        model = create_baseline_model(baseline_cfg.model_name, num_labels, pretrained=False).to(device)
        model = patch_timm_model_for_pruning(model, baseline_cfg.model_name, device)

        # 3. Pruning 적용 (필요 시)
        if baseline_cfg.use_l1_pruning or baseline_cfg.use_fpgm_pruning:
            logging.info("Applying pruning structure to match trained model...")
            model = run_torch_pruning(model, baseline_cfg, model_cfg, device)

    else:
        logging.info("Building HYBRID (Proposed) model...")
        num_patches_per_dim = (model_cfg.img_size - model_cfg.patch_size) // model_cfg.stride + 1
        num_encoder_patches = num_patches_per_dim ** 2
        
        decoder_params = {
            'num_encoder_patches': num_encoder_patches,
            'num_labels': num_labels,
            'num_decoder_layers': model_cfg.num_decoder_layers,
            'num_decoder_patches': model_cfg.num_decoder_patches,
            'featured_patch_dim': model_cfg.featured_patch_dim,
            'adaptive_initial_query': getattr(model_cfg, 'adaptive_initial_query', False),
            'emb_dim': model_cfg.emb_dim,
            'num_heads': model_cfg.num_heads,
            'decoder_ff_ratio': model_cfg.decoder_ff_ratio,
            'dropout': 0.0, # 추론 시에는 Dropout이 적용되지 않으므로 0.0으로 고정
            'positional_encoding': model_cfg.positional_encoding,
            'res_attention': model_cfg.res_attention,
            'save_attention': False,
        }
        decoder_args = SimpleNamespace(**decoder_params)

        encoder = PatchConvEncoder(img_size=model_cfg.img_size, patch_size=model_cfg.patch_size, stride=model_cfg.stride,
                                    featured_patch_dim=model_cfg.featured_patch_dim, cnn_feature_extractor_name=model_cfg.cnn_feature_extractor.name,
                                    pre_trained=False)
        decoder = DecoderBackbone(args=decoder_args)
        classifier = Classifier(num_decoder_patches=model_cfg.num_decoder_patches,
                                featured_patch_dim=model_cfg.featured_patch_dim, num_labels=num_labels, dropout=0.0)
        model = HybridModel(encoder, decoder, classifier).to(device)

    # 6. 가중치 로드
    # CLI 인자로 받은 경로 사용
    model_path = args.model
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return

    logging.info(f"Loading weights from: {model_path}")

    # 이미 ONNX 파일을 전달받는 경우: ONNX Runtime 세션으로 바로 로드하고
    # PyTorch의 torch.load를 시도하지 않도록 분기 처리합니다.
    model_ext = os.path.splitext(model_path)[1].lower()
    skip_torch_load = False
    if model_ext == '.onnx':
        if not onnxruntime:
            logging.error("ONNX Runtime이 설치되어 있지 않아 .onnx 파일을 직접 로드할 수 없습니다.")
            return
        try:
            sess_options = onnxruntime.SessionOptions()
            # 기본적으로 CPUExecutionProvider 사용 (환경에 따라 변경 가능)
            onnx_session = onnxruntime.InferenceSession(model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
            logging.info(f"Loaded ONNX model into ONNX Runtime session: {model_path}")
            skip_torch_load = True
        except Exception as e:
            logging.error(f"Failed to create ONNX Runtime session from provided .onnx file: {e}")
            return

    if not skip_torch_load:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        except Exception as e:
            logging.error(f"Failed to load model weights: {e}")
            return

    model.eval()

    # --- 양자화(Quantization) 적용 ---
    use_fp16 = getattr(run_cfg, 'use_fp16_inference', False)
    use_int8 = getattr(run_cfg, 'use_int8_inference', False)
    
    # 더미 입력 생성 (ONNX 변환용)
    dummy_input = torch.randn(1, 3, model_cfg.img_size, model_cfg.img_size).to(device)
    single_dummy_input = dummy_input

    if use_int8 and use_fp16:
        logging.error("use_int8_inference과 use_fp16_inference이 동시에 설정되었습니다. 둘 중 하나만 True로 설정해주세요.")
        raise ValueError("Conflicting quantization options: use_int8_inference and use_fp16_inference are both True.")

    onnx_session = None

    if use_int8:
        logging.info("INT8 Static Quantization (ONNX)을 적용합니다.")
        if not onnxruntime:
            logging.error("ONNX Runtime이 설치되지 않아 INT8 양자화를 수행할 수 없습니다.")
            return

        # 1. FP32 ONNX 변환
        fp32_onnx_path = os.path.join(run_dir_path, 'model_fp32_for_quant.onnx')
        model.to('cpu')
        dummy_input_cpu = single_dummy_input.to('cpu')
        
        torch.onnx.export(model, dummy_input_cpu, fp32_onnx_path,
                          export_params=True, opset_version=14,
                          do_constant_folding=True,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        
        # 2. Preprocess
        preprocessed_onnx_path = os.path.join(run_dir_path, 'model_fp32_preprocessed.onnx')
        quant_pre_process(fp32_onnx_path, preprocessed_onnx_path)
        
        # 3. Calibration
        # InferenceImageDataset을 사용하여 Calibration 수행
        calib_samples = getattr(run_cfg, 'int8_calib_samples', 256)
        logging.info(f"Calibration 진행 ({calib_samples} samples)...")
        
        # 데이터셋에서 일부 샘플링
        indices = torch.randperm(len(dataset))[:calib_samples].tolist()
        calib_subset = Subset(dataset, indices)
        calib_loader = DataLoader(calib_subset, batch_size=1, shuffle=False, num_workers=0)
        
        dr = ONNXCalibrationDataReader(calib_loader, input_name='input')
        
        # 4. Quantize
        int8_onnx_path = os.path.join(run_dir_path, 'model_int8.onnx')

        if calib_method_str == 'entropy':
            calib_method = CalibrationMethod.Entropy
        elif calib_method_str == 'percentile':
            calib_method = CalibrationMethod.Percentile
        else:
            calib_method = CalibrationMethod.MinMax
        logging.info(f"Calibration Method: {calib_method_str} ({calib_method})")

        quantize_static(preprocessed_onnx_path, int8_onnx_path, dr, quant_format=QuantFormat.QDQ, per_channel=True, weight_type=QuantType.QInt8, calibrate_method=calib_method)
        
        # 5. Session Create
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        onnx_session = onnxruntime.InferenceSession(int8_onnx_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
    
    elif use_fp16:
        logging.info("FP16 Inference (ONNX)을 적용합니다.")
        if not (onnxruntime and onnx and float16):
            logging.error("ONNX 관련 라이브러리가 부족합니다.")
            return

        # 1. FP32 Export
        fp32_onnx_path = os.path.join(run_dir_path, 'model_fp32_for_fp16.onnx')
        model.to('cpu')
        dummy_input_cpu = single_dummy_input.to('cpu')
        torch.onnx.export(model, dummy_input_cpu, fp32_onnx_path, export_params=True, opset_version=13, do_constant_folding=True, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        
        # 2. Convert to FP16
        fp16_onnx_path = os.path.join(run_dir_path, 'model_fp16.onnx')
        onnx_model = onnx.load(fp32_onnx_path)
        fp16_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
        onnx.save(fp16_model, fp16_onnx_path)
        
        # 3. Session Create
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        onnx_session = onnxruntime.InferenceSession(fp16_onnx_path, sess_options=sess_options, providers=['CPUExecutionProvider'])

    # 7. 추론 실행
    all_filenames = []
    all_predictions = []
    all_confidences = []

    logging.info("Starting inference loop...")
    progress_bar = tqdm(test_loader, desc="Inference", leave=False)
    
    if onnx_session:
        # ONNX Inference
        input_name = onnx_session.get_inputs()[0].name
        input_type = onnx_session.get_inputs()[0].type
        
        # CPU Memory Measurement
        if device.type == 'cpu':
             measure_cpu_peak_memory_during_inference(onnx_session, test_loader, device)

        for images, _, filenames in progress_bar:
            images_np = images.cpu().numpy()
            if 'float16' in input_type:
                images_np = images_np.astype(np.float16)
            
            outputs = onnx_session.run(None, {input_name: images_np})[0]
            
            # Softmax & Argmax (numpy)
            # exp(x - max(x)) for numerical stability
            shift_x = outputs - np.max(outputs, axis=1, keepdims=True)
            exps = np.exp(shift_x)
            probs = exps / np.sum(exps, axis=1, keepdims=True)
            
            preds = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)
            
            all_filenames.extend(filenames)
            all_predictions.extend([class_names[p] for p in preds])
            all_confidences.extend(confs)
    else:
        # PyTorch Inference (FP32)
        with torch.no_grad():
            for images, _, filenames in progress_bar:
                images = images.to(device)
                outputs = model(images)
                
                # Softmax로 확률 계산
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predicted_indices = torch.max(probabilities, 1)
                
                all_filenames.extend(filenames)
                all_predictions.extend([class_names[p] for p in predicted_indices.cpu().numpy()])
                all_confidences.extend(confidences.cpu().numpy())

    # 8. 결과 CSV 저장
    results_df = pd.DataFrame({
        'filename': all_filenames,
        'prediction': all_predictions,
        'confidence': all_confidences
    })
    results_df['confidence'] = results_df['confidence'].map('{:.4f}'.format)
    
    csv_filename = f'inference_results_{timestamp}.csv'
    csv_path = os.path.join(run_dir_path, csv_filename)
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logging.info(f"Inference results saved to: {csv_path}")

if __name__ == '__main__':
    main()
