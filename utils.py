import logging
import copy
import gc
import torch

try:
    from thop import profile
except ImportError:
    profile = None

def measure_model_complexity(model, img_size, device):
    """
    thop 라이브러리를 사용하여 모델의 FLOPs, MACs, Parameters를 측정하고 로깅합니다.
    """
    if profile:
        try:
            logging.info("FLOPs 및 MACs 측정을 시작합니다...")
            dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
            if device.type == 'cpu':
                dummy_input = dummy_input.to(memory_format=torch.channels_last)
            
            # thop는 모델에 hook을 등록하므로, 원본 모델 오염 방지를 위해 복사본 사용
            model_for_profiling = copy.deepcopy(model)
            model_for_profiling.eval()
            
            macs, params = profile(model_for_profiling, inputs=(dummy_input,), verbose=False)
            
            gmacs = macs / 1e9
            gflops = macs * 2 / 1e9 # 통상적으로 1 MAC = 2 FLOPs (Multiply + Add)
            
            logging.info(f"Model FLOPs: {gflops:.4f} GFLOPs")
            logging.info(f"Model MACs: {gmacs:.4f} GMACs")
            logging.info(f"Model Params (thop): {params / 1e6:.4f} M")
            
            del model_for_profiling
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logging.warning(f"FLOPs/MACs 측정 중 오류 발생: {e}")
    else:
        logging.info("thop 라이브러리가 설치되지 않아 FLOPs/MACs 측정을 건너뜁니다. (pip install thop)")

def log_model_parameters(model):
    """모델의 총 파라미터 수를 계산하고 로깅합니다 (Baseline용)."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("="*50)
    logging.info("모델 파라미터 수:")
    logging.info(f"  - 총 파라미터: {total_params:,} 개")
    logging.info(f"  - 학습 가능한 파라미터: {trainable_params:,} 개")

def log_hybrid_model_parameters(model):
    """HybridModel의 구간별 및 총 파라미터 수를 계산하고 로깅합니다 (Main용)."""
    
    def count_parameters(m):
        if m is None:
            return 0
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    # Encoder 내부를 세분화하여 파라미터 계산
    # 1. Encoder (Encoder) 내부 파라미터 계산
    cnn_feature_extractor = model.encoder.shared_conv[0]
    conv_front_params = count_parameters(cnn_feature_extractor.conv_front)
    conv_1x1_params = count_parameters(cnn_feature_extractor.conv_1x1)
    encoder_norm_params = count_parameters(model.encoder.norm_tokens)
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
    
    feat_to_emb_params = count_parameters(embedding_module.feat_to_emb)
    
    w_k_init_params = 0
    w_v_init_params = 0
    if hasattr(embedding_module, 'W_K_init'):
        w_k_init_params = count_parameters(embedding_module.W_K_init)
    if hasattr(embedding_module, 'W_V_init'):
        w_v_init_params = count_parameters(embedding_module.W_V_init)

    # Embedding4Decoder의 파라미터 총합 (내부 Decoder 레이어 제외)
    embedding4decoder_total_params = feat_to_emb_params + w_k_init_params + w_v_init_params + query_params + pe_params

    # Decoder 내부의 트랜스포머 레이어 파라미터 계산
    decoder_layers_params = count_parameters(model.decoder.embedding4decoder.decoder)
    decoder_total_params = embedding4decoder_total_params + decoder_layers_params

    # 3. Classifier (MLP) 파라미터 계산
    classifier_projection_params = count_parameters(model.classifier.projection)
    classifier_total_params = classifier_projection_params

    total_params = encoder_total_params + decoder_total_params + classifier_total_params

    logging.info("="*50)
    logging.info(f"모델 파라미터 수: {total_params:,} 개")
    logging.info(f"  - Encoder (Encoder):         {encoder_total_params:,} 개")
    logging.info(f"    - conv_front (CNN Backbone):        {conv_front_params:,} 개")
    logging.info(f"    - 1x1_conv (Channel Proj):          {conv_1x1_params:,} 개")
    logging.info(f"    - norm (LayerNorm):                 {encoder_norm_params:,} 개")
    logging.info(f"  - Decoder (Cross-Attention-based):    {decoder_total_params:,} 개")
    logging.info(f"    - Embedding Layer (feat_to_emb):   {feat_to_emb_params:,} 개")
    logging.info(f"    - Init Key Proj (W_K_init):         {w_k_init_params:,} 개")
    logging.info(f"    - Init Value Proj (W_V_init):       {w_v_init_params:,} 개")
    logging.info(f"    - Learnable Queries:                {query_params:,} 개")
    logging.info(f"    - Decoder Layers:                   {decoder_layers_params:,} 개")
    logging.info(f"  - Classifier (Projection MLP):        {classifier_total_params:,} 개")
