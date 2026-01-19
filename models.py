import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import timm
except ImportError:  # timm is optional unless you select a timm-based backbone
    timm = None
import math
from torch import Tensor
from torchvision import models

# =============================================================================
# 1. 이미지 인코더 모델 정의
# =============================================================================
class SqueezeExcitation(nn.Module):
    """EfficientNet의 SE Block"""
    def __init__(self, input_channels, squeeze_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.act1 = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.act1(scale)
        scale = self.fc2(scale)
        scale = self.act2(scale)
        return x * scale

class MBConvBlock(nn.Module):
    """EfficientNet의 MBConv Block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_dim = int(in_channels * expand_ratio)
        layers = []

        # 1. Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        # 2. Depthwise convolution phase
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                      padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])

        # 3. Squeeze and Excitation phase (Removed for embedded optimization)

        # 4. Output phase
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class CnnFeatureExtractor(nn.Module):
    """
    다양한 CNN 아키텍처의 앞부분을 특징 추출기로 사용하는 범용 클래스입니다.
    config.yaml의 `cnn_feature_extractor.name` 설정에 따라 모델 구조가 결정됩니다.

    출력: [B, featured_patch_dim, Hf, Wf]
    """

    def __init__(self, cnn_feature_extractor_name='resnet18_layer1', pretrained=True, featured_patch_dim=None):
        super().__init__()
        self.cnn_feature_extractor_name = cnn_feature_extractor_name

        # CNN 모델 이름에 따라 모델과 잘라낼 레이어, 기본 출력 채널을 설정합니다.
        if cnn_feature_extractor_name == 'resnet18_layer1':
            base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.conv_front = nn.Sequential(*list(base_model.children())[:5])  # layer1까지
            base_out_channels = 64
        elif cnn_feature_extractor_name == 'resnet18_layer2':
            base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.conv_front = nn.Sequential(*list(base_model.children())[:6])  # layer2까지
            base_out_channels = 128

        elif cnn_feature_extractor_name == 'mobilenet_v3_small_feat1':
            base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self.conv_front = base_model.features[:2]  # features의 2번째 블록까지
            base_out_channels = 16
        elif cnn_feature_extractor_name == 'mobilenet_v3_small_feat3':
            base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self.conv_front = base_model.features[:4]  # features의 4번째 블록까지
            base_out_channels = 24
        elif cnn_feature_extractor_name == 'mobilenet_v3_small_feat4':
            base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self.conv_front = base_model.features[:5]  # features의 5번째 블록까지
            base_out_channels = 40

        elif cnn_feature_extractor_name == 'efficientnet_b0_feat2':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            self.conv_front = base_model.features[:3]  # features의 3번째 블록까지
            base_out_channels = 24
        elif cnn_feature_extractor_name == 'efficientnet_b0_feat3':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            self.conv_front = base_model.features[:4]  # features의 4번째 블록까지
            base_out_channels = 40

        # --- MobileNetV4 (timm) ---
        elif cnn_feature_extractor_name == 'mobilenet_v4_feat1':
            if timm is None:
                raise ImportError("timm is required for mobilenet_v4_* backbones. Install with: pip install timm")
            base_model = timm.create_model('mobilenetv4_conv_small', pretrained=pretrained, features_only=True, out_indices=(0,))
            self.conv_front = base_model
            base_out_channels = 32  # feat1 출력 채널
        elif cnn_feature_extractor_name == 'mobilenet_v4_feat2':
            if timm is None:
                raise ImportError("timm is required for mobilenet_v4_* backbones. Install with: pip install timm")
            base_model = timm.create_model('mobilenetv4_conv_small', pretrained=pretrained, features_only=True, out_indices=(0, 1))
            self.conv_front = base_model
            base_out_channels = 48  # feat2 출력 채널
        elif cnn_feature_extractor_name == 'mobilenet_v4_feat3':
            if timm is None:
                raise ImportError("timm is required for mobilenet_v4_* backbones. Install with: pip install timm")
            base_model = timm.create_model('mobilenetv4_conv_small', pretrained=pretrained, features_only=True, out_indices=(0, 1, 2))
            self.conv_front = base_model
            base_out_channels = 64  # feat3 출력 채널
        elif cnn_feature_extractor_name == 'mobilenet_v4_feat4':
            if timm is None:
                raise ImportError("timm is required for mobilenet_v4_* backbones. Install with: pip install timm")
            base_model = timm.create_model('mobilenetv4_conv_small', pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
            self.conv_front = base_model
            base_out_channels = 96  # feat4 출력 채널

        elif cnn_feature_extractor_name == 'custom':
            # EfficientNet-B0 feat2 구조를 직접 코드로 구현 (커스터마이징 용도)
            # 주의: 이 옵션은 pretrained 가중치를 자동으로 로드하지 않습니다.
            bn_eps = 1e-5
            bn_momentum = 0.1
            layers = [
                # Stem: 채널3 -> 32, stride 2: 해상도224-> 112
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU6(inplace=True),
                
                # Block 1: MBConv1, 3x3, 32->16, stride 1, expand 1
                MBConvBlock(32, 16, kernel_size=3, stride=1, expand_ratio=1),
                
                # Block 2: MBConv6, 3x3, 16->24, stride 2, expand 6 (2 layers)
                MBConvBlock(16, 24, kernel_size=3, stride=2, expand_ratio=6), # stride2: 해상도 112 -> 56
                MBConvBlock(24, 24, kernel_size=3, stride=1, expand_ratio=6)
            ]
            self.conv_front = nn.Sequential(*layers)
            base_out_channels = 24

        else:
            raise ValueError(f"지원하지 않는 CNN 피처 추출기 이름입니다: {cnn_feature_extractor_name}")

        # 최종 출력 채널 수를 `featured_patch_dim`에 맞추기 위한 1x1 컨볼루션 레이어입니다.
        if featured_patch_dim is not None and featured_patch_dim != base_out_channels:
            self.conv_1x1 = nn.Conv2d(base_out_channels, featured_patch_dim, kernel_size=1)
        else:
            self.conv_1x1 = nn.Identity()

    def forward(self, x):
        x = self.conv_front(x)

        # timm의 features_only=True 모델은 리스트를 반환하므로 마지막 요소만 사용
        if isinstance(x, list):
            x = x[-1]

        x = self.conv_1x1(x)  # 최종 채널 수 조정
        return x


class PatchConvEncoder(nn.Module):
    """Full-frame CNN 1회 + Grid Pooling으로 패치 토큰을 생성하는 인코더.

    - self-attention 없음
    - (제거됨) patch_mixer: 토큰 간 depthwise mixing을 제거하여 latency/memory를 더 안정화

    출력: [B, N, D] (N = num_patches_H * num_patches_W)

    참고: patch_size/stride는 토큰 그리드 크기(H_p, W_p)를 결정하는 하이퍼파라미터로 사용됩니다.
    (기존처럼 실제 패치를 잘라 CNN을 여러 번 돌리지는 않습니다.)
    """

    def __init__(self, grid_size, featured_patch_dim, cnn_feature_extractor_name, pre_trained=True):
        super(PatchConvEncoder, self).__init__()
        self.grid_size = grid_size
        self.featured_patch_dim = featured_patch_dim

        self.num_patches_H = grid_size
        self.num_patches_W = grid_size
        self.num_encoder_patches = self.num_patches_H * self.num_patches_W

        # 1) Full-frame CNN feature extractor (1회)
        #    main.py의 파라미터 로깅 호환을 위해 shared_conv[0]에 extractor가 위치하도록 유지
        self.shared_conv = nn.Sequential(
            CnnFeatureExtractor(
                cnn_feature_extractor_name=cnn_feature_extractor_name,
                pretrained=pre_trained,
                featured_patch_dim=featured_patch_dim,
            )
        )

        # 2) Grid pooling: feature map -> [B, D, H_p, W_p]
        self.grid_pool = nn.AdaptiveAvgPool2d((self.num_patches_H, self.num_patches_W))

        # 3) Token norm
        self.norm = nn.LayerNorm(featured_patch_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B = x.shape[0]

        # 1) Full-frame CNN
        feat = self.shared_conv(x)  # [B, D, Hf, Wf]

        # 2) Grid pooling -> [B, D, H_p, W_p]
        grid = self.grid_pool(feat)

        # 3) Flatten: [B, D, H_p, W_p] -> [B, H_p*W_p, D]
        tokens = grid.permute(0, 2, 3, 1).contiguous().view(B, -1, self.featured_patch_dim)

        # Layer Normalization
        tokens = self.norm(tokens)
        return tokens

# =============================================================================
# 2. 디코더 모델 정의
# =============================================================================

class Embedding4Decoder(nn.Module): 
    """
    Decoder 입력을 위한 임베딩 레이어.
    [Idea 1-1] 2D Sinusoidal Position Encoding을 적용하여 위치 정보를 주입합니다.
    """
    def __init__(self, num_encoder_patches, featured_patch_dim, num_decoder_patches, 
                    grid_size_h, grid_size_w, # 2D PE 생성을 위해 그리드 크기 인자 추가
                    adaptive_initial_query=False, num_decoder_layers=3, emb_dim=128, num_heads=16, 
                    decoder_ff_dim=256, attn_dropout=0., dropout=0., save_attention=False, res_attention=False, positional_encoding=True):
        
        super().__init__()
        
        self.adaptive_initial_query = adaptive_initial_query

        # --- 입력 인코딩 ---
        self.W_feat2emb = nn.Linear(featured_patch_dim, emb_dim)      
        self.W_Q_init = nn.Linear(featured_patch_dim, emb_dim)
        self.dropout = nn.Dropout(dropout, inplace=True)

        # --- 학습 가능한 쿼리(Learnable Query) ---
        self.learnable_queries = nn.Parameter(torch.empty(num_decoder_patches, featured_patch_dim))
        nn.init.xavier_uniform_(self.learnable_queries)
        
        # --- [Idea 1-1] 2D Sinusoidal Positional Encoding ---
        self.use_positional_encoding = positional_encoding
        if self.use_positional_encoding:
            # 고정된 2D Sinusoidal PE 생성 (학습되지 않음)
            pos_embed = self.get_2d_sincos_pos_embed(emb_dim, grid_size_h, grid_size_w)
            # 모델의 state_dict에 저장되지만 optimizer에 의해 업데이트되지 않도록 register_buffer 사용
            self.register_buffer('pos_embed', pos_embed, persistent=False)
        else:
            self.pos_embed = None

        if self.adaptive_initial_query:
            self.W_K_init = nn.Linear(emb_dim, emb_dim)
            self.W_V_init = nn.Linear(emb_dim, emb_dim)

        # --- 디코더 ---
        self.decoder = Decoder(num_encoder_patches, emb_dim, num_heads, num_decoder_patches, decoder_ff_dim=decoder_ff_dim, attn_dropout=attn_dropout, dropout=dropout,
                                res_attention=res_attention, num_decoder_layers=num_decoder_layers, save_attention=save_attention)
        
    def get_2d_sincos_pos_embed(self, embed_dim, grid_h, grid_w):
        """
        2D Grid에 대한 Sinusoidal Positional Embedding을 생성합니다.
        embed_dim의 절반은 Height 정보, 나머지 절반은 Width 정보에 할당합니다.
        """
        assert embed_dim % 2 == 0, "Embedding 차원은 짝수여야 합니다."
        
        # 1D Sinusoidal PE 생성 함수
        def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
            """
            embed_dim: output dimension for each position
            pos: [H*W] list of positions to be encoded
            """
            omega = torch.arange(embed_dim // 2, dtype=torch.float)
            omega /= embed_dim / 2.
            omega = 1. / 10000**omega  # (D/2,)

            pos = pos.reshape(-1)  # (M,)
            out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

            emb_sin = torch.sin(out) # (M, D/2)
            emb_cos = torch.cos(out) # (M, D/2)

            emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
            return emb

        # Grid 좌표 생성
        grid_h_arange = torch.arange(grid_h, dtype=torch.float)
        grid_w_arange = torch.arange(grid_w, dtype=torch.float)
        
        # Meshgrid로 좌표 확장
        grid_w_coords, grid_h_coords = torch.meshgrid(grid_w_arange, grid_h_arange, indexing='xy')
        
        # 각각에 대해 1D PE 생성 (차원의 절반씩 사용)
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h_coords) # [H*W, D/2]
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w_coords) # [H*W, D/2]

        # Concat하여 최종 2D PE 생성 [H*W, D]
        pos_embed = torch.cat([emb_h, emb_w], dim=1)
        return pos_embed.unsqueeze(0) # [1, H*W, D] (Broadcasting을 위해 배치 차원 추가)

    def forward(self, x) -> Tensor:
        # x: [B, num_encoder_patches, featured_patch_dim]
        bs = x.shape[0]

        x = self.W_feat2emb(x) # [B, N, emb_dim]

        # Value용: 위치 인코딩을 더하지 않은 순수 특징 (Content)
        x_clean = self.dropout(x)

        # --- 위치 인코딩 더하기 ---
        if self.use_positional_encoding and self.pos_embed is not None:
            # self.pos_embed: [1, N, emb_dim] -> Broadcasting으로 더해짐
            x = x + self.pos_embed.to(x.device)

        # Key용: 위치 인코딩이 포함된 특징 (Content + Position)
        seq_encoder_patches = self.dropout(x)
        
        # --- 2. 디코더에 입력할 쿼리(Query) 준비 ---
        if self.adaptive_initial_query:
            latent_queries = self.W_Q_init(self.learnable_queries)
            # [Optimization] repeat 대신 expand 사용 (메모리 복사 방지)
            latent_queries = latent_queries.unsqueeze(0).expand(bs, -1, -1)
            
            # Key는 위치 정보 포함(seq_encoder_patches), Value는 순수 특징(x_clean) 사용
            k_init = self.W_K_init(seq_encoder_patches)
            v_init = self.W_V_init(x_clean)
            
            latent_attn_scores = torch.bmm(latent_queries, k_init.transpose(1, 2))
            latent_attn_weights = F.softmax(latent_attn_scores, dim=-1)
            
            seq_decoder_patches = torch.bmm(latent_attn_weights, v_init)
        else:
            learnable_queries = self.W_Q_init(self.learnable_queries)
            # [Optimization] repeat 대신 expand 사용
            seq_decoder_patches = learnable_queries.unsqueeze(0).expand(bs, -1, -1)

        return seq_encoder_patches, x_clean, seq_decoder_patches
            

class Projection4Classifier(nn.Module):
    """디코더의 출력을 받아 최종 분류기가 사용할 수 있는 고정 차원 특징 벡터로 변환합니다.

    기존: [B, Q, emb_dim] -> Linear -> Flatten => [B, Q*D] (Q에 따라 입력 차원이 변함)
    변경: [B, Q, emb_dim] -> Mean Pool(Q) -> Linear => [B, D] (Q와 무관하게 입력 차원 고정)
    """

    def __init__(self, emb_dim, featured_patch_dim):
        super().__init__()
    
        self.linear = nn.Linear(emb_dim, featured_patch_dim)

    def forward(self, x):
        # x: [B, Q, emb_dim]
        x = x.mean(dim=1)  # [B, emb_dim]
        x = self.linear(x)  # [B, featured_patch_dim]
        return x

class Decoder(nn.Module):
    def __init__(self, num_encoder_patches, emb_dim, num_heads, num_decoder_patches, decoder_ff_dim=None, attn_dropout=0., dropout=0.,
                    res_attention=False, num_decoder_layers=1, save_attention=False):
        super().__init__()
        
        self.layers = nn.ModuleList([DecoderLayer(num_encoder_patches, emb_dim, num_decoder_patches, num_heads=num_heads, decoder_ff_dim=decoder_ff_dim, attn_dropout=attn_dropout, dropout=dropout,
                                                        res_attention=res_attention, save_attention=save_attention) for i in range(num_decoder_layers)])
        self.res_attention = res_attention

    def forward(self, seq_encoder_k:Tensor, seq_encoder_v:Tensor, seq_decoder:Tensor):
        scores = None
        if self.res_attention:
            for mod in self.layers: _, seq_decoder, scores = mod(seq_encoder_k, seq_encoder_v, seq_decoder, prev=scores)
            return seq_decoder
        else:
            for mod in self.layers: _, seq_decoder = mod(seq_encoder_k, seq_encoder_v, seq_decoder)
            return seq_decoder

class DecoderLayer(nn.Module):
    def __init__(self, num_encoder_patches, emb_dim, num_decoder_patches, num_heads, decoder_ff_dim=256, save_attention=False,
                    attn_dropout=0, dropout=0., bias=False, res_attention=False): # [Optimization] bias=False 권장
        super().__init__()
        assert not emb_dim%num_heads, f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})"
        
        self.res_attention = res_attention
        self.cross_attn = _MultiheadAttention(emb_dim, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention, qkv_bias=False)
        self.dropout_attn = nn.Dropout(dropout, inplace=True)
        self.norm_attn = nn.LayerNorm(emb_dim)

        # GEGLU -> ReLU, LayerNorm -> BatchNorm1d
        self.ffn = nn.Sequential(nn.Linear(emb_dim, decoder_ff_dim, bias=bias),
                                nn.ReLU6(inplace=True),
                                nn.Dropout(dropout, inplace=True),
                                nn.Linear(decoder_ff_dim, emb_dim, bias=bias)) 
        # Note: GEGLU는 입력을 2배로 쪼개므로 output dim이 절반이었으나, ReLU는 그대로 유지하므로 decoder_ff_dim 전체 사용
        self.dropout_ffn = nn.Dropout(dropout, inplace=True)
        self.norm_ffn = nn.LayerNorm(emb_dim)
        
        self.save_attention = save_attention

    def forward(self, seq_encoder_k:Tensor, seq_encoder_v:Tensor, seq_decoder:Tensor, prev=None) -> Tensor:
        # [Optimization] Pre-Norm Architecture 적용
        # 구조: x = x + Dropout(Sublayer(Norm(x)))
        
        # 1. Cross-Attention Block
        residual = seq_decoder
        seq_decoder = self.norm_attn(seq_decoder) # Norm first

        if self.res_attention:
            decoder_out, attn, scores = self.cross_attn(seq_decoder, seq_encoder_k, seq_encoder_v, prev)
        else:
            decoder_out, attn = self.cross_attn(seq_decoder, seq_encoder_k, seq_encoder_v)
        
        if self.save_attention:
            self.attn = attn
        
        seq_decoder = residual + self.dropout_attn(decoder_out)
        
        # 2. FFN Block
        residual = seq_decoder
        seq_decoder = self.norm_ffn(seq_decoder) # Norm first
        ffn_out = self.ffn(seq_decoder)
        seq_decoder = residual + self.dropout_ffn(ffn_out)
        
        if self.res_attention: return seq_encoder_k, seq_decoder, scores
        else: return seq_encoder_k, seq_decoder

class _MultiheadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=False, save_attention=False, **kwargs):
        super().__init__()
        
        head_dim = emb_dim // num_heads
        self.scale = head_dim**-0.5
        self.num_heads, self.head_dim = num_heads, head_dim

        self.W_Q = nn.Linear(emb_dim, head_dim * num_heads, bias=qkv_bias)
        self.W_K = nn.Linear(emb_dim, head_dim * num_heads, bias=qkv_bias)
        self.W_V = nn.Linear(emb_dim, head_dim * num_heads, bias=qkv_bias)

        self.res_attention = res_attention
        self.save_attention = save_attention
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        self.concatheads2emb = nn.Sequential(nn.Linear(num_heads * head_dim, emb_dim), nn.Dropout(proj_dropout))
        self._sdpa_logged = False

    def forward(self, Q:Tensor, K:Tensor, V:Tensor, prev=None):
        bs = Q.size(0)
        
        q_s = self.W_Q(Q).view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_s = self.W_K(K).view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_s = self.W_V(V).view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # [Optimization] SDPA (Scaled Dot-Product Attention) 적용
        # res_attention이나 save_attention이 필요 없는 경우, PyTorch의 최적화된 커널(FlashAttention 등) 사용
        # if not self.res_attention and not self.save_attention:
        #     if not self._sdpa_logged:
        #         logging.info(f"[Optimization] SDPA (Scaled Dot-Product Attention) is ACTIVE in {self.__class__.__name__}.")
        #         self._sdpa_logged = True
        #     # F.scaled_dot_product_attention은 (B, H, L, D) 입력을 기대함
        #     output = F.scaled_dot_product_attention(q_s, k_s, v_s, dropout_p=self.attn_dropout.p if self.training else 0.)
        #     output = output.permute(0, 2, 1, 3).reshape(bs, -1, self.num_heads * self.head_dim)
        #     output = self.concatheads2emb(output)
        #     return output, None

        # [Optimization] einsum 대신 matmul 사용 (속도 향상)
        # attn_scores = torch.einsum('bhqd, bhkd -> bhqk', q_s, k_s) * self.scale
        attn_scores = torch.matmul(q_s, k_s.transpose(-1, -2)) * self.scale
        
        if prev is not None: attn_scores = attn_scores + prev
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # output = torch.einsum('bhqk, bhkd -> bhqd', attn_weights, v_s)
        output = torch.matmul(attn_weights, v_s)
        # [Optimization] Use reshape instead of contiguous().view() to avoid copy if possible
        output = output.permute(0, 2, 1, 3).reshape(bs, -1, self.num_heads * self.head_dim)
        
        output = self.concatheads2emb(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

class Model(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        
        num_encoder_patches = args.num_encoder_patches 
        num_labels = args.num_labels 
        num_decoder_patches = args.num_decoder_patches 
        self.featured_patch_dim = args.featured_patch_dim 
        adaptive_initial_query = args.adaptive_initial_query 
        emb_dim = args.emb_dim           
        num_heads = args.num_heads           
        num_decoder_layers = args.num_decoder_layers 
        decoder_ff_ratio = args.decoder_ff_ratio 
        dropout = args.dropout           
        attn_dropout = dropout           
        positional_encoding = args.positional_encoding 
        save_attention = args.save_attention     
        res_attention = getattr(args, 'res_attention', False)
        # 2D PE 생성을 위해 그리드 크기 전달 (main.py에서 계산된 값이 있으면 사용)
        grid_size_h = getattr(args, 'grid_size_h', None)
        grid_size_w = getattr(args, 'grid_size_w', None)
        if grid_size_h is None or grid_size_w is None:
            # fallback: 정사각형 그리드라고 가정
            grid_size = int(math.sqrt(num_encoder_patches))
            grid_size_h = grid_size
            grid_size_w = grid_size

        decoder_ff_dim = emb_dim * decoder_ff_ratio 

        self.embedding4decoder = Embedding4Decoder(num_encoder_patches=num_encoder_patches, featured_patch_dim=self.featured_patch_dim, num_decoder_patches=num_decoder_patches, 
                                grid_size_h=grid_size_h, grid_size_w=grid_size_w, # 추가된 인자
                                adaptive_initial_query=adaptive_initial_query,
                                num_decoder_layers=num_decoder_layers, emb_dim=emb_dim, num_heads=num_heads, decoder_ff_dim=decoder_ff_dim, positional_encoding=positional_encoding,
                                attn_dropout=attn_dropout, dropout=dropout,
                                res_attention=res_attention, save_attention=save_attention)

        self.projection4classifier = Projection4Classifier(emb_dim, self.featured_patch_dim)

    def forward(self, x): 
        # x: [B, num_encoder_patches, featured_patch_dim]
        # (PatchConvEncoder의 출력이 여기로 들어옴)
        
        seq_encoder_patches, seq_encoder_clean, seq_decoder_patches = self.embedding4decoder(x)
        z = self.embedding4decoder.decoder(seq_encoder_patches, seq_encoder_clean, seq_decoder_patches)
        features = self.projection4classifier(z)
        return features

# =============================================================================
# 3. 전체 모델 구성
# =============================================================================
class Classifier(nn.Module):
    """디코더 백본의 출력을 받아 최종 클래스 로짓으로 매핑하는 분류기입니다."""
    def __init__(self, num_decoder_patches, featured_patch_dim, num_labels, dropout):
        super().__init__()
        input_dim = featured_patch_dim 
        hidden_dim = (input_dim + num_labels) // 2 

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x):
        x = self.projection(x) 
        return x

class HybridModel(torch.nn.Module):
    """인코더, 디코더, 분류기를 결합한 최종 하이브리드 모델입니다."""
    def __init__(self, encoder, decoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        
    def forward(self, x):
        # 1. 인코딩: 2D 이미지 -> 패치 시퀀스 (여기서 Mixer가 동작)
        x = self.encoder(x)
        # 2. 크로스-어텐션: 패치 시퀀스 -> 특징 벡터
        x = self.decoder(x)
        # 3. 분류: 특징 벡터 -> 클래스 로짓
        out = self.classifier(x)
        return out