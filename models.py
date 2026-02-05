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

# [추가] DropPath (Stochastic Depth) 구현
# timm 라이브러리가 없어도 동작하도록 내장 구현
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

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

    출력: [B, encoder_dim, Hf, Wf]
    """

    def __init__(self, cnn_feature_extractor_name='resnet18_layer1', pretrained=True, encoder_dim=None):
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

        elif cnn_feature_extractor_name == 'custom24':
            # EfficientNet-B0 feat2 구조를 기반으로 코드 구현 (커스터마이징 용도)
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

        elif cnn_feature_extractor_name == 'custom32':
            # EfficientNet-B0 feat2 구조를 기반으로 코드 구현 (커스터마이징 용도)
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

                # Block 2: MBConv6, 3x3, 16->32, stride 2, expand 6 (2 layers)
                MBConvBlock(16, 32, kernel_size=3, stride=2, expand_ratio=6), # stride2: 해상도 112 -> 56
                MBConvBlock(32, 32, kernel_size=3, stride=1, expand_ratio=6)
            ]
            self.conv_front = nn.Sequential(*layers)
            base_out_channels = 32

        else:
            raise ValueError(f"지원하지 않는 CNN 피처 추출기 이름입니다: {cnn_feature_extractor_name}")

        # 최종 출력 채널 수를 `encoder_dim`에 맞추기 위한 1x1 컨볼루션 레이어입니다.
        if encoder_dim is not None and encoder_dim != base_out_channels:
            self.conv_1x1 = nn.Conv2d(base_out_channels, encoder_dim, kernel_size=1)
        else:
            self.conv_1x1 = nn.Identity()

    def forward(self, x):
        x = self.conv_front(x)

        # timm의 features_only=True 모델은 리스트를 반환하므로 마지막 요소만 사용
        if isinstance(x, list):
            x = x[-1]

        x = self.conv_1x1(x)  # 최종 채널 수 조정
        return x


class Encoder(nn.Module):
    """Full-frame CNN 1회 + Grid Pooling으로 패치 토큰을 생성하는 인코더.

    - self-attention 없음
    - (제거됨) patch_mixer: 토큰 간 depthwise mixing을 제거하여 latency/memory를 더 안정화

    출력: [B, N, D] (N = num_patches_H * num_patches_W)

    참고: grid_size는 토큰 그리드 크기(H_p, W_p)를 결정하는 하이퍼파라미터로 사용됩니다.
    (기존처럼 실제 패치를 잘라 CNN을 여러 번 돌리지는 않습니다.)
    """

    def __init__(self, grid_size, encoder_dim, cnn_feature_extractor_name, pre_trained=True):
        super(Encoder, self).__init__()
        self.grid_size = grid_size
        self.encoder_dim = encoder_dim

        self.num_patches_H = grid_size
        self.num_patches_W = grid_size
        self.num_encoder_patches = self.num_patches_H * self.num_patches_W

        # 1) Full-frame CNN feature extractor (1회)
        #    main.py의 파라미터 로깅 호환을 위해 shared_conv[0]에 extractor가 위치하도록 유지
        self.shared_conv = nn.Sequential(
            CnnFeatureExtractor(
                cnn_feature_extractor_name=cnn_feature_extractor_name,
                pretrained=pre_trained,
                encoder_dim=encoder_dim,
            )
        )

        # 2) Grid pooling: feature map -> [B, D, H_p, W_p]
        self.grid_pool = nn.AdaptiveAvgPool2d((self.num_patches_H, self.num_patches_W))

        # 3) Token norm
        self.norm = nn.LayerNorm(encoder_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B = x.shape[0]

        # 1) Full-frame CNN
        feat = self.shared_conv(x)  # [B, D, Hf, Wf]

        # 2) Grid pooling -> [B, D, H_p, W_p]
        grid = self.grid_pool(feat)

        # 3) Flatten: [B, D, H_p, W_p] -> [B, H_p*W_p, D]
        tokens = grid.permute(0, 2, 3, 1).contiguous().view(B, -1, self.encoder_dim)

        # Layer Normalization
        tokens = self.norm(tokens)
        return tokens

# =============================================================================
# 2. 디코더 모델 정의
# =============================================================================

class Embedding4Decoder(nn.Module): 
    """
    Decoder 입력을 위한 임베딩 레이어.

    [Method B Refactor]
    - V는 content-only (positional encoding 미적용)
    - K는 content에 positional encoding을 더한 값을 쓰되,
      (W_K(content + pos) == W_K(content) + W_K(pos)) 선형성을 이용해
      content+pos 토큰 텐서를 별도로 유지하지 않고 K projection 단계에서 pos 성분을 더합니다.

    주의:
    - eval/inference에서의 출력은 기존 구현과 수치적으로 동일합니다(드롭아웃 비활성).
    - training에서 dropout>0이면 기존과 완전히 동일한 dropout 분포를 보장하지는 않습니다.
      (inference 최적화 목적이므로, training 정확도 보장이 필요하면 별도 옵션으로 기존 경로 유지 권장)
    """

    def __init__(
        self,
        num_encoder_patches,
        encoder_dim,
        num_decoder_patches,
        grid_size_h,
        grid_size_w,
        adaptive_initial_query=False,
        num_decoder_layers=3,
        emb_dim=128,
        num_heads=16,
        decoder_ff_dim=256,
        attn_dropout=0.0,
        dropout=0.0,
        drop_path_ratio=0.0,
        save_attention=False,
        res_attention=False,
        positional_encoding=True,
    ):
        super().__init__()

        self.adaptive_initial_query = adaptive_initial_query

        # --- 입력 인코딩 ---
        self.W_feat2emb = nn.Linear(encoder_dim, emb_dim)
        self.dropout = nn.Dropout(dropout, inplace=True)

        # --- 학습 가능한 쿼리(Learnable Query) ---
        self.learnable_queries = nn.Parameter(torch.empty(num_decoder_patches, emb_dim))
        # ViT/BERT 등에서 널리 사용되는 방식인 정규분포 초기화를 적용합니다.
        nn.init.normal_(self.learnable_queries, std=0.02)

        # --- 2D Sinusoidal Positional Encoding (fixed) ---
        self.use_positional_encoding = positional_encoding
        if self.use_positional_encoding:
            pos_embed = self.get_2d_sincos_pos_embed(emb_dim, grid_size_h, grid_size_w)
            self.register_buffer('pos_embed', pos_embed, persistent=False)  # [1, N, D]
        else:
            self.pos_embed = None

        # --- Adaptive initial query projections ---
        if self.adaptive_initial_query:
            self.W_K_init = nn.Linear(emb_dim, emb_dim)
            self.W_V_init = nn.Linear(emb_dim, emb_dim)
            # inference cache (W_K_init(pos_embed))
            self.register_buffer('_k_init_pos_cache', None, persistent=False)  # [1, N, D]

        # --- 디코더 ---
        self.decoder = Decoder(
            num_encoder_patches,
            emb_dim,
            num_heads,
            num_decoder_patches,
            decoder_ff_dim=decoder_ff_dim,
            attn_dropout=attn_dropout,
            dropout=dropout,
            drop_path_ratio=drop_path_ratio,
            res_attention=res_attention,
            num_decoder_layers=num_decoder_layers,
            save_attention=save_attention,
        )

    def get_2d_sincos_pos_embed(self, embed_dim, grid_h, grid_w):
        """2D Grid에 대한 Sinusoidal Positional Embedding 생성."""
        assert embed_dim % 2 == 0, "Embedding 차원은 짝수여야 합니다."

        def get_1d_sincos_pos_embed_from_grid(embed_dim_1d, pos):
            omega = torch.arange(embed_dim_1d // 2, dtype=torch.float)
            omega /= embed_dim_1d / 2.0
            omega = 1.0 / 10000**omega
            pos = pos.reshape(-1)
            out = torch.einsum('m,d->md', pos, omega)
            emb_sin = torch.sin(out)
            emb_cos = torch.cos(out)
            return torch.cat([emb_sin, emb_cos], dim=1)

        grid_h_arange = torch.arange(grid_h, dtype=torch.float)
        grid_w_arange = torch.arange(grid_w, dtype=torch.float)
        grid_w_coords, grid_h_coords = torch.meshgrid(grid_w_arange, grid_h_arange, indexing='xy')

        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h_coords)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w_coords)

        pos_embed = torch.cat([emb_h, emb_w], dim=1)
        return pos_embed.unsqueeze(0)  # [1, H*W, D]

    # ---------------------------------------------------------------------
    # Inference cache builders (call AFTER loading weights, BEFORE ONNX export)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def prepare_inference_caches(self):
        """Method B에서 pos 성분을 projection-space에서 더하기 위한 캐시를 미리 생성합니다.

        - adaptive_initial_query: _k_init_pos_cache = W_K_init(pos_embed)
        - decoder layers: 각 cross-attn 모듈의 k_pos_cache = W_K_layer(pos_embed) (head-split 형태)

        사용 시점:
          model.eval(); model.embedding4decoder.prepare_inference_caches(); torch.onnx.export(...)
        """
        if not self.use_positional_encoding or self.pos_embed is None:
            return

        # 1) adaptive init cache
        if self.adaptive_initial_query:
            dev = self.W_K_init.weight.device
            dt = self.W_K_init.weight.dtype
            pos = self.pos_embed.to(device=dev, dtype=dt)
            self._k_init_pos_cache = self.W_K_init(pos).contiguous()  # [1, N, D]

        # 2) decoder-layer caches
        self.decoder.prepare_inference_caches(self.pos_embed)

    def forward(self, x) -> Tensor:
        # x: [B, num_encoder_patches, encoder_dim]
        bs = x.shape[0]

        # content embedding
        x = self.W_feat2emb(x)  # [B, N, D]

        # V용(content-only)
        x_clean = self.dropout(x)

        # 2) Query 준비
        if self.adaptive_initial_query:
            latent_queries = self.learnable_queries.unsqueeze(0).expand(bs, -1, -1)

            # K = content + pos (but implemented as W_K(content) + W_K(pos))
            k_init = self.W_K_init(x_clean)
            if self.use_positional_encoding and self.pos_embed is not None:
                if (not self.training) and (self._k_init_pos_cache is not None):
                    k_init = k_init + self._k_init_pos_cache.to(device=k_init.device, dtype=k_init.dtype)
                else:
                    # training or cache-miss fallback
                    pos = self.pos_embed.to(device=k_init.device, dtype=k_init.dtype)
                    k_init = k_init + self.W_K_init(pos)

            v_init = self.W_V_init(x_clean)

            latent_attn_scores = torch.bmm(latent_queries, k_init.transpose(1, 2))
            latent_attn_weights = F.softmax(latent_attn_scores, dim=-1)
            seq_decoder_patches = torch.bmm(latent_attn_weights, v_init)
        else:
            seq_decoder_patches = self.learnable_queries.unsqueeze(0).expand(bs, -1, -1)

        # API 호환을 위해 (K,V)를 둘 다 content-only 텐서로 반환합니다.
        # 실제 K의 positional 성분은 Decoder/Cross-Attn에서 projection-space로 더해집니다.
        return x_clean, x_clean, seq_decoder_patches




class Decoder(nn.Module):
    def __init__(
        self,
        num_encoder_patches,
        emb_dim,
        num_heads,
        num_decoder_patches,
        decoder_ff_dim=None,
        attn_dropout=0.0,
        dropout=0.0,
        drop_path_ratio=0.0,
        res_attention=False,
        num_decoder_layers=1,
        save_attention=False,
    ):
        super().__init__()

        # Stochastic Depth Decay Rule: 0부터 drop_path_ratio까지 선형적으로 증가
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, num_decoder_layers)]

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    num_encoder_patches,
                    emb_dim,
                    num_decoder_patches,
                    num_heads=num_heads,
                    decoder_ff_dim=decoder_ff_dim,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    drop_path=dpr[i],
                    res_attention=res_attention,
                    save_attention=save_attention,
                )
                for i in range(num_decoder_layers)
            ]
        )
        self.res_attention = res_attention

    @torch.no_grad()
    def prepare_inference_caches(self, pos_embed: Tensor):
        """각 DecoderLayer의 cross-attn에 대해 W_K(pos_embed) 캐시를 생성합니다."""
        for layer in self.layers:
            layer.prepare_inference_caches(pos_embed)

    def forward(self, seq_encoder_k: Tensor, seq_encoder_v: Tensor, seq_decoder: Tensor, pos_embed: Tensor = None):
        """seq_encoder_k/seq_encoder_v: 현재는 content-only가 들어오며,
        cross-attn 내부에서 pos_embed(또는 캐시된 k_pos)를 K projection에 더합니다.
        """
        scores = None
        if self.res_attention:
            for mod in self.layers:
                _, seq_decoder, scores = mod(seq_encoder_k, seq_encoder_v, seq_decoder, pos_embed=pos_embed, prev=scores)
            return seq_decoder
        else:
            for mod in self.layers:
                _, seq_decoder = mod(seq_encoder_k, seq_encoder_v, seq_decoder, pos_embed=pos_embed)
            return seq_decoder


class DecoderLayer(nn.Module):
    def __init__(
        self,
        num_encoder_patches,
        emb_dim,
        num_decoder_patches,
        num_heads,
        decoder_ff_dim=256,
        save_attention=False,
        attn_dropout=0,
        drop_path=0.0, # [추가] DropPath 비율
        dropout=0.0,
        bias=False,
        res_attention=False,
    ):
        super().__init__()
        assert not emb_dim % num_heads, f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})"

        self.res_attention = res_attention
        self.cross_attn = _MultiheadAttention(
            emb_dim,
            num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
            qkv_bias=False,
        )
        self.dropout_attn = nn.Dropout(dropout, inplace=True)
        # [수정] DropPath 적용
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_attn = nn.LayerNorm(emb_dim)

        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, decoder_ff_dim, bias=bias),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(decoder_ff_dim, emb_dim, bias=bias),
        )
        self.dropout_ffn = nn.Dropout(dropout, inplace=True)
        self.norm_ffn = nn.LayerNorm(emb_dim)

        self.save_attention = save_attention

    @torch.no_grad()
    def prepare_inference_caches(self, pos_embed: Tensor):
        """cross-attn에서 사용할 k_pos cache를 생성합니다."""
        self.cross_attn.prepare_k_pos_cache(pos_embed)

    def forward(self, seq_encoder_k: Tensor, seq_encoder_v: Tensor, seq_decoder: Tensor, pos_embed: Tensor = None, prev=None) -> Tensor:
        # 1) Cross-Attention (Pre-Norm)
        residual = seq_decoder
        seq_decoder = self.norm_attn(seq_decoder)

        if self.res_attention:
            decoder_out, attn, scores = self.cross_attn(seq_decoder, seq_encoder_k, seq_encoder_v, pos_embed=pos_embed, prev=prev)
        else:
            decoder_out, attn = self.cross_attn(seq_decoder, seq_encoder_k, seq_encoder_v, pos_embed=pos_embed)

        if self.save_attention:
            self.attn = attn

        # [수정] DropPath 적용 (추론 시에는 영향 없음)
        seq_decoder = residual + self.drop_path(self.dropout_attn(decoder_out))

        # 2) FFN (Pre-Norm)
        residual = seq_decoder
        seq_decoder = self.norm_ffn(seq_decoder)
        ffn_out = self.ffn(seq_decoder)
        # [수정] DropPath 적용
        seq_decoder = residual + self.drop_path(self.dropout_ffn(ffn_out))

        if self.res_attention:
            return seq_encoder_k, seq_decoder, scores
        else:
            return seq_encoder_k, seq_decoder


class _MultiheadAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        num_heads,
        res_attention=False,
        attn_dropout=0.0,
        proj_dropout=0.0,
        qkv_bias=False,
        save_attention=False,
        **kwargs,
    ):
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

        # [Method B] inference cache: W_K(pos_embed) projected & head-split
        # shape: [1, H, N, Dh]
        self.register_buffer('_k_pos_cache', None, persistent=False)

    @torch.no_grad()
    def prepare_k_pos_cache(self, pos_embed: Tensor):
        """W_K(pos_embed)를 미리 계산해 head-split 형태로 캐시합니다.

        - pos_embed: [1, N, D]
        - cache:     [1, H, N, Dh]

        주의: 반드시 weight 로드 후(model.load_state_dict 이후), eval 모드에서 호출 권장.
        """
        if pos_embed is None:
            self._k_pos_cache = None
            return

        dev = self.W_K.weight.device
        dt = self.W_K.weight.dtype
        pos = pos_embed.to(device=dev, dtype=dt)

        k_pos = self.W_K(pos)  # [1, N, H*Dh]
        k_pos = k_pos.view(1, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        self._k_pos_cache = k_pos

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, pos_embed: Tensor = None, prev=None):
        bs = Q.size(0)

        # Q Projection
        q_s = self.W_Q(Q).view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # K projection (content)
        k_s = self.W_K(K).view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # [Method B] add positional contribution in projection-space
        if pos_embed is not None:
            if (not self.training) and (self._k_pos_cache is not None):
                k_pos = self._k_pos_cache
                # 안전장치: 디바이스/타입이 다르면 캐스트
                if k_pos.device != k_s.device or k_pos.dtype != k_s.dtype:
                    k_pos = k_pos.to(device=k_s.device, dtype=k_s.dtype)
                k_s = k_s + k_pos
            else:
                # training or cache-miss fallback (will appear in ONNX if cache not prepared)
                pos = pos_embed.to(device=k_s.device, dtype=k_s.dtype)
                k_pos = self.W_K(pos).view(1, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                k_s = k_s + k_pos
        
        v_s = self.W_V(V).view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q_s, k_s.transpose(-1, -2)) * self.scale
        if prev is not None:
            attn_scores = attn_scores + prev

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v_s)
        output = output.permute(0, 2, 1, 3).reshape(bs, -1, self.num_heads * self.head_dim)
        output = self.concatheads2emb(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights

class Model(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        
        num_encoder_patches = args.num_encoder_patches 
        num_labels = args.num_labels 
        num_decoder_patches = args.num_decoder_patches 
        self.encoder_dim = args.encoder_dim 
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
        # [수정] Drop Path Ratio (config에서 설정 가능하도록 하거나 기본값 0.1 사용)
        drop_path_ratio = getattr(args, 'drop_path_ratio', 0.1)

        # 2D PE 생성을 위해 그리드 크기 전달 (main.py에서 계산된 값이 있으면 사용)
        grid_size_h = getattr(args, 'grid_size_h', None)
        grid_size_w = getattr(args, 'grid_size_w', None)
        if grid_size_h is None or grid_size_w is None:
            # fallback: 정사각형 그리드라고 가정
            grid_size = int(math.sqrt(num_encoder_patches))
            grid_size_h = grid_size
            grid_size_w = grid_size

        decoder_ff_dim = emb_dim * decoder_ff_ratio 

        self.embedding4decoder = Embedding4Decoder(num_encoder_patches=num_encoder_patches, encoder_dim=self.encoder_dim, num_decoder_patches=num_decoder_patches, 
                                grid_size_h=grid_size_h, grid_size_w=grid_size_w, # 추가된 인자
                                adaptive_initial_query=adaptive_initial_query,
                                num_decoder_layers=num_decoder_layers, emb_dim=emb_dim, num_heads=num_heads, decoder_ff_dim=decoder_ff_dim, positional_encoding=positional_encoding,
                                attn_dropout=attn_dropout, dropout=dropout, drop_path_ratio=drop_path_ratio,
                                res_attention=res_attention, save_attention=save_attention)
        

        

        # self.projection4classifier = Projection4Classifier(emb_dim, self.encoder_dim)



    def forward(self, x): 

        # x: [B, num_encoder_patches, encoder_dim]

        # (PatchConvEncoder의 출력이 여기로 들어옴)

        

        seq_encoder_patches, seq_encoder_clean, seq_decoder_patches = self.embedding4decoder(x)

        pos_embed = self.embedding4decoder.pos_embed if getattr(self.embedding4decoder, 'use_positional_encoding', False) else None

        z = self.embedding4decoder.decoder(seq_encoder_patches, seq_encoder_clean, seq_decoder_patches, pos_embed=pos_embed)

        # features = self.projection4classifier(z)

        return z
# =============================================================================
# 3. 전체 모델 구성
# =============================================================================
class Classifier(nn.Module):
    """디코더의 [B, Q, D_emb] 출력을 받아 최종 클래스 로짓으로 매핑합니다.
    모든 쿼리 토큰을 flatten하고 Linear 레이어를 통과시켜 클래스별 점수를 계산합니다.
    """
    def __init__(self, num_decoder_patches, emb_dim, num_labels, dropout):
        super().__init__()
        input_dim = num_decoder_patches * emb_dim
        hidden_dim = emb_dim * 2  # emb_dim 기반의 hidden dimension

        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),      # [B, Q, D_emb] -> [B, Q * D_emb]
            nn.LayerNorm(input_dim),      # LayerNorm 추가
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x):
        # x shape: [B, num_decoder_patches, emb_dim]
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
