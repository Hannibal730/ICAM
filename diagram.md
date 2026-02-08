graph TD
    classDef tensor fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef process fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef param fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    subgraph Input_Stage ["1. Input Stage"]
        IMG["Input Image<br/>(Batch, 3, 224, 224)"]:::tensor
    end

    subgraph Encoder_Stage ["2. Encoder (PatchingEncoder)"]
        direction TB
        subgraph Custom24 ["CNN Feature Extractor (custom24)"]
            direction TB
            C24_STEM["Stem<br/>Conv3x3 s2 (3->32)<br/>BN + ReLU6<br/>224->112"]:::process
            C24_MB1["MBConv1<br/>3x3 s1 (32->16)<br/>expand=1<br/>112->112"]:::process
            C24_MB2["MBConv6<br/>3x3 s2 (16->24)<br/>expand=6<br/>112->56"]:::process
            C24_MB3["MBConv6<br/>3x3 s1 (24->24)<br/>expand=6<br/>56->56"]:::process
        end

        FEAT_MAP["Feature Map (custom24 out)<br/>(B, 24, 56, 56)"]:::tensor
        GRID_POOL["Adaptive Avg Pool<br/>Target: 7x7"]:::process
        FLATTEN["Flatten & Permute"]:::process
        LN_ENC["LayerNorm (Dim: 24)"]:::process
        ENC_OUT["Encoder Tokens<br/>(B, 49, 24)"]:::tensor
    end

    subgraph Decoder_Init_Stage ["3. Decoder Initialization (Embedding4Decoder)"]
        direction TB
        LQ["Learnable Queries<br/>Param: (1, 24)"]:::param
        POS_EMB["2D SinCos Pos Embed<br/>(1, 49, 24)"]:::param

        subgraph Adaptive_Logic ["Adaptive Initial Query: True"]
            INIT_Q["Q_init from Learnable Queries"]:::process
            INIT_K["K_init from Encoder Tokens"]:::process
            INIT_ADD_POS(("Add PE to K_init")):::process
            INIT_V["V_init from Encoder Tokens"]:::process
            INIT_ATTN["Initialization Attention<br/>(Q, K+PE, V)"]:::process
        end

        DEC_Q["Initial Decoder Query<br/>(B, 1, 24)"]:::tensor
    end

    subgraph PE_Generation ["PE Generation (get_2d_sincos_pos_embed)"]
        direction TB
        PE_INPUT["Inputs<br/>embed_dim=24, grid_h=7, grid_w=7"]:::param
        PE_ARANGE["arange(7), arange(7)"]:::process
        PE_MESH["meshgrid(indexing='xy')"]:::process
        PE_H_COORD["grid_h_coords<br/>(7, 7)"]:::tensor
        PE_W_COORD["grid_w_coords<br/>(7, 7)"]:::tensor

        subgraph PE_H_1D ["1D SinCos for H (D/2=12)"]
            direction TB
            PE_H_POS["pos_h = grid_h_coords.flatten()<br/>(N=49)"]:::tensor
            PE_H_OMEGA["omega_h = 1 / 10000^(arange(6)/6)"]:::process
            PE_H_OUT["out_h = pos_h * omega_h<br/>(49, 6)"]:::process
            PE_H_SIN["sin(out_h)"]:::process
            PE_H_COS["cos(out_h)"]:::process
            PE_H_CAT["emb_h = [sin, cos]<br/>(49, 12)"]:::tensor
        end

        subgraph PE_W_1D ["1D SinCos for W (D/2=12)"]
            direction TB
            PE_W_POS["pos_w = grid_w_coords.flatten()<br/>(N=49)"]:::tensor
            PE_W_OMEGA["omega_w = 1 / 10000^(arange(6)/6)"]:::process
            PE_W_OUT["out_w = pos_w * omega_w<br/>(49, 6)"]:::process
            PE_W_SIN["sin(out_w)"]:::process
            PE_W_COS["cos(out_w)"]:::process
            PE_W_CAT["emb_w = [sin, cos]<br/>(49, 12)"]:::tensor
        end

        PE_CAT["concat(emb_h, emb_w)<br/>pos_embed (49, 24)"]:::tensor
        PE_UNSQUEEZE["unsqueeze(0)<br/>(1, 49, 24)"]:::tensor
    end

    subgraph Decoder_Stage ["4. Decoder Backbone"]
        direction TB

        subgraph Decoder_Layer ["Decoder Layer (Repeated x6, heads=2, D=24)"]
            direction TB
            DL_IN["Layer Input"]:::tensor

            LN1["LayerNorm"]:::process
            subgraph MHCA ["Multi-Head Cross-Attention"]
                Q_Proj["Q Projection"]:::process
                K_Proj["K Projection"]:::process
                ADD_POS(("Add PE to K")):::process
                V_Proj["V Projection"]:::process
                ATTN["Scaled Dot-Product<br/>Attention<br/>(Q, K+PE, V)"]:::process
                OUT_Proj["Output Projection"]:::process
            end
            ADD1(("Add")):::process

            LN2["LayerNorm"]:::process
            FFN["FFN<br/>Linear -> ReLU6 -> Dropout -> Linear"]:::process
            ADD2(("Add")):::process

            DL_OUT["Layer Output"]:::tensor
        end

        DL_OUT -. "Repeat 6 times" .-> DL_IN
    end

    subgraph Classifier_Stage ["5. Classifier"]
        PROJ["Projection4Classifier<br/>Mean Pool -> Linear(24->24)"]:::process
        MLP["MLP Head<br/>Linear(24->13) -> ReLU6 -> Dropout -> Linear(13->2)"]:::process
        LOGITS["Output Logits<br/>(B, 2)"]:::tensor
    end

    %% Main connections
    IMG --> C24_STEM --> C24_MB1 --> C24_MB2 --> C24_MB3 --> FEAT_MAP
    FEAT_MAP --> GRID_POOL --> FLATTEN --> LN_ENC --> ENC_OUT

    %% Init attention: PE is added to K only
    LQ --> INIT_Q --> INIT_ATTN
    ENC_OUT -.-> INIT_K --> INIT_ADD_POS --> INIT_ATTN
    POS_EMB -. "Add to K only" .-> INIT_ADD_POS
    ENC_OUT -.-> INIT_V --> INIT_ATTN
    INIT_ATTN --> DEC_Q

    %% Decoder cross-attention: PE is added to K only
    DEC_Q --> DL_IN
    DL_IN --> LN1 --> Q_Proj
    DL_IN --> ADD1
    ENC_OUT -.-> K_Proj
    ENC_OUT -.-> V_Proj
    K_Proj --> ADD_POS
    POS_EMB -. "Add to K only" .-> ADD_POS
    ADD_POS --> ATTN
    V_Proj --> ATTN
    Q_Proj --> ATTN
    ATTN --> OUT_Proj --> ADD1

    ADD1 --> LN2 --> FFN --> ADD2
    ADD1 --> ADD2
    ADD2 --> DL_OUT
    DL_OUT --> PROJ --> MLP --> LOGITS

    %% PE generation flow
    PE_INPUT --> PE_ARANGE --> PE_MESH
    PE_MESH --> PE_H_COORD
    PE_MESH --> PE_W_COORD
    PE_H_COORD --> PE_H_POS --> PE_H_OUT
    PE_H_OMEGA --> PE_H_OUT
    PE_H_OUT --> PE_H_SIN --> PE_H_CAT
    PE_H_OUT --> PE_H_COS --> PE_H_CAT
    PE_W_COORD --> PE_W_POS --> PE_W_OUT
    PE_W_OMEGA --> PE_W_OUT
    PE_W_OUT --> PE_W_SIN --> PE_W_CAT
    PE_W_OUT --> PE_W_COS --> PE_W_CAT
    PE_H_CAT --> PE_CAT
    PE_W_CAT --> PE_CAT
    PE_CAT --> PE_UNSQUEEZE --> POS_EMB
