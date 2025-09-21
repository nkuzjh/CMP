# TTA records



# model structure
model: {'', 'itm_head', 'pose_block', 'text_proj', 'vision_proj', 'pose_conv', 'avgpool', 'vision_encoder', 'text_encoder'}
## Search(
  ### (vision_encoder): SwinTransformer(
    (patch_embed): PatchEmbed(
      (proj): Conv2d(3, 128, kernel_size=(4, 4), stride=(4, 4))
      (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    )
    (pos_drop): Dropout(p=0, inplace=False)
    (layers): ModuleList(
      (0): BasicLayer(
        dim=128, input_resolution=(56, 56), depth=2
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=128, input_resolution=(56, 56), num_heads=4, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=128, window_size=(7, 7), num_heads=4
              (qkv): Linear(in_features=128, out_features=384, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=128, out_features=128, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=512, out_features=128, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=128, input_resolution=(56, 56), num_heads=4, window_size=7, shift_size=3, mlp_ratio=4.0
            (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=128, window_size=(7, 7), num_heads=4
              (qkv): Linear(in_features=128, out_features=384, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=128, out_features=128, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.004)
            (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=512, out_features=128, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          input_resolution=(56, 56), dim=128
          (reduction): Linear(in_features=512, out_features=256, bias=False)
          (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
      )
      (1): BasicLayer(
        dim=256, input_resolution=(28, 28), depth=2
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=256, input_resolution=(28, 28), num_heads=8, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=256, window_size=(7, 7), num_heads=8
              (qkv): Linear(in_features=256, out_features=768, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=256, out_features=256, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.009)
            (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=256, out_features=1024, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1024, out_features=256, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=256, input_resolution=(28, 28), num_heads=8, window_size=7, shift_size=3, mlp_ratio=4.0
            (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=256, window_size=(7, 7), num_heads=8
              (qkv): Linear(in_features=256, out_features=768, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=256, out_features=256, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.013)
            (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=256, out_features=1024, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1024, out_features=256, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          input_resolution=(28, 28), dim=256
          (reduction): Linear(in_features=1024, out_features=512, bias=False)
          (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
      )
      (2): BasicLayer(
        dim=512, input_resolution=(14, 14), depth=18
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(7, 7), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.017)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=3, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(7, 7), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.022)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (2): SwinTransformerBlock(
            dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(7, 7), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.026)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (3): SwinTransformerBlock(
            dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=3, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(7, 7), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.030)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (4): SwinTransformerBlock(
            dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(7, 7), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.035)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (5): SwinTransformerBlock(
            dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=3, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(7, 7), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.039)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (6): SwinTransformerBlock(
            dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(7, 7), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.043)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (7): SwinTransformerBlock(
            dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=3, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(7, 7), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.048)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (8): SwinTransformerBlock(
            dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(7, 7), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.052)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (9): SwinTransformerBlock(
            dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=3, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(7, 7), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.057)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (10): SwinTransformerBlock(
            dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(7, 7), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.061)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (11): SwinTransformerBlock(
            dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=3, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(7, 7), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.065)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (12): SwinTransformerBlock(
            dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(7, 7), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.070)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (13): SwinTransformerBlock(
            dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=3, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(7, 7), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.074)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (14): SwinTransformerBlock(
            dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(7, 7), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.078)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (15): SwinTransformerBlock(
            dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=3, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(7, 7), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.083)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (16): SwinTransformerBlock(
            dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(7, 7), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.087)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (17): SwinTransformerBlock(
            dim=512, input_resolution=(14, 14), num_heads=16, window_size=7, shift_size=3, mlp_ratio=4.0
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=512, window_size=(7, 7), num_heads=16
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.091)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          input_resolution=(14, 14), dim=512
          (reduction): Linear(in_features=2048, out_features=1024, bias=False)
          (norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        )
      )
      (3): BasicLayer(
        dim=1024, input_resolution=(7, 7), depth=2
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=1024, input_resolution=(7, 7), num_heads=32, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=1024, window_size=(7, 7), num_heads=32
              (qkv): Linear(in_features=1024, out_features=3072, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=1024, out_features=1024, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.096)
            (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=1024, input_resolution=(7, 7), num_heads=32, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=1024, window_size=(7, 7), num_heads=32
              (qkv): Linear(in_features=1024, out_features=3072, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=1024, out_features=1024, bias=True)
              (proj_drop): Dropout(p=0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.100)
            (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
              (drop): Dropout(p=0, inplace=False)
            )
          )
        )
      )
    )
    (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    (avgpool): AdaptiveAvgPool1d(output_size=1)
  )
  ### (text_encoder): BertForMaskedLM(
    (bert): BertModel(
      (embeddings): BertEmbeddings(
        (word_embeddings): Embedding(30522, 768, padding_idx=0)
        (position_embeddings): Embedding(512, 768)
        (token_type_embeddings): Embedding(2, 768)
        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (encoder): BertEncoder(
        (layer): ModuleList(
          (0-5): 6 x BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (6-11): 6 x BertLayer(
            (attention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (crossattention): BertAttention(
              (self): BertSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=1024, out_features=768, bias=True)
                (value): Linear(in_features=1024, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
      )
    )
    (cls): BertOnlyMLMHead(
      (predictions): BertLMPredictionHead(
        (transform): BertPredictionHeadTransform(
          (dense): Linear(in_features=768, out_features=768, bias=True)
          (transform_act_fn): GELUActivation()
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        )
        (decoder): Linear(in_features=768, out_features=30522, bias=True)
      )
    )
  )
  (avgpool): AdaptiveAvgPool1d(output_size=3)
  ### (vision_proj): Sequential(
    (0): BatchNorm1d(3072, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): Dropout(p=0.5, inplace=False)
    (2): Linear(in_features=3072, out_features=2048, bias=True)
  )
  ### (text_proj): Sequential(
    (0): BatchNorm1d(3072, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): Dropout(p=0.5, inplace=False)
    (2): Linear(in_features=3072, out_features=2048, bias=True)
  )
  ### (itm_head): Sequential(
    (0): Linear(in_features=768, out_features=1536, bias=True)
    (1): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
    (2): GELU(approximate='none')
    (3): Linear(in_features=1536, out_features=2, bias=True)
  )
  ### (pose_block): Block(
    (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    (attn): Attention(
      (q): Linear(in_features=1024, out_features=1024, bias=True)
      (k): Linear(in_features=1024, out_features=1024, bias=True)
      (v): Linear(in_features=1024, out_features=1024, bias=True)
      (attn_drop): Dropout(p=0.0, inplace=False)
      (proj): Linear(in_features=1024, out_features=1024, bias=True)
      (proj_drop): Dropout(p=0.0, inplace=False)
    )
  )
  ### (pose_conv): ConvExpandReduce(
    (expand): Sequential(
      (0): Conv2d(3, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (reduce): Sequential(
      (0): Conv2d(96, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
  )
)



# baseline score
+-------+--------+--------+--------+--------+--------+
| epoch |   R1   |   R5   |  R10   |  mAP   |  mINP  |
+-------+--------+--------+--------+--------+--------+
|  ITC  | 69.414 | 95.197 | 97.776 | 81.233 | 81.233 |
|  ITM  | 84.277 | 99.039 | 99.596 | 91.276 | 91.276 |
| Paper | 84.93  | 99.09  | 99.75  | 91.66  |   --   |
+-------+--------+--------+--------+--------+--------+


# command

1. train
    python3 run.py --task "cmp" --dist "f4" --output_dir "output/cmp"
2. evaluate
    python3 run.py --task "cmp" --evaluate --dist "f4" --output_dir "output/cmp_eval" --checkpoint "checkpoint/cmp.pth"
    python3 run.py --task "cmp" --evaluate --dist "f2" --output_dir "output/cmp_eval" --checkpoint "checkpoint/cmp.pth"
    python3 run.py --task "cmp" --evaluate --dist "gpu0" --output_dir "output/cmp_eval" --checkpoint "checkpoint/cmp.pth"

3. tta: run.py
    python3 run.py --task "tta" --tta
4. tta: tta.py
    CUDA_VISIBLE_DEVICES=1 python3 tta.py --tta --task tta --config configs/tta.yaml  --output_dir output/tta/2025081715503  --checkpoint checkpoint/cmp.pth --bs 32 --epo 10 --lr 0.0001 --seed 42


# exp

## tta_debug
    python3 run.py --task "tta_debug" --tta --checkpoint "checkpoint/cmp.pth" --bs 1 --epo 10 --lr 1e-4 --seed 42

    CUDA_VISIBLE_DEVICES=1 python3 tta.py --config configs/tta_debug.yaml --task tta_debug --output_dir output/tta_debug/2025081715503 --checkpoint checkpoint/cmp.pth --bs 3 --epo 10 --lr 0.0001 --seed 42 --tta

    CUDA_VISIBLE_DEVICES=1 python3 tta.py --config configs/tta_exp1.yaml --task tta_debug --output_dir output/tta_debug/exp1 --checkpoint checkpoint/cmp.pth --bs 3 --epo 10 --lr 0.001 --seed 42 --tta

    CUDA_VISIBLE_DEVICES=1 python3 tta.py --config configs/tta_exp2.yaml --task tta_debug --output_dir output/tta_debug/exp2 --checkpoint checkpoint/cmp.pth --bs 3 --epo 10 --lr 0.001 --seed 42 --tta

    CUDA_VISIBLE_DEVICES=2 python3 tta.py --config configs/tta_exp3.yaml --task tta_debug --output_dir output/tta_debug/exp3 --checkpoint checkpoint/cmp.pth --bs 3 --epo 10 --lr 0.001 --seed 42 --tta

## exp0
**entropy**
    nohup python3 run.py --tta --task "tta_exp0"> logs/tta_exp0.log 2>&1 &
        {"epo": "1", "R1": "84.681", "R5": "98.584", "R10": "99.292", "mAP": "91.252", "mINP": "91.252", "lr": "0.000876", "entropy": "0.189884", "loss": "0.189884"}
- exp0.1~4
    nohup python3 run.py --tta --task "tta_exp0.1"> logs/tta_exp0.1.log 2>&1 &
        {"epo": "0", "R1": "84.681", "R5": "98.989", "R10": "99.596", "mAP": "91.451", "mINP": "91.451", "lr": "3.1e-05", "entropy": "0.35984", "loss": "0.35984"}
    nohup python3 run.py --tta --task "tta_exp0.2"> logs/tta_exp0.2.log 2>&1 &
        {"epo": "0", "R1": "84.631", "R5": "98.837", "R10": "99.494", "mAP": "91.341", "mINP": "91.341", "lr": "0.000155", "entropy": "0.335118", "loss": "0.335118"}
    nohup python3 run.py --tta --task "tta_exp0.3"> logs/tta_exp0.3.log 2>&1 &
        {"epo": "1", "R1": "85.086", "R5": "98.787", "R10": "99.393", "mAP": "91.578", "mINP": "91.578", "lr": "0.000488", "entropy": "0.184185", "loss": "0.184185"}
    nohup python3 run.py --tta --task "tta_exp0.4"> logs/tta_exp0.4.log 2>&1 &
        {"epo": "3", "R1": "85.187", "R5": "98.787", "R10": "99.444", "mAP": "91.577", "mINP": "91.577", "lr": "0.000567", "entropy": "0.043413", "loss": "0.043413"}

## exp1
**entropy + ss**
    nohup python3 run.py --tta --task "tta_exp1"> logs/tta_exp1.log 2>&1 &
        {"epo": "3", "R1": "84.833", "R5": "98.787", "R10": "99.343", "mAP": "91.35", "mINP": "91.35", "lr": "0.000924", "entropy": "0.043405", "loss": "0.043405"}
- exp1.1-5
    1.1 {"epo": "4", "R1": "84.783", "R5": "98.787", "R10": "99.343", "mAP": "91.405", "mINP": "91.405", "lr": "0.000445", "entropy": "0.059843", "loss": "0.059843"}
    1.2 {"epo": "0", "R1": "84.783", "R5": "98.989", "R10": "99.596", "mAP": "91.505", "mINP": "91.505", "lr": "4e-05", "entropy": "0.331566", "loss": "0.331566"}
    1.3 {"epo": "0", "R1": "84.681", "R5": "98.888", "R10": "99.494", "mAP": "91.378", "mINP": "91.378", "lr": "0.0002", "entropy": "0.289342", "loss": "0.289342"}
    1.4 {"epo": "0", "R1": "84.783", "R5": "98.888", "R10": "99.545", "mAP": "91.477", "mINP": "91.477", "lr": "0.0001", "entropy": "0.303752", "loss": "0.303752"}
    1.5 {"epo": "1", "R1": "84.732", "R5": "98.888", "R10": "99.545", "mAP": "91.463", "mINP": "91.463", "lr": "5.9e-05", "entropy": "0.293154", "loss": "0.293154"}

## exp2
**entropy + ss + unc**
    nohup python3 run.py --tta --task "tta_exp2"> logs/tta_exp2.log 2>&1 &
        {"epo": "3", "R1": "84.783", "R5": "98.736", "R10": "99.343", "mAP": "91.308", "mINP": "91.308", "lr": "0.000924", "entropy": "0.043163", "loss": "2.704784"}
- exp2.1-5
    2.1 {"epo": "3", "R1": "84.884", "R5": "98.736", "R10": "99.343", "mAP": "91.382", "mINP": "91.382", "lr": "0.000924", "entropy": "0.044278", "loss": "2.693614"}
    2.2 {"epo": "3", "R1": "84.681", "R5": "98.736", "R10": "99.292", "mAP": "91.24", "mINP": "91.24", "lr": "0.000924", "entropy": "0.042831", "loss": "2.626209"}
    2.3 {"epo": "4", "R1": "84.681", "R5": "98.736", "R10": "99.444", "mAP": "91.305", "mINP": "91.305", "lr": "0.000889", "entropy": "0.023986", "loss": "2.319043"}
    2.4 {"epo": "4", "R1": "84.732", "R5": "98.686", "R10": "99.444", "mAP": "91.321", "mINP": "91.321", "lr": "0.000889", "entropy": "0.024042", "loss": "1.563877"}
    2.5 {"epo": "4", "R1": "84.783", "R5": "98.837", "R10": "99.444", "mAP": "91.356", "mINP": "91.356", "lr": "0.000889", "entropy": "0.024167", "loss": "1.161326"}

# exp rerun
- 重新检查cmp_xvlm的cross_modal模型结构设计，并设置了仅更新text_encoder的后6层bertlayer
- 重新检查Tent系列方法的setting设置，设置了tta时model.train()和batchnorm,layernorm参数更新方案
- 目前方案为：
    1. dropout： 所有dropout通过model.train()打开（包括visison_encoder、text_encoder前6层、其他modules 和 需要梯度更新的实现了itm.cross_modal功能的text_encoder后六层）；
    2. requires_grad： text_encoder的后六层 和 itm_head的norm layer（无论batchnorm和layernorm）打开偏置params（γ、β）的梯度更新，但关闭track_running_stats并且train和eval都使用单个batch的stats（running_mean和running_var置为None）。

## tta_debug
    CUDA_VISIBLE_DEVICES=2 python3 tta.py --config configs/tta_exp4.yaml --task tta_debug --output_dir output/tta_debug/exp4 --checkpoint checkpoint/cmp.pth --bs 3 --epo 10 --lr 0.001 --seed 42 --tta

## exp0
**entropy**
    nohup python3 run.py --tta --task "tta_exp0"> logs/tta_exp0.log 2>&1 &
        {"epo": "2", "R1": "84.681", "R5": "98.787", "R10": "99.393", "mAP": "91.301", "mINP": "91.301", "lr": "0.00097", "entropy": "0.128151", "loss": "0.128151"}
- exp0.1~4
    nohup python3 run.py --tta --task "tta_exp0.1"> logs/tta_exp0.1.log 2>&1 &
        {"epo": "0", "R1": "84.681", "R5": "99.039", "R10": "99.596", "mAP": "91.46", "mINP": "91.46", "lr": "3.1e-05", "entropy": "0.34599", "loss": "0.34599"}
    nohup python3 run.py --tta --task "tta_exp0.2"> logs/tta_exp0.2.log 2>&1 &
        {"epo": "16", "R1": "85.035", "R5": "98.736", "R10": "99.343", "mAP": "91.467", "mINP": "91.467", "lr": "4.8e-05", "entropy": "0.062567", "loss": "0.062567"}
    nohup python3 run.py --tta --task "tta_exp0.3"> logs/tta_exp0.3.log 2>&1 &
        {"epo": "8", "R1": "85.086", "R5": "98.736", "R10": "99.494", "mAP": "91.578", "mINP": "91.578", "lr": "7.4e-05", "entropy": "0.081542", "loss": "0.081542"}
    nohup python3 run.py --tta --task "tta_exp0.4"> logs/tta_exp0.4.log 2>&1 &
        {"epo": "4", "R1": "85.086", "R5": "98.686", "R10": "99.393", "mAP": "91.529", "mINP": "91.529", "lr": "0.000437", "entropy": "0.067716", "loss": "0.067716"}

## exp1
**entropy + ss**
    nohup python3 run.py --tta --task "tta_exp1"> logs/tta_exp1.log 2>&1 &
        {"epo": "2", "R1": "85.592", "R5": "98.989", "R10": "99.444", "mAP": "91.939", "mINP": "91.939", "lr": "0.000958", "entropy": "0.089662", "loss": "0.089662"}
- exp1.1-5
    1.1
        {"epo": "4", "R1": "85.086", "R5": "98.686", "R10": "99.343", "mAP": "91.566", "mINP": "91.566", "lr": "0.000445", "entropy": "0.076353", "loss": "0.076353"}
    1.2
        {"epo": "7", "R1": "84.732", "R5": "98.635", "R10": "99.191", "mAP": "91.341", "mINP": "91.341", "lr": "3.9e-05", "entropy": "0.152249", "loss": "0.152249"}
    1.3
        {"epo": "2", "R1": "85.035", "R5": "98.736", "R10": "99.393", "mAP": "91.554", "mINP": "91.554", "lr": "0.000937", "entropy": "0.123062", "loss": "0.123062"}
    1.4
        {"epo": "6", "R1": "85.137", "R5": "98.635", "R10": "99.292", "mAP": "91.557", "mINP": "91.557", "lr": "0.00043", "entropy": "0.076485", "loss": "0.076485"}
    1.5
        {"epo": "1", "R1": "84.783", "R5": "98.989", "R10": "99.596", "mAP": "91.492", "mINP": "91.492", "lr": "5.9e-05", "entropy": "0.277572", "loss": "0.277572"}

## exp2
**entropy + ss + unc**
    nohup python3 run.py --tta --task "tta_exp2"> logs/tta_exp2.log 2>&1 &
        {"epo": "2", "R1": "85.49", "R5": "98.989", "R10": "99.444", "mAP": "91.883", "mINP": "91.883", "lr": "0.000958", "entropy": "0.089834", "loss": "2.722146"}
- exp2.1-5
    2.1
        {"epo": "2", "R1": "85.541", "R5": "98.989", "R10": "99.444", "mAP": "91.878", "mINP": "91.878", "lr": "0.000958", "entropy": "0.089567", "loss": "2.710543"}
    2.2
        {"epo": "2", "R1": "85.642", "R5": "99.039", "R10": "99.444", "mAP": "91.961", "mINP": "91.961", "lr": "0.000958", "entropy": "0.089432", "loss": "2.644134"}
    2.3
        {"epo": "2", "R1": "85.592", "R5": "99.039", "R10": "99.444", "mAP": "91.931", "mINP": "91.931", "lr": "0.000958", "entropy": "0.089636", "loss": "2.347391"}
    2.4
        {"epo": "2", "R1": "85.49", "R5": "98.989", "R10": "99.444", "mAP": "91.857", "mINP": "91.857", "lr": "0.000958", "entropy": "0.09095", "loss": "1.608024"}
    2.5
        {"epo": "2", "R1": "85.592", "R5": "99.039", "R10": "99.444", "mAP": "91.946", "mINP": "91.946", "lr": "0.000958", "entropy": "0.089624", "loss": "1.21968"}

# exp rerun 2
- 修复了uncertainty维度与entropy不一致的问题，该问题会导致loss=entropy/uncertainty成为一个(bs,bs)的tensor

- 重新检查cmp_xvlm的cross_modal模型结构设计，并设置了仅更新text_encoder的后6层bertlayer
- 重新检查Tent系列方法的setting设置，设置了tta时model.train()和batchnorm,layernorm参数更新方案
- 目前方案为：
    1. dropout： 所有dropout通过model.train()打开（包括visison_encoder、text_encoder前6层、其他modules 和 需要梯度更新的实现了itm.cross_modal功能的text_encoder后六层）
    2. requires_grad： text_encoder的后六层 和 itm_head的norm layer（无论batchnorm和layernorm）打开偏置params（γ、β）的梯度更新，但关闭track_running_stats并且train和eval都使用单个batch的stats（running_mean和running_var置为None）

## tta_debug
    CUDA_VISIBLE_DEVICES=1 python3 tta.py --config configs/tta_exp2.2.yaml --task tta_debug --output_dir output/tta_debug/exp2.2 --checkpoint checkpoint/cmp.pth --tta --bs 3 --epo 10 --lr 0.001 --seed 42

## exp0
**entropy**
    nohup python3 run.py --tta --task "tta_exp0"> logs/tta_exp0.log 2>&1 &
        {"epo": "2", "R1": "84.681", "R5": "98.787", "R10": "99.393", "mAP": "91.301", "mINP": "91.301", "lr": "0.00097", "entropy": "0.128151", "loss": "0.128151"}
- exp0.1~4
    nohup python3 run.py --tta --task "tta_exp0.1"> logs/tta_exp0.1.log 2>&1 &
        {"epo": "0", "R1": "84.681", "R5": "99.039", "R10": "99.596", "mAP": "91.46", "mINP": "91.46", "lr": "3.1e-05", "entropy": "0.34599", "loss": "0.34599"}
    nohup python3 run.py --tta --task "tta_exp0.2"> logs/tta_exp0.2.log 2>&1 &
        {"epo": "16", "R1": "85.035", "R5": "98.736", "R10": "99.343", "mAP": "91.467", "mINP": "91.467", "lr": "4.8e-05", "entropy": "0.062567", "loss": "0.062567"}
    nohup python3 run.py --tta --task "tta_exp0.3"> logs/tta_exp0.3.log 2>&1 &
        {"epo": "8", "R1": "85.086", "R5": "98.736", "R10": "99.494", "mAP": "91.578", "mINP": "91.578", "lr": "7.4e-05", "entropy": "0.081542", "loss": "0.081542"}
    nohup python3 run.py --tta --task "tta_exp0.4"> logs/tta_exp0.4.log 2>&1 &
        {"epo": "4", "R1": "85.086", "R5": "98.686", "R10": "99.393", "mAP": "91.529", "mINP": "91.529", "lr": "0.000437", "entropy": "0.067716", "loss": "0.067716"}

## exp1
**entropy + ss**
    nohup python3 run.py --tta --task "tta_exp1"> logs/tta_exp1.log 2>&1 &
        {"epo": "2", "R1": "85.592", "R5": "98.989", "R10": "99.444", "mAP": "91.939", "mINP": "91.939", "lr": "0.000958", "entropy": "0.089662", "loss": "0.089662"}
- exp1.1-5
    1.1
        {"epo": "4", "R1": "85.086", "R5": "98.686", "R10": "99.343", "mAP": "91.566", "mINP": "91.566", "lr": "0.000445", "entropy": "0.076353", "loss": "0.076353"}
    1.2
        {"epo": "7", "R1": "84.732", "R5": "98.635", "R10": "99.191", "mAP": "91.341", "mINP": "91.341", "lr": "3.9e-05", "entropy": "0.152249", "loss": "0.152249"}
    1.3
        {"epo": "2", "R1": "85.035", "R5": "98.736", "R10": "99.393", "mAP": "91.554", "mINP": "91.554", "lr": "0.000937", "entropy": "0.123062", "loss": "0.123062"}
    1.4
        {"epo": "6", "R1": "85.137", "R5": "98.635", "R10": "99.292", "mAP": "91.557", "mINP": "91.557", "lr": "0.00043", "entropy": "0.076485", "loss": "0.076485"}
    1.5
        {"epo": "1", "R1": "84.783", "R5": "98.989", "R10": "99.596", "mAP": "91.492", "mINP": "91.492", "lr": "5.9e-05", "entropy": "0.277572", "loss": "0.277572"}

## exp2
**entropy + ss + unc**
    nohup python3 run.py --tta --task "tta_exp2"> logs/tta_exp2.log 2>&1 &
        {"epo": "2", "R1": "85.592", "R5": "98.989", "R10": "99.444", "mAP": "91.912", "mINP": "91.912", "lr": "0.000958", "entropy": "0.089008", "loss": "2.721839"}
- exp2.1-5
    2.1
        {"epo": "2", "R1": "85.642", "R5": "99.039", "R10": "99.444", "mAP": "91.972", "mINP": "91.972", "lr": "0.000958", "entropy": "0.089416", "loss": "2.710488"}
    2.2
        {"epo": "2", "R1": "85.693", "R5": "98.989", "R10": "99.444", "mAP": "91.958", "mINP": "91.958", "lr": "0.000958", "entropy": "0.088998", "loss": "2.643966"}
    2.3
        {"epo": "2", "R1": "85.44", "R5": "98.989", "R10": "99.444", "mAP": "91.837", "mINP": "91.837", "lr": "0.000958", "entropy": "0.08838", "loss": "2.346493"}
    2.4
        {"epo": "4", "R1": "85.086", "R5": "98.787", "R10": "99.444", "mAP": "91.537", "mINP": "91.537", "lr": "0.000889", "entropy": "0.046609", "loss": "1.575436"}
    2.5
        {"epo": "2", "R1": "85.541", "R5": "98.989", "R10": "99.444", "mAP": "91.878", "mINP": "91.878", "lr": "0.000958", "entropy": "0.089748", "loss": "1.213151"}

## exp3
**entropy + unc**
    nohup python3 run.py --tta --task "tta_exp3"> logs/tta_exp3.log 2>&1 &
        {"epo": "26", "R1": "85.541", "R5": "98.534", "R10": "99.444", "mAP": "91.756", "mINP": "91.756", "lr": "2.5e-05", "entropy": "0.035251", "loss": "2.702479"}
- exp3.1-5
    3.1
        {"epo": "3", "R1": "85.137", "R5": "98.686", "R10": "99.393", "mAP": "91.604", "mINP": "91.604", "lr": "0.000567", "entropy": "0.075018", "loss": "2.706798"}
    3.2
        {"epo": "3", "R1": "85.086", "R5": "98.686", "R10": "99.393", "mAP": "91.58", "mINP": "91.58", "lr": "0.000567", "entropy": "0.0757", "loss": "2.648713"}
    3.3
        {"epo": "1", "R1": "84.631", "R5": "98.837", "R10": "99.343", "mAP": "91.365", "mINP": "91.365", "lr": "0.000977", "entropy": "0.14208", "loss": "2.427362"}
    3.4
        {"epo": "21", "R1": "84.681", "R5": "98.736", "R10": "99.393", "mAP": "91.393", "mINP": "91.393", "lr": "6e-06", "entropy": "0.177065", "loss": "2.755205"}
    3.5
        {"epo": "7", "R1": "85.035", "R5": "98.787", "R10": "99.494", "mAP": "91.555", "mINP": "91.555", "lr": "7.7e-05", "entropy": "0.084124", "loss": "2.72065"}
    3.6
        {"epo": "2", "R1": "84.732", "R5": "98.736", "R10": "99.393", "mAP": "91.317", "mINP": "91.317", "lr": "0.00097", "entropy": "0.128045", "loss": "2.736988"}
    3.7
        {"epo": "0", "R1": "84.681", "R5": "99.039", "R10": "99.596", "mAP": "91.459", "mINP": "91.459", "lr": "3.1e-05", "entropy": "0.34605", "loss": "2.818058"}
    3.8
        {"epo": "8", "R1": "84.985", "R5": "98.736", "R10": "99.292", "mAP": "91.447", "mINP": "91.447", "lr": "0.00019", "entropy": "0.083007", "loss": "2.720241"}

## exp4
**entropy + unc_temper_learn**
    {"epo": "2", "R1": "84.226", "R5": "98.483", "R10": "99.343", "mAP": "91.066", "mINP": "91.066", "lr": "0.000943", "entropy": "0.094519", "loss": "2.152995"}
- exp4.1-5
    4.1
        {"epo": "3", "R1": "85.086", "R5": "98.686", "R10": "99.444", "mAP": "91.55", "mINP": "91.55", "lr": "0.000567", "entropy": "0.071233", "loss": "5.156881"}
    4.2
        {"epo": "15", "R1": "84.732", "R5": "98.736", "R10": "99.393", "mAP": "91.357", "mINP": "91.357", "lr": "0.0001", "entropy": "0.040845", "loss": "74.359854"}
    4.3
        {"epo": "25", "R1": "84.58", "R5": "98.635", "R10": "99.393", "mAP": "91.271", "mINP": "91.271", "lr": "3.2e-05", "entropy": "0.036328", "loss": "0.573821"}
    4.4
        {"epo": "22", "R1": "84.681", "R5": "98.736", "R10": "99.393", "mAP": "91.393", "mINP": "91.393", "lr": "5e-06", "entropy": "0.174912", "loss": "2.578751"}
    4.5
        {"epo": "8", "R1": "84.985", "R5": "98.736", "R10": "99.494", "mAP": "91.527", "mINP": "91.527", "lr": "7.4e-05", "entropy": "0.080887", "loss": "2.116667"}

## exp5
**entropy + ss + unc_temper_learn**
    {"epo": "6", "R1": "85.44", "R5": "98.888", "R10": "99.393", "mAP": "91.74", "mINP": "91.74", "lr": "0.00041", "entropy": "0.035814", "loss": "5.017853"}
- exp5.1-5
    5.1
        {"epo": "0", "R1": "84.732", "R5": "98.989", "R10": "99.596", "mAP": "91.472", "mINP": "91.472", "lr": "4e-05", "entropy": "0.315383", "loss": "7.265886"}
    5.2
        {"epo": "7", "R1": "85.339", "R5": "98.686", "R10": "99.343", "mAP": "91.646", "mINP": "91.646", "lr": "0.000197", "entropy": "0.05217", "loss": "5.916965"}
    5.3
        {"epo": "9", "R1": "84.833", "R5": "98.332", "R10": "99.191", "mAP": "91.207", "mINP": "91.207", "lr": "0.000754", "entropy": "0.032004", "loss": "5.464747"}
    5.4
        {"epo": "6", "R1": "84.732", "R5": "98.938", "R10": "99.444", "mAP": "91.371", "mINP": "91.371", "lr": "0.00041", "entropy": "0.035236", "loss": "1.887671"}

## exp6
**entropy + pl**
    {"epo": "1", "R1": "85.389", "R5": "98.635", "R10": "99.494", "mAP": "91.709", "mINP": "91.709", "lr": "0.000977", "entropy": "0.143004", "loss": "0.143004"}
- exp6.1-5
    6.0.1
        {"epo": "4", "R1": "85.137", "R5": "98.787", "R10": "99.444", "mAP": "91.594", "mINP": "91.594", "lr": "4.4e-05", "entropy": "0.217917", "loss": "0.217917"}
    6.0.2
        {"epo": "15", "R1": "85.49", "R5": "98.686", "R10": "99.444", "mAP": "91.797", "mINP": "91.797", "lr": "5e-05", "entropy": "0.070464", "loss": "0.070464"}
    6.1
        {"epo": "1", "R1": "85.137", "R5": "98.736", "R10": "99.494", "mAP": "91.59", "mINP": "91.59", "lr": "0.000977", "entropy": "0.149469", "loss": "0.149469"}
    6.1.1
        {"epo": "7", "R1": "85.035", "R5": "98.837", "R10": "99.444", "mAP": "91.539", "mINP": "91.539", "lr": "1.5e-05", "entropy": "0.205474", "loss": "0.205474"}
    6.1.2
        {"epo": "22", "R1": "85.137", "R5": "98.736", "R10": "99.494", "mAP": "91.637", "mINP": "91.637", "lr": "2.6e-05", "entropy": "0.062026", "loss": "0.062026"}
    6.2
        {"epo": "0", "R1": "85.187", "R5": "98.787", "R10": "99.393", "mAP": "91.668", "mINP": "91.668", "lr": "0.000597", "entropy": "0.294532", "loss": "0.294532"}
    6.2.1
        {"epo": "1", "R1": "85.086", "R5": "98.888", "R10": "99.494", "mAP": "91.597", "mINP": "91.597", "lr": "9.8e-05", "entropy": "0.301802", "loss": "0.301802"}
    6.2.2
        {"epo": "2", "R1": "85.339", "R5": "98.787", "R10": "99.393", "mAP": "91.675", "mINP": "91.675", "lr": "0.000471", "entropy": "0.141841", "loss": "0.141841"}
    6.3
        {"epo": "1", "R1": "84.783", "R5": "98.686", "R10": "99.343", "mAP": "91.37", "mINP": "91.37", "lr": "0.000977", "entropy": "0.139118", "loss": "0.139118"}
    6.3.1
        {"epo": "2", "R1": "84.884", "R5": "98.837", "R10": "99.444", "mAP": "91.499", "mINP": "91.499", "lr": "9.4e-05", "entropy": "0.267697", "loss": "0.267697"}
    6.3.2
        {"epo": "4", "R1": "84.681", "R5": "98.736", "R10": "99.444", "mAP": "91.38", "mINP": "91.38", "lr": "0.000219", "entropy": "0.097198", "loss": "0.097198"}
    6.4
        {"epo": "0", "R1": "84.732", "R5": "98.837", "R10": "99.343", "mAP": "91.344", "mINP": "91.344", "lr": "0.000597", "entropy": "0.305543", "loss": "0.305543"}
    6.4.1
        {"epo": "2", "R1": "85.086", "R5": "98.837", "R10": "99.444", "mAP": "91.618", "mINP": "91.618", "lr": "9.4e-05", "entropy": "0.267884", "loss": "0.267884"}
    6.4.2
        {"epo": "4", "R1": "85.44", "R5": "98.787", "R10": "99.444", "mAP": "91.767", "mINP": "91.767", "lr": "0.000219", "entropy": "0.102473", "loss": "0.102473"}
    6.5
        {"epo": "0", "R1": "85.187", "R5": "98.483", "R10": "99.292", "mAP": "91.607", "mINP": "91.607", "lr": "0.000597", "entropy": "0.310546", "loss": "0.310546"}
    6.5.1
        {"epo": "3", "R1": "84.833", "R5": "98.787", "R10": "99.444", "mAP": "91.512", "mINP": "91.512", "lr": "5.7e-05", "entropy": "0.243152", "loss": "0.243152"}
    6.5.2
        {"epo": "1", "R1": "84.732", "R5": "98.584", "R10": "99.343", "mAP": "91.388", "mINP": "91.388", "lr": "0.000488", "entropy": "0.198149", "loss": "0.198149"}

## exp7
**entropy + ss + unc + pl**
    {"epo": "1", "R1": "85.137", "R5": "98.837", "R10": "99.444", "mAP": "91.608", "mINP": "91.608", "lr": "0.000961", "entropy": "0.128144", "loss": "2.658938"}
7.1
    {"epo": "1", "R1": "84.833", "R5": "98.787", "R10": "99.494", "mAP": "91.457", "mINP": "91.457", "lr": "9.6e-05", "entropy": "0.255827", "loss": "2.707937"}
7.2
    {"epo": "0", "R1": "85.137", "R5": "98.837", "R10": "99.494", "mAP": "91.598", "mINP": "91.598", "lr": "0.000198", "entropy": "0.274943", "loss": "2.715255"}
7.3
    {"epo": "10", "R1": "85.49", "R5": "98.736", "R10": "99.393", "mAP": "91.75", "mINP": "91.75", "lr": "0.000137", "entropy": "0.024345", "loss": "2.619359"}
7.4
    {"epo": "4", "R1": "85.288", "R5": "98.787", "R10": "99.444", "mAP": "91.666", "mINP": "91.666", "lr": "8.9e-05", "entropy": "0.173761", "loss": "2.676484"}
7.5
    {"epo": "11", "R1": "85.339", "R5": "98.584", "R10": "99.292", "mAP": "91.634", "mINP": "91.634", "lr": "6.5e-05", "entropy": "0.045799", "loss": "2.627524"}
7.6
    {"epo": "1", "R1": "85.288", "R5": "98.736", "R10": "99.393", "mAP": "91.715", "mINP": "91.715", "lr": "0.000961", "entropy": "0.145646", "loss": "2.66558"}
7.7
    {"epo": "8", "R1": "85.238", "R5": "98.736", "R10": "99.343", "mAP": "91.69", "mINP": "91.69", "lr": "3.7e-05", "entropy": "0.150832", "loss": "2.667852"}
7.8
    {"epo": "3", "R1": "85.44", "R5": "98.787", "R10": "99.393", "mAP": "91.742", "mINP": "91.742", "lr": "0.000462", "entropy": "0.098569", "loss": "2.647607"}

## exp8
**entropy + iaug**
    {"epo": "4", "R1": "82.154", "R5": "98.534", "R10": "99.393", "mAP": "89.921", "mINP": "89.921", "lr": "0.000437", "entropy": "0.133625", "loss": "0.133625"}
    8.1
        {"epo": "0", "R1": "84.125", "R5": "98.989", "R10": "99.494", "mAP": "91.204", "mINP": "91.204", "lr": "6e-05", "entropy": "1.05116", "loss": "1.05116"}
    8.2
        {"epo": "28", "R1": "79.474", "R5": "98.686", "R10": "99.343", "mAP": "88.652", "mINP": "88.652", "lr": "6e-06", "entropy": "0.152585", "loss": "0.152585"}
    8.3
        {"epo": "0", "R1": "83.822", "R5": "98.938", "R10": "99.494", "mAP": "91.014", "mINP": "91.014", "lr": "0.00031", "entropy": "0.840344", "loss": "0.840344"}
    8.4
        {"epo": "1", "R1": "84.58", "R5": "98.888", "R10": "99.444", "mAP": "91.386", "mINP": "91.386", "lr": "8.8e-05", "entropy": "0.860963", "loss": "0.860963"}
    8.5
        {"epo": "0", "R1": "84.53", "R5": "98.888", "R10": "99.444", "mAP": "91.375", "mINP": "91.375", "lr": "0.000155", "entropy": "0.941851", "loss": "0.941851"}

## exp9
**entropy + ss + unc + iaug**
- 由于exp8效果不好，不试验exp9的setting了

## exp10
**entropy_steps + ss + unc**
    nohup python3 run.py --tta --task "tta_exp10"> logs/tta_exp10.log 2>&1 &
    10.0
        {"epo": "8", "R1": "84.934", "R5": "98.686", "R10": "99.393", "mAP": "91.397", "mINP": "91.397", "lr": "0.000367", "entropy": "0.032907", "loss": "2.622468"}
    10.1
        {"epo": "2", "R1": "85.389", "R5": "98.837", "R10": "99.343", "mAP": "91.79", "mINP": "91.79", "lr": "0.000729", "entropy": "0.043528", "loss": "2.626565"}
    10.2
        {"epo": "0", "R1": "84.985", "R5": "98.888", "R10": "99.494", "mAP": "91.633", "mINP": "91.633", "lr": "0.000787", "entropy": "0.158636", "loss": "2.670734"}
    10.3
        {"epo": "5", "R1": "84.328", "R5": "98.736", "R10": "99.343", "mAP": "91.037", "mINP": "91.037", "lr": "0.000166", "entropy": "0.021528", "loss": "2.618173"}
    10.4
        {"epo": "0", "R1": "84.53", "R5": "98.18", "R10": "99.039", "mAP": "91.081", "mINP": "91.081", "lr": "0.000869", "entropy": "0.116297", "loss": "2.654539"}
    10.5
        {"epo": "23", "R1": "84.429", "R5": "98.332", "R10": "99.343", "mAP": "90.941", "mINP": "90.941", "lr": "4.6e-05", "entropy": "0.006819", "loss": "2.612519"}
    10.6
        shell脚本遗漏了，但不影响实验结论，该setting无提升

## exp11
**entropy + iaug_itm**
    best = exp11
        {"epo": "8", "R1": "85.743", "R5": "98.635", "R10": "99.444", "mAP": "91.866", "mINP": "91.866", "lr": "0.000148", "entropy": "0.068551", "loss": "0.068551"}

## exp9
**entropy + ss + unc + iaug_itm**
- 等exp11结果，再决定是否加上img_aug
- 三个setting组合后效果下降
    best = exp9.9
        {"epo": "4", "R1": "85.086", "R5": "98.635", "R10": "99.292", "mAP": "91.554", "mINP": "91.554", "lr": "0.000957", "entropy": "0.092873", "loss": "2.711785"}

## exp12
**entropy + plv1**
- coding~

## exp13
**entropy + ss + unc + plv1**
- 等exp12结果，再决定是否加上plv1

## exp14
**entropy + unc_temper_learn_coeffi**
    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2))).mean()
        tensor(0.2211, device='cuda:0', grad_fn=<MeanBackward0>)
    uncertainty.mean()
        tensor(2.6295, device='cuda:0', grad_fn=<MeanBackward0>)
    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2)*-3 )).mean()
        tensor(10.5920, device='cuda:0', grad_fn=<MeanBackward0>)
    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2)*-2 )).mean()
        tensor(4.0260, device='cuda:0', grad_fn=<MeanBackward0>)
    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2)*-1 )).mean()
        tensor(1.5303, device='cuda:0', grad_fn=<MeanBackward0>)
    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2)*-1.5 )).mean()
        tensor(2.4822, device='cuda:0', grad_fn=<MeanBackward0>)
    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2)*-1.6 )).mean()
        tensor(2.7343, device='cuda:0', grad_fn=<MeanBackward0>)

    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2) )).mean() / (uncertainty).mean()
        tensor(0.0841, device='cuda:0', grad_fn=<DivBackward0>)
    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2)*5 )).mean() / (uncertainty).mean()
        tensor(0.0018, device='cuda:0', grad_fn=<DivBackward0>)
    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2)*2 )).mean() / (uncertainty).mean()
        tensor(0.0320, device='cuda:0', grad_fn=<DivBackward0>)
    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2)*0 )).mean() / (uncertainty).mean()
        tensor(0.2212, device='cuda:0', grad_fn=<DivBackward0>)
**根据以上分析有两个思路: 1 unc_temper以-2作为初始值训练 2 unc_coeffi使用一个小于0.2212/0.0841/0.0320/0.0018的值(需要根据unc_temper初始值决定)**
    best = exp14.2
        {"epo": "26", "R1": "85.238", "R5": "98.635", "R10": "99.343", "mAP": "91.602", "mINP": "91.602", "lr": "2.5e-05", "entropy": "0.038986", "loss": "0.031098", "entr_unc": "0.000636", "uncertainty": "60.923837", "uncertainty_multiply_coeffi": "0.030462"}

## exp15
**entropy + ss + unc_temper_learn_coeffi**
(entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2) *1 )).mean() / (uncertainty).mean()
    tensor(0.0264, device='cuda:0', grad_fn=<DivBackward0>)
(entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2) *2 )).mean() / (uncertainty).mean()
    tensor(0.0101, device='cuda:0', grad_fn=<DivBackward0>)
(entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2) *5 )).mean() / (uncertainty).mean()
    tensor(0.0006, device='cuda:0', grad_fn=<DivBackward0>)
(entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2) *0 )).mean() / (uncertainty).mean()
    tensor(0.0688, device='cuda:0', grad_fn=<DivBackward0>)
(entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2) *-2 )).mean() / (uncertainty).mean()
    tensor(0.4681, device='cuda:0', grad_fn=<DivBackward0>)
(entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2) *-3 )).mean() / (uncertainty).mean()
    tensor(1.2205, device='cuda:0', grad_fn=<DivBackward0>)
**由于增加了sample selection，uncertainty的质量更好，所以loss会更小，使得uncertainty本身数值和entropy/uncertainty的差距进一步拉大，因此有:**
**1 unc_temper以-3作为初始值训练 2 unc_coeffi使用一个小于0.01（unc_temper=0，1），0.001（unc_temper=2），0.0001（unc_temper=5）的值**
    best = exp15.4
        {"epo": "4", "R1": "85.288", "R5": "98.433", "R10": "99.292", "mAP": "91.532", "mINP": "91.532", "lr": "0.000889", "entropy": "0.047062", "loss": "0.044365", "entr_unc": "0.017577", "uncertainty": "2.678756", "uncertainty_multiply_coeffi": "0.026788"}

## exp16
**entropy + ss + unc + plv1 + iaug_itm** unc_temper_learn_coeffi
- 等exp11结果，再决定是否加上img_aug **exp11有提升，等exp9结果再决定是否增加img_aug**
- 等exp12结果，再决定是否加上plv1
- 等exp15结果，再决定是否加上unc_temper_learn_coeffi **exp14和exp15都不如固定unc_temper，exp16不增加unc_temper_learn_coeffi的方法**


# exp rerun 2
- 修复了随机数种子未生效的问题

- 修复了uncertainty维度与entropy不一致的问题，该问题会导致loss=entropy/uncertainty成为一个(bs,bs)的tensor

- 重新检查cmp_xvlm的cross_modal模型结构设计，并设置了仅更新text_encoder的后6层bertlayer
- 重新检查Tent系列方法的setting设置，设置了tta时model.train()和batchnorm,layernorm参数更新方案
- 目前方案为：
    1. dropout： 所有dropout通过model.train()打开（包括visison_encoder、text_encoder前6层、其他modules 和 需要梯度更新的实现了itm.cross_modal功能的text_encoder后六层）
    2. requires_grad： text_encoder的后六层 和 itm_head的norm layer（无论batchnorm和layernorm）打开偏置params（γ、β）的梯度更新，但关闭track_running_stats并且train和eval都使用单个batch的stats（running_mean和running_var置为None）

## tta_debug
    CUDA_VISIBLE_DEVICES=1 python3 tta.py --config configs/tta_exp2.2.yaml --task tta_debug --output_dir output/tta_debug/exp2.2 --checkpoint checkpoint/cmp.pth --tta --bs 3 --epo 10 --lr 0.001 --seed 42

## exp0
**entropy**
    nohup python3 run.py --tta --task "tta_exp0"> logs/tta_exp0.log 2>&1 &

- exp0.1~4
    nohup python3 run.py --tta --task "tta_exp0.1"> logs/tta_exp0.1.log 2>&1 &

    nohup python3 run.py --tta --task "tta_exp0.2"> logs/tta_exp0.2.log 2>&1 &

    nohup python3 run.py --tta --task "tta_exp0.3"> logs/tta_exp0.3.log 2>&1 &

    nohup python3 run.py --tta --task "tta_exp0.4"> logs/tta_exp0.4.log 2>&1 &


## exp1
**entropy + ss**
    nohup python3 run.py --tta --task "tta_exp1"> logs/tta_exp1.log 2>&1 &

- exp1.1-5
    1.1

    1.2

    1.3

    1.4

    1.5


## exp2
**entropy + ss + unc**
    nohup python3 run.py --tta --task "tta_exp2"> logs/tta_exp2.log 2>&1 &
        {"epo": "2", "R1": "85.592", "R5": "98.989", "R10": "99.444", "mAP": "91.912", "mINP": "91.912", "lr": "0.000958", "entropy": "0.089008", "loss": "2.721839"}
- exp2.1-5
    2.1
        {"epo": "2", "R1": "85.642", "R5": "99.039", "R10": "99.444", "mAP": "91.972", "mINP": "91.972", "lr": "0.000958", "entropy": "0.089416", "loss": "2.710488"}
    2.2
        {"epo": "2", "R1": "85.693", "R5": "98.989", "R10": "99.444", "mAP": "91.958", "mINP": "91.958", "lr": "0.000958", "entropy": "0.088998", "loss": "2.643966"}
    2.3
        {"epo": "2", "R1": "85.44", "R5": "98.989", "R10": "99.444", "mAP": "91.837", "mINP": "91.837", "lr": "0.000958", "entropy": "0.08838", "loss": "2.346493"}
    2.4
        {"epo": "4", "R1": "85.086", "R5": "98.787", "R10": "99.444", "mAP": "91.537", "mINP": "91.537", "lr": "0.000889", "entropy": "0.046609", "loss": "1.575436"}
    2.5
        {"epo": "2", "R1": "85.541", "R5": "98.989", "R10": "99.444", "mAP": "91.878", "mINP": "91.878", "lr": "0.000958", "entropy": "0.089748", "loss": "1.213151"}

## exp3
**entropy + unc**
    nohup python3 run.py --tta --task "tta_exp3"> logs/tta_exp3.log 2>&1 &
        {"epo": "26", "R1": "85.541", "R5": "98.534", "R10": "99.444", "mAP": "91.756", "mINP": "91.756", "lr": "2.5e-05", "entropy": "0.035251", "loss": "2.702479"}
- exp3.1-5
    3.1
        {"epo": "3", "R1": "85.137", "R5": "98.686", "R10": "99.393", "mAP": "91.604", "mINP": "91.604", "lr": "0.000567", "entropy": "0.075018", "loss": "2.706798"}
    3.2
        {"epo": "3", "R1": "85.086", "R5": "98.686", "R10": "99.393", "mAP": "91.58", "mINP": "91.58", "lr": "0.000567", "entropy": "0.0757", "loss": "2.648713"}
    3.3
        {"epo": "1", "R1": "84.631", "R5": "98.837", "R10": "99.343", "mAP": "91.365", "mINP": "91.365", "lr": "0.000977", "entropy": "0.14208", "loss": "2.427362"}
    3.4
        {"epo": "21", "R1": "84.681", "R5": "98.736", "R10": "99.393", "mAP": "91.393", "mINP": "91.393", "lr": "6e-06", "entropy": "0.177065", "loss": "2.755205"}
    3.5
        {"epo": "7", "R1": "85.035", "R5": "98.787", "R10": "99.494", "mAP": "91.555", "mINP": "91.555", "lr": "7.7e-05", "entropy": "0.084124", "loss": "2.72065"}
    3.6
        {"epo": "2", "R1": "84.732", "R5": "98.736", "R10": "99.393", "mAP": "91.317", "mINP": "91.317", "lr": "0.00097", "entropy": "0.128045", "loss": "2.736988"}
    3.7
        {"epo": "0", "R1": "84.681", "R5": "99.039", "R10": "99.596", "mAP": "91.459", "mINP": "91.459", "lr": "3.1e-05", "entropy": "0.34605", "loss": "2.818058"}
    3.8
        {"epo": "8", "R1": "84.985", "R5": "98.736", "R10": "99.292", "mAP": "91.447", "mINP": "91.447", "lr": "0.00019", "entropy": "0.083007", "loss": "2.720241"}

## exp4 舍弃
**entropy + unc_temper_learn**
    {"epo": "2", "R1": "84.226", "R5": "98.483", "R10": "99.343", "mAP": "91.066", "mINP": "91.066", "lr": "0.000943", "entropy": "0.094519", "loss": "2.152995"}
- exp4.1-5
    4.1
        {"epo": "3", "R1": "85.086", "R5": "98.686", "R10": "99.444", "mAP": "91.55", "mINP": "91.55", "lr": "0.000567", "entropy": "0.071233", "loss": "5.156881"}
    4.2
        {"epo": "15", "R1": "84.732", "R5": "98.736", "R10": "99.393", "mAP": "91.357", "mINP": "91.357", "lr": "0.0001", "entropy": "0.040845", "loss": "74.359854"}
    4.3
        {"epo": "25", "R1": "84.58", "R5": "98.635", "R10": "99.393", "mAP": "91.271", "mINP": "91.271", "lr": "3.2e-05", "entropy": "0.036328", "loss": "0.573821"}
    4.4
        {"epo": "22", "R1": "84.681", "R5": "98.736", "R10": "99.393", "mAP": "91.393", "mINP": "91.393", "lr": "5e-06", "entropy": "0.174912", "loss": "2.578751"}
    4.5
        {"epo": "8", "R1": "84.985", "R5": "98.736", "R10": "99.494", "mAP": "91.527", "mINP": "91.527", "lr": "7.4e-05", "entropy": "0.080887", "loss": "2.116667"}

## exp5 舍弃
**entropy + ss + unc_temper_learn**
    {"epo": "6", "R1": "85.44", "R5": "98.888", "R10": "99.393", "mAP": "91.74", "mINP": "91.74", "lr": "0.00041", "entropy": "0.035814", "loss": "5.017853"}
- exp5.1-5
    5.1
        {"epo": "0", "R1": "84.732", "R5": "98.989", "R10": "99.596", "mAP": "91.472", "mINP": "91.472", "lr": "4e-05", "entropy": "0.315383", "loss": "7.265886"}
    5.2
        {"epo": "7", "R1": "85.339", "R5": "98.686", "R10": "99.343", "mAP": "91.646", "mINP": "91.646", "lr": "0.000197", "entropy": "0.05217", "loss": "5.916965"}
    5.3
        {"epo": "9", "R1": "84.833", "R5": "98.332", "R10": "99.191", "mAP": "91.207", "mINP": "91.207", "lr": "0.000754", "entropy": "0.032004", "loss": "5.464747"}
    5.4
        {"epo": "6", "R1": "84.732", "R5": "98.938", "R10": "99.444", "mAP": "91.371", "mINP": "91.371", "lr": "0.00041", "entropy": "0.035236", "loss": "1.887671"}

## exp6
**entropy + pl**
    {"epo": "1", "R1": "85.389", "R5": "98.635", "R10": "99.494", "mAP": "91.709", "mINP": "91.709", "lr": "0.000977", "entropy": "0.143004", "loss": "0.143004"}
- exp6.1-5
    6.0.1
        {"epo": "4", "R1": "85.137", "R5": "98.787", "R10": "99.444", "mAP": "91.594", "mINP": "91.594", "lr": "4.4e-05", "entropy": "0.217917", "loss": "0.217917"}
    6.0.2
        {"epo": "15", "R1": "85.49", "R5": "98.686", "R10": "99.444", "mAP": "91.797", "mINP": "91.797", "lr": "5e-05", "entropy": "0.070464", "loss": "0.070464"}
    6.1
        {"epo": "1", "R1": "85.137", "R5": "98.736", "R10": "99.494", "mAP": "91.59", "mINP": "91.59", "lr": "0.000977", "entropy": "0.149469", "loss": "0.149469"}
    6.1.1
        {"epo": "7", "R1": "85.035", "R5": "98.837", "R10": "99.444", "mAP": "91.539", "mINP": "91.539", "lr": "1.5e-05", "entropy": "0.205474", "loss": "0.205474"}
    6.1.2
        {"epo": "22", "R1": "85.137", "R5": "98.736", "R10": "99.494", "mAP": "91.637", "mINP": "91.637", "lr": "2.6e-05", "entropy": "0.062026", "loss": "0.062026"}
    6.2
        {"epo": "0", "R1": "85.187", "R5": "98.787", "R10": "99.393", "mAP": "91.668", "mINP": "91.668", "lr": "0.000597", "entropy": "0.294532", "loss": "0.294532"}
    6.2.1
        {"epo": "1", "R1": "85.086", "R5": "98.888", "R10": "99.494", "mAP": "91.597", "mINP": "91.597", "lr": "9.8e-05", "entropy": "0.301802", "loss": "0.301802"}
    6.2.2
        {"epo": "2", "R1": "85.339", "R5": "98.787", "R10": "99.393", "mAP": "91.675", "mINP": "91.675", "lr": "0.000471", "entropy": "0.141841", "loss": "0.141841"}
    6.3
        {"epo": "1", "R1": "84.783", "R5": "98.686", "R10": "99.343", "mAP": "91.37", "mINP": "91.37", "lr": "0.000977", "entropy": "0.139118", "loss": "0.139118"}
    6.3.1
        {"epo": "2", "R1": "84.884", "R5": "98.837", "R10": "99.444", "mAP": "91.499", "mINP": "91.499", "lr": "9.4e-05", "entropy": "0.267697", "loss": "0.267697"}
    6.3.2
        {"epo": "4", "R1": "84.681", "R5": "98.736", "R10": "99.444", "mAP": "91.38", "mINP": "91.38", "lr": "0.000219", "entropy": "0.097198", "loss": "0.097198"}
    6.4
        {"epo": "0", "R1": "84.732", "R5": "98.837", "R10": "99.343", "mAP": "91.344", "mINP": "91.344", "lr": "0.000597", "entropy": "0.305543", "loss": "0.305543"}
    6.4.1
        {"epo": "2", "R1": "85.086", "R5": "98.837", "R10": "99.444", "mAP": "91.618", "mINP": "91.618", "lr": "9.4e-05", "entropy": "0.267884", "loss": "0.267884"}
    6.4.2
        {"epo": "4", "R1": "85.44", "R5": "98.787", "R10": "99.444", "mAP": "91.767", "mINP": "91.767", "lr": "0.000219", "entropy": "0.102473", "loss": "0.102473"}
    6.5
        {"epo": "0", "R1": "85.187", "R5": "98.483", "R10": "99.292", "mAP": "91.607", "mINP": "91.607", "lr": "0.000597", "entropy": "0.310546", "loss": "0.310546"}
    6.5.1
        {"epo": "3", "R1": "84.833", "R5": "98.787", "R10": "99.444", "mAP": "91.512", "mINP": "91.512", "lr": "5.7e-05", "entropy": "0.243152", "loss": "0.243152"}
    6.5.2
        {"epo": "1", "R1": "84.732", "R5": "98.584", "R10": "99.343", "mAP": "91.388", "mINP": "91.388", "lr": "0.000488", "entropy": "0.198149", "loss": "0.198149"}

## exp7
**entropy + ss + unc + pl**
    {"epo": "1", "R1": "85.137", "R5": "98.837", "R10": "99.444", "mAP": "91.608", "mINP": "91.608", "lr": "0.000961", "entropy": "0.128144", "loss": "2.658938"}
7.1
    {"epo": "1", "R1": "84.833", "R5": "98.787", "R10": "99.494", "mAP": "91.457", "mINP": "91.457", "lr": "9.6e-05", "entropy": "0.255827", "loss": "2.707937"}
7.2
    {"epo": "0", "R1": "85.137", "R5": "98.837", "R10": "99.494", "mAP": "91.598", "mINP": "91.598", "lr": "0.000198", "entropy": "0.274943", "loss": "2.715255"}
7.3
    {"epo": "10", "R1": "85.49", "R5": "98.736", "R10": "99.393", "mAP": "91.75", "mINP": "91.75", "lr": "0.000137", "entropy": "0.024345", "loss": "2.619359"}
7.4
    {"epo": "4", "R1": "85.288", "R5": "98.787", "R10": "99.444", "mAP": "91.666", "mINP": "91.666", "lr": "8.9e-05", "entropy": "0.173761", "loss": "2.676484"}
7.5
    {"epo": "11", "R1": "85.339", "R5": "98.584", "R10": "99.292", "mAP": "91.634", "mINP": "91.634", "lr": "6.5e-05", "entropy": "0.045799", "loss": "2.627524"}
7.6
    {"epo": "1", "R1": "85.288", "R5": "98.736", "R10": "99.393", "mAP": "91.715", "mINP": "91.715", "lr": "0.000961", "entropy": "0.145646", "loss": "2.66558"}
7.7
    {"epo": "8", "R1": "85.238", "R5": "98.736", "R10": "99.343", "mAP": "91.69", "mINP": "91.69", "lr": "3.7e-05", "entropy": "0.150832", "loss": "2.667852"}
7.8
    {"epo": "3", "R1": "85.44", "R5": "98.787", "R10": "99.393", "mAP": "91.742", "mINP": "91.742", "lr": "0.000462", "entropy": "0.098569", "loss": "2.647607"}

## exp8 舍弃
**entropy + iaug**
    {"epo": "4", "R1": "82.154", "R5": "98.534", "R10": "99.393", "mAP": "89.921", "mINP": "89.921", "lr": "0.000437", "entropy": "0.133625", "loss": "0.133625"}
    8.1
        {"epo": "0", "R1": "84.125", "R5": "98.989", "R10": "99.494", "mAP": "91.204", "mINP": "91.204", "lr": "6e-05", "entropy": "1.05116", "loss": "1.05116"}
    8.2
        {"epo": "28", "R1": "79.474", "R5": "98.686", "R10": "99.343", "mAP": "88.652", "mINP": "88.652", "lr": "6e-06", "entropy": "0.152585", "loss": "0.152585"}
    8.3
        {"epo": "0", "R1": "83.822", "R5": "98.938", "R10": "99.494", "mAP": "91.014", "mINP": "91.014", "lr": "0.00031", "entropy": "0.840344", "loss": "0.840344"}
    8.4
        {"epo": "1", "R1": "84.58", "R5": "98.888", "R10": "99.444", "mAP": "91.386", "mINP": "91.386", "lr": "8.8e-05", "entropy": "0.860963", "loss": "0.860963"}
    8.5
        {"epo": "0", "R1": "84.53", "R5": "98.888", "R10": "99.444", "mAP": "91.375", "mINP": "91.375", "lr": "0.000155", "entropy": "0.941851", "loss": "0.941851"}

## exp9 改写，继承exp11
**entropy + ss + unc + iaug**
- 由于exp8效果不好，不试验exp9的setting了

## exp10
**entropy_steps + ss + unc**
    nohup python3 run.py --tta --task "tta_exp10"> logs/tta_exp10.log 2>&1 &
    10.0
        {"epo": "8", "R1": "84.934", "R5": "98.686", "R10": "99.393", "mAP": "91.397", "mINP": "91.397", "lr": "0.000367", "entropy": "0.032907", "loss": "2.622468"}
    10.1
        {"epo": "2", "R1": "85.389", "R5": "98.837", "R10": "99.343", "mAP": "91.79", "mINP": "91.79", "lr": "0.000729", "entropy": "0.043528", "loss": "2.626565"}
    10.2
        {"epo": "0", "R1": "84.985", "R5": "98.888", "R10": "99.494", "mAP": "91.633", "mINP": "91.633", "lr": "0.000787", "entropy": "0.158636", "loss": "2.670734"}
    10.3
        {"epo": "5", "R1": "84.328", "R5": "98.736", "R10": "99.343", "mAP": "91.037", "mINP": "91.037", "lr": "0.000166", "entropy": "0.021528", "loss": "2.618173"}
    10.4
        {"epo": "0", "R1": "84.53", "R5": "98.18", "R10": "99.039", "mAP": "91.081", "mINP": "91.081", "lr": "0.000869", "entropy": "0.116297", "loss": "2.654539"}
    10.5
        {"epo": "23", "R1": "84.429", "R5": "98.332", "R10": "99.343", "mAP": "90.941", "mINP": "90.941", "lr": "4.6e-05", "entropy": "0.006819", "loss": "2.612519"}
    10.6
        shell脚本遗漏了，但不影响实验结论，该setting无提升

## exp11
**entropy + iaug_itm**
    best = exp11
        {"epo": "8", "R1": "85.743", "R5": "98.635", "R10": "99.444", "mAP": "91.866", "mINP": "91.866", "lr": "0.000148", "entropy": "0.068551", "loss": "0.068551"}

## exp9
**entropy + ss + unc + iaug_itm**
- 等exp11结果，再决定是否加上img_aug
- 三个setting组合后效果下降
    best = exp9.9
        {"epo": "4", "R1": "85.086", "R5": "98.635", "R10": "99.292", "mAP": "91.554", "mINP": "91.554", "lr": "0.000957", "entropy": "0.092873", "loss": "2.711785"}

## exp12 待定
**entropy + plv1**
- coding~

## exp13 待定
**entropy + ss + unc + plv1**
- 等exp12结果，再决定是否加上plv1

## exp14
**entropy + unc_temper_learn_coeffi**
    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2))).mean()
        tensor(0.2211, device='cuda:0', grad_fn=<MeanBackward0>)
    uncertainty.mean()
        tensor(2.6295, device='cuda:0', grad_fn=<MeanBackward0>)
    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2)*-3 )).mean()
        tensor(10.5920, device='cuda:0', grad_fn=<MeanBackward0>)
    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2)*-2 )).mean()
        tensor(4.0260, device='cuda:0', grad_fn=<MeanBackward0>)
    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2)*-1 )).mean()
        tensor(1.5303, device='cuda:0', grad_fn=<MeanBackward0>)
    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2)*-1.5 )).mean()
        tensor(2.4822, device='cuda:0', grad_fn=<MeanBackward0>)
    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2)*-1.6 )).mean()
        tensor(2.7343, device='cuda:0', grad_fn=<MeanBackward0>)

    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2) )).mean() / (uncertainty).mean()
        tensor(0.0841, device='cuda:0', grad_fn=<DivBackward0>)
    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2)*5 )).mean() / (uncertainty).mean()
        tensor(0.0018, device='cuda:0', grad_fn=<DivBackward0>)
    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2)*2 )).mean() / (uncertainty).mean()
        tensor(0.0320, device='cuda:0', grad_fn=<DivBackward0>)
    (entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2)*0 )).mean() / (uncertainty).mean()
        tensor(0.2212, device='cuda:0', grad_fn=<DivBackward0>)
**根据以上分析有两个思路: 1 unc_temper以-2作为初始值训练 2 unc_coeffi使用一个小于0.2212/0.0841/0.0320/0.0018的值(需要根据unc_temper初始值决定)**
    best = exp14.2
        {"epo": "26", "R1": "85.238", "R5": "98.635", "R10": "99.343", "mAP": "91.602", "mINP": "91.602", "lr": "2.5e-05", "entropy": "0.038986", "loss": "0.031098", "entr_unc": "0.000636", "uncertainty": "60.923837", "uncertainty_multiply_coeffi": "0.030462"}

## exp15
**entropy + ss + unc_temper_learn_coeffi**
(entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2) *1 )).mean() / (uncertainty).mean()
    tensor(0.0264, device='cuda:0', grad_fn=<DivBackward0>)
(entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2) *2 )).mean() / (uncertainty).mean()
    tensor(0.0101, device='cuda:0', grad_fn=<DivBackward0>)
(entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2) *5 )).mean() / (uncertainty).mean()
    tensor(0.0006, device='cuda:0', grad_fn=<DivBackward0>)
(entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2) *0 )).mean() / (uncertainty).mean()
    tensor(0.0688, device='cuda:0', grad_fn=<DivBackward0>)
(entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2) *-2 )).mean() / (uncertainty).mean()
    tensor(0.4681, device='cuda:0', grad_fn=<DivBackward0>)
(entropy / torch.exp( (1 - (proba_top1_sim + proba_inversed_sim) / 2) *-3 )).mean() / (uncertainty).mean()
    tensor(1.2205, device='cuda:0', grad_fn=<DivBackward0>)
**由于增加了sample selection，uncertainty的质量更好，所以loss会更小，使得uncertainty本身数值和entropy/uncertainty的差距进一步拉大，因此有:**
**1 unc_temper以-3作为初始值训练 2 unc_coeffi使用一个小于0.01（unc_temper=0，1），0.001（unc_temper=2），0.0001（unc_temper=5）的值**
    best = exp15.4
        {"epo": "4", "R1": "85.288", "R5": "98.433", "R10": "99.292", "mAP": "91.532", "mINP": "91.532", "lr": "0.000889", "entropy": "0.047062", "loss": "0.044365", "entr_unc": "0.017577", "uncertainty": "2.678756", "uncertainty_multiply_coeffi": "0.026788"}

## exp16
**entropy + ss + unc + plv1 + iaug_itm** unc_temper_learn_coeffi
- 等exp11结果，再决定是否加上img_aug **exp11有提升，等exp9结果再决定是否增加img_aug**
- 等exp12结果，再决定是否加上plv1
- 等exp15结果，再决定是否加上unc_temper_learn_coeffi **exp14和exp15都不如固定unc_temper，exp16不增加unc_temper_learn_coeffi的方法**


# exp rerun 3
- 全新的uncertainty和sample_selection策略
    - uncertainty: |a-b|/(a+b)  and  |log(a)-log(b)|
    - sample_selection: 放宽到互为topk

- 修复了随机数种子未生效的问题

- 修复了uncertainty维度与entropy不一致的问题，该问题会导致loss=entropy/uncertainty成为一个(bs,bs)的tensor

- 重新检查cmp_xvlm的cross_modal模型结构设计，并设置了仅更新text_encoder的后6层bertlayer
- 重新检查Tent系列方法的setting设置，设置了tta时model.train()和batchnorm,layernorm参数更新方案
- 目前方案为：
    1. dropout： 所有dropout通过model.train()打开（包括visison_encoder、text_encoder前6层、其他modules 和 需要梯度更新的实现了itm.cross_modal功能的text_encoder后六层）
    2. requires_grad： text_encoder的后六层 和 itm_head的norm layer（无论batchnorm和layernorm）打开偏置params（γ、β）的梯度更新，但关闭track_running_stats并且train和eval都使用单个batch的stats（running_mean和running_var置为None）

## tta_debug
    CUDA_VISIBLE_DEVICES=0 python3 tta.py --config configs_rerun3/exp_debug.yaml --task exp_debug --output_dir output_rerun3/exp_debug --checkpoint checkpoint/cmp.pth --tta --bs 3 --epo 10 --lr 0.001 --seed 42

    CUDA_VISIBLE_DEVICES=0 python3 tta.py --config configs_rerun3/exp5.yaml --task exp_debug --output_dir output_rerun3/exp_debug/exp5 --checkpoint checkpoint/cmp.pth --tta --bs 3 --epo 10 --lr 0.001 --seed 42

    CUDA_VISIBLE_DEVICES=0 python3 tta.py --config configs_rerun3/exp2.0.1.yaml --task exp_debug --output_dir output_rerun3/exp_debug/exp2.0.1 --checkpoint checkpoint/cmp.pth --tta

## exp0
- ss=all
    "49", "R1": "85.187", "R5": "98.584", "R10": "99.242", "mAP": "91.62", "mINP": "91.62"
     第二遍结果：
    "2", "R1": "85.288", "R5": "98.736", "R10": "99.393", "mAP": "91.709", "mINP": "91.709"

## exp1
- ss=topk
    "29", "R1": "85.541", "R5": "98.483", "R10": "99.343", "mAP": "91.719", "mINP": "91.719"
    "39", "R1": "85.541", "R5": "98.686", "R10": "99.292", "mAP": "91.747", "mINP": "91.747"
    第二遍结果：
    "14", "R1": "85.035", "R5": "98.635", "R10": "99.494", "mAP": "91.479", "mINP": "91.479"

## exp2 top1实际未生效，重跑~
- top1实际未生效，ss=top1后又进入ss=all的代码逻辑，重跑~
- ss=top1
    top1改完重跑后结果：
    "29", "R1": "85.288", "R5": "98.736", "R10": "99.393", "mAP": "91.668", "mINP": "91.668"

## exp3 重跑~  去掉uncertainty_coeffi放大loss，再次重跑~
- ss=topk
- diff_div_mean
    "49", "R1": "85.137", "R5": "98.736", "R10": "99.343", "mAP": "91.6", "mINP": "91.6"
    第二遍结果：
    "14", "R1": "84.985", "R5": "98.787", "R10": "99.393", "mAP": "91.578", "mINP": "91.578"
    第三遍结果：

## exp4
- ss=topk
- abs_diff_log
    "49", "R1": "84.985", "R5": "98.938", "R10": "99.494", "mAP": "91.566", "mINP": "91.566"

## exp5 重跑~
- ss=topk
- inversed_recall_proba
    "39", "R1": "85.187", "R5": "98.584", "R10": "99.444", "mAP": "91.485", "mINP": "91.485"
    第二遍结果：
    "4", "R1": "85.541", "R5": "98.787", "R10": "99.393", "mAP": "91.827", "mINP": "91.827"

## exp6 去掉uncertainty_coeffi放大loss，再次重跑~
- ss=all
- diff_div_mean
    "4", "R1": "85.137", "R5": "98.736", "R10": "99.343", "mAP": "91.545", "mINP": "91.545"
    第三遍结果：

## exp7 top1实际未生效 重跑~ 去掉uncertainty_coeffi放大loss，再次重跑~
- top1实际未生效，ss=top1后又进入ss=all的代码逻辑，重跑~
- ss=top1
- diff_div_mean
    "4", "R1": "85.137", "R5": "98.736", "R10": "99.343", "mAP": "91.545", "mINP": "91.545"
    top1改完重跑后结果：
    "39", "R1": "85.187", "R5": "98.787", "R10": "99.343", "mAP": "91.601", "mINP": "91.601"
    第三遍结果：

