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


# command

1. train
    python3 run.py --task "cmp" --dist "f4" --output_dir "output/cmp"
2. evaluate
    python3 run.py --task "cmp" --evaluate --dist "f4" --output_dir "output/cmp_eval" --checkpoint "checkpoint/cmp.pth"
    python3 run.py --task "cmp" --evaluate --dist "f2" --output_dir "output/cmp_eval" --checkpoint "checkpoint/cmp.pth"
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
### exp0.1~4
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
## exp1.1-5
    1.1 {"epo": "4", "R1": "84.783", "R5": "98.787", "R10": "99.343", "mAP": "91.405", "mINP": "91.405", "lr": "0.000445", "entropy": "0.059843", "loss": "0.059843"}
    1.2 {"epo": "0", "R1": "84.783", "R5": "98.989", "R10": "99.596", "mAP": "91.505", "mINP": "91.505", "lr": "4e-05", "entropy": "0.331566", "loss": "0.331566"}
    1.3 {"epo": "0", "R1": "84.681", "R5": "98.888", "R10": "99.494", "mAP": "91.378", "mINP": "91.378", "lr": "0.0002", "entropy": "0.289342", "loss": "0.289342"}
    1.4 {"epo": "0", "R1": "84.783", "R5": "98.888", "R10": "99.545", "mAP": "91.477", "mINP": "91.477", "lr": "0.0001", "entropy": "0.303752", "loss": "0.303752"}
    1.5 {"epo": "1", "R1": "84.732", "R5": "98.888", "R10": "99.545", "mAP": "91.463", "mINP": "91.463", "lr": "5.9e-05", "entropy": "0.293154", "loss": "0.293154"}

## exp2
**entropy + ss + unc**
    nohup python3 run.py --tta --task "tta_exp2"> logs/tta_exp2.log 2>&1 &
        {"epo": "3", "R1": "84.783", "R5": "98.736", "R10": "99.343", "mAP": "91.308", "mINP": "91.308", "lr": "0.000924", "entropy": "0.043163", "loss": "2.704784"}
## exp2.1-5
    2.1 {"epo": "3", "R1": "84.884", "R5": "98.736", "R10": "99.343", "mAP": "91.382", "mINP": "91.382", "lr": "0.000924", "entropy": "0.044278", "loss": "2.693614"}
    2.2 {"epo": "3", "R1": "84.681", "R5": "98.736", "R10": "99.292", "mAP": "91.24", "mINP": "91.24", "lr": "0.000924", "entropy": "0.042831", "loss": "2.626209"}
    2.3 {"epo": "4", "R1": "84.681", "R5": "98.736", "R10": "99.444", "mAP": "91.305", "mINP": "91.305", "lr": "0.000889", "entropy": "0.023986", "loss": "2.319043"}
    2.4 {"epo": "4", "R1": "84.732", "R5": "98.686", "R10": "99.444", "mAP": "91.321", "mINP": "91.321", "lr": "0.000889", "entropy": "0.024042", "loss": "1.563877"}
    2.5 {"epo": "4", "R1": "84.783", "R5": "98.837", "R10": "99.444", "mAP": "91.356", "mINP": "91.356", "lr": "0.000889", "entropy": "0.024167", "loss": "1.161326"}

## exp3
**entropy + ss + unc_temper_learn**

## exp4
**entropy + ss + unc_temper_learn + pl**

## exp5
**entropy + ss + unc_temper_learn + pl + iaug**


