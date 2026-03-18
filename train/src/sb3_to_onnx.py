import torch
import numpy as np
from sb3_contrib import MaskablePPO

# 1. 載入模型
model = MaskablePPO.load("../models/robot_strategy_y_v2_800K.zip")

# 2. 定義一個與 Observation Space 相符的 Dummy Input (41 維)
# 注意：若有 Action Mask，通常需要將 Mask 作為第二個輸入導出，或在部署端手動處理
dummy_input = torch.randn(1, 41) 

# 3. 提取 Policy 網路並設為評估模式
policy = model.policy.to("cpu")
policy.eval()

# 4. 執行導出
torch.onnx.export(
    policy,
    (dummy_input,),             # 模型輸入
    "strategy_y_v2_800K.onnx",  # 輸出路徑
    export_params=True,         # 包含權重
    opset_version=12,           # 建議 12 以上
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'}, 
        'output': {0: 'batch_size'}
    } # 支援 Batch
)