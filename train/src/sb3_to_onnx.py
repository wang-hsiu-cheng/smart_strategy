import torch
import numpy as np
from sb3_contrib import MaskablePPO

model = MaskablePPO.load("../models/robot_strategy_y_v2_800K.zip")
policy = model.policy.to("cpu")
policy.eval()

class OnnxablePolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        # 提取特徵 (Features Extractor)
        features = self.policy.extract_features(obs) # extract features
        # 經過 MLP 提取器得到 Actor 的隱藏層輸出
        latent_pi, _ = self.policy.mlp_extractor(features) # get middle action inside model
        # 得到動作的 Logits (18 維)
        return self.policy.action_net(latent_pi) # get final 18 actions

# declare dummy input
obs_tensor = torch.randn(1, 41)
onnx_policy = OnnxablePolicy(policy)

# export
torch.distributions.Distribution.set_default_validate_args(False) # close distribution validate

try:
    torch.onnx.export(
        onnx_policy,
        obs_tensor,
        "strategy_y_v2_800K.onnx",
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    print("export static network model")
except Exception as e:
    print(f"fail to export: {e}")