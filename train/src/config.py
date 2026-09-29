import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# ==============================================================================
# 1. 動態計算專案根目錄 (Root Anchor)
# ==============================================================================
# __file__ = .../rl_main/train/src/config.py
# parents[0] = .../rl_main/train/src
# parents[1] = .../rl_main/train
# parents[2] = .../rl_main  <-- 專案根目錄
SRC_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SRC_DIR.parent
ROOT_DIR = TRAIN_DIR.parent

# 明確指定載入根目錄下的 .env 檔案
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# ==============================================================================
# 2. 型別轉換輔助函式
# ==============================================================================
def _to_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes")

def _to_optional_str(val: Optional[str]) -> Optional[str]:
    if not val or val.strip().lower() in ("none", ""):
        return None
    return val.strip()

def _resolve_path(env_var: str, default_rel_path: str) -> Path:
    """將 .env 中的相對路徑轉換為基於 ROOT_DIR 的絕對路徑"""
    raw_path = os.getenv(env_var, default_rel_path)
    path_obj = Path(raw_path)
    if path_obj.is_absolute():
        return path_obj
    return (ROOT_DIR / path_obj).resolve()

# ==============================================================================
# 3. 通用路徑與硬體設定 (自動轉為絕對路徑)
# ==============================================================================
MODELS_DIR = _resolve_path("MODELS_DIR", "train/models")
TENSORBOARD_LOG_DIR = _resolve_path("TENSORBOARD_LOG_DIR", "train/logs")
ONNX_EXPORT_PATH = _resolve_path("ONNX_EXPORT_PATH", "train/models/robot_strategy.onnx")

BEST_MODEL_NAME = os.getenv("BEST_MODEL_NAME", "best_model.zip")
BEST_MODEL_PATH = str(MODELS_DIR / BEST_MODEL_NAME)

TRAIN_NUM_ENVS = int(os.getenv("TRAIN_NUM_ENVS", 4))
TRAIN_DEVICE = os.getenv("TRAIN_DEVICE", "cuda")

# 自動遞迴建立目標資料夾（若不存在）
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
ONNX_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 4. 推論 (inference.py) 設定
# ==============================================================================
INF_MODEL_NAME = os.getenv("INFERENCE_MODEL_NAME", "model.zip")
INF_MODEL_PATH = str(MODELS_DIR / INF_MODEL_NAME)
INF_DEVICE = os.getenv("INFERENCE_DEVICE", "cpu")
INF_RENDER_MODE = _to_optional_str(os.getenv("INFERENCE_RENDER_MODE"))
INF_MY_COLOR = os.getenv("INFERENCE_MY_COLOR", "yellow")

_inf_enemy = _to_optional_str(os.getenv("INFERENCE_ENEMY_MODEL_NAME"))
INF_ENEMY_MODEL_PATH = str(MODELS_DIR / _inf_enemy) if _inf_enemy else None

# ==============================================================================
# 5. 訓練配置工廠函數 (動態提取 YELLOW / BLUE)
# ==============================================================================
def get_train_config(color: str) -> Dict[str, Any]:
    prefix = color.upper()
    
    pretrained_name = _to_optional_str(os.getenv(f"{prefix}_PRETRAINED_MODEL"))
    save_model_name = os.getenv(f"{prefix}_SAVE_MODEL_NAME", f"robot_strategy_{color}_default.zip")
    enemy_model_name = _to_optional_str(os.getenv(f"{prefix}_ENEMY_MODEL_NAME"))
    
    return {
        "color": color.lower(),
        "tb_log_name": os.getenv(f"{prefix}_TB_LOG_NAME", f"{color}_run"),
        "total_timesteps": int(os.getenv(f"{prefix}_TOTAL_TIMESTEPS", 500000)),
        "reset_timesteps": _to_bool(os.getenv(f"{prefix}_RESET_TIMESTEPS"), False),
        "save_path": str(MODELS_DIR / save_model_name),
        "pretrained_path": str(MODELS_DIR / pretrained_name) if pretrained_name else None,
        "enemy_model_path": str(MODELS_DIR / enemy_model_name) if enemy_model_name else None,
        "ppo_params": {
            "learning_rate": float(os.getenv(f"{prefix}_PPO_LR", 3e-4)),
            "clip_range": float(os.getenv(f"{prefix}_PPO_CLIP_RANGE", 0.2)),
            "ent_coef": float(os.getenv(f"{prefix}_PPO_ENT_COEF", 0.05)),
            "n_steps": int(os.getenv(f"{prefix}_PPO_N_STEPS", 2048)),
            "batch_size": int(os.getenv(f"{prefix}_PPO_BATCH_SIZE", 64)),
            "gamma": float(os.getenv(f"{prefix}_PPO_GAMMA", 0.99)),
            "verbose": 1,
            "tensorboard_log": str(TENSORBOARD_LOG_DIR),
            "device": TRAIN_DEVICE,
        }
    }