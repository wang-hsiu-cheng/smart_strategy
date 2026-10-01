import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

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
# 模型發布相關配置
# ==============================================================================
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "")

# 自動錨定至專案根目錄下的 models 資料夾
PUBLISH_MODEL_NAME = os.getenv("PUBLISH_MODEL_NAME", "robot_strategy_y_test.zip")
PUBLISH_MODEL_PATH = str(MODELS_DIR / PUBLISH_MODEL_NAME)

PUBLISH_INCLUDE_LOGS = os.getenv("PUBLISH_INCLUDE_LOGS", "true").lower() == "true"
PUBLISH_GIT_PUSH = os.getenv("PUBLISH_GIT_PUSH", "true").lower() == "true"

# ==============================================================================
# 解析模型路徑並自動下載本地缺少的模型
# ==============================================================================
def resolve_model_path(model_filename: Optional[str]) -> Optional[str]:
    """
    解析模型路徑：
    1. 本地已存在 -> 直接返回絕對路徑。
    2. 本地不存在但有設定 HF_MODEL_REPO -> 自動從 Hugging Face 下載快取至 models 目錄。
    3. 未指定或皆不存在 -> 返回 None。
    """
    if not model_filename:
        return None

    local_path = MODELS_DIR / model_filename
    if local_path.exists():
        return str(local_path)

    # 本地不存在，嘗試從雲端拉取
    hf_repo = os.getenv("HF_MODEL_REPO")
    if hf_repo:
        try:
            print(f"[*] 本地未找到 {model_filename}，正在從 Hugging Face ({hf_repo}) 下載...")
            downloaded_path = hf_hub_download(
                repo_id=hf_repo,
                filename=model_filename,
                local_dir=str(MODELS_DIR)
            )
            # 嘗試一併抓取同名 .json 快照
            try:
                hf_hub_download(
                    repo_id=hf_repo,
                    filename=str(Path(model_filename).with_suffix(".json")),
                    local_dir=str(MODELS_DIR)
                )
            except Exception:
                pass

            return downloaded_path
        except Exception as e:
            print(f"[!] 無法從 Hugging Face 下載模型: {e}")

    return str(local_path)

# ==============================================================================
# 6. 訓練配置工廠函數 (動態提取 YELLOW / BLUE)
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
        "pretrained_path": resolve_model_path(pretrained_name),
        "enemy_model_path": resolve_model_path(enemy_model_name),
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