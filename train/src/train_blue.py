import os
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from game_env import RobotMatchEnv
import config


def make_env(rank: int, color: str, enemy_model_path: str = None, seed: int = 0):
    """
    建立個別子進程環境，並於進程內部完成敵方模型載入與隨機種子設定
    """
    def _init():
        env = RobotMatchEnv(my_color=color)
        
        # 若有指定敵方模型，在子進程環境內部加載
        if enemy_model_path and os.path.exists(enemy_model_path):
            env.load_enemy_model(enemy_model_path)
            
        env.reset(seed=seed + rank)
        return env
    return _init


def train():
    # 1. 取得 Yellow 隊伍專屬設定 (由 .env 動態載入)
    cfg = config.get_train_config("blue")
    
    print("=" * 60)
    print(f"[*] Starting Training for: [{cfg['color'].upper()}]")
    print(f"[*] Parallel Envs     : {config.TRAIN_NUM_ENVS}")
    print(f"[*] Device            : {config.TRAIN_DEVICE}")
    print(f"[*] Total Timesteps   : {cfg['total_timesteps']}")
    print(f"[*] Learning Rate     : {cfg['ppo_params']['learning_rate']}")
    print(f"[*] Entropy Coef      : {cfg['ppo_params']['ent_coef']}")
    print(f"[*] Enemy Opponent    : {cfg['enemy_model_path']}")
    print(f"[*] Target Save Path  : {cfg['save_path']}")
    print("=" * 60)

    # 2. 建立多程序平行向量化環境
    env = SubprocVecEnv([
        make_env(
            rank=i,
            color=cfg["color"],
            enemy_model_path=cfg["enemy_model_path"]
        ) for i in range(config.TRAIN_NUM_ENVS)
    ])

    # 3. 判斷接續訓練 (Resuming) 或全新訓練 (From Scratch)
    if cfg["pretrained_path"] and os.path.exists(cfg["pretrained_path"]):
        print(f"[+] Resuming training from checkpoint: {cfg['pretrained_path']}")
        model = MaskablePPO.load(
            cfg["pretrained_path"],
            env=env,
            custom_objects=cfg["ppo_params"],  # 覆蓋為當前進度的超參數（LR、Entropy 等）
            device=config.TRAIN_DEVICE
        )
    else:
        if cfg["pretrained_path"]:
            print(f"[!] Warning: Pretrained path '{cfg['pretrained_path']}' not found. Training from scratch.")
        else:
            print("[+] Initializing new MaskablePPO model from scratch.")
            
        model = MaskablePPO(
            "MlpPolicy",
            env,
            **cfg["ppo_params"]
        )

    # 4. 開始訓練
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        tb_log_name=cfg["tb_log_name"],
        reset_num_timesteps=cfg["reset_timesteps"]
    )

    # 5. 儲存最終模型權重
    model.save(cfg["save_path"])
    print(f"[+] Training complete. Model saved to: {cfg['save_path']}")

    # 關閉向量化環境進程
    env.close()


if __name__ == "__main__":
    train()