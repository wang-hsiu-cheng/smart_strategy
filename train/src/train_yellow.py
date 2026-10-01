import os
import config
from game_env import RobotMatchEnv
from metadata_utils import resolve_step_lineage
from metadata_utils import tag_latest_tb_event
from metadata_utils import save_training_metadata
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv

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
    cfg = config.get_train_config("yellow")

    # 1. 自動推導步數血統
    base_steps, cumulative_steps = resolve_step_lineage(
        pretrained_path=cfg["pretrained_path"],
        delta_timesteps=cfg["total_timesteps"]
    )

    print("=" * 60)
    print(f"[*] 訓練隊伍           : [{cfg['color'].upper()}]")
    print(f"[*] 父模型起點步數     : {base_steps:,} 步")
    print(f"[*] 當次推進步數 (Δ)   : {cfg['total_timesteps']:,} 步")
    print(f"[*] 累計總步數 (目標)  : {cumulative_steps:,} 步")
    print(f"[*] 權重儲存目標       : {cfg['save_path']}")
    print("=" * 60)

    # 1. 建立向量化環境
    env = SubprocVecEnv([
        make_env(
            rank=i,
            color=cfg["color"],
            enemy_model_path=cfg["enemy_model_path"]
        ) for i in range(config.TRAIN_NUM_ENVS)
    ])

    # 2. 載入 Checkpoint 或建立全新模型
    if cfg["pretrained_path"] and os.path.exists(cfg["pretrained_path"]):
        print(f"[+] Resuming training from checkpoint: {cfg['pretrained_path']}")
        model = MaskablePPO.load(
            cfg["pretrained_path"],
            env=env,
            custom_objects=cfg["ppo_params"],
            device=config.TRAIN_DEVICE
        )
    else:
        if cfg["pretrained_path"]:
            print(f"[!] Checkpoint '{cfg['pretrained_path']}' not found. Initializing from scratch.")
        model = MaskablePPO("MlpPolicy", env, **cfg["ppo_params"])
    
    # 3. 執行訓練
    # 注意：接續訓練時 reset_num_timesteps 設為 False，TensorBoard 才會從 800K 往後畫
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        tb_log_name=cfg["tb_log_name"],
        reset_num_timesteps=cfg["reset_timesteps"]
    )

    # 4. 存檔模型權重
    model.save(cfg["save_path"])
    env.close()

    # 5. 標記當次產出的 TensorBoard Event 檔名（加上 step_800k_to_1200k）
    run_log_dir = os.path.join(config.TENSORBOARD_LOG_DIR, cfg["tb_log_name"])
    tagged_event = tag_latest_tb_event(run_log_dir, base_steps, cumulative_steps)

    # 6. 保存語意清晰的 metadata.json
    save_training_metadata(
        cfg=cfg,
        save_path=cfg["save_path"],
        base_timesteps=base_steps,
        cumulative_timesteps=cumulative_steps,
        tagged_event_file=tagged_event
    )

if __name__ == "__main__":
    train()