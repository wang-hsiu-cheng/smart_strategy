from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from game_env import RobotMatchEnv
import os

def mask_fn(env):
    return env.action_masks()

def train():
    # build basic env
    env = RobotMatchEnv(my_color="blue")
    # use ActionMasker env
    env.load_enemy_model("../models/robot_strategy_y_v4-4_1.6M.zip")
    env = ActionMasker(env, mask_fn)
    
    # init MaskablePPO
    model = MaskablePPO.load(
        "MlpPolicy", 
        "../models/robot_strategy_b_v4-2_1.6M",
        env, 
        verbose=1, 
        tensorboard_log="../logs",
        learning_rate=1e-3,
        clip_range=0.1,
        n_steps=4096,
        batch_size=512,
        gamma=0.995,
        ent_coef=0.06,
        device="cpu"  # or "cuda"
    )

    # checkpoint_callback = CheckpointCallback(
    #     save_freq=400000,           # 每 500,000 步存一次
    #     save_path="../models/",     # 存檔路徑
    #     name_prefix="robot_strategy_b_v4-3",
    #     verbose=1
    # )

    model.learn(
        total_timesteps=800000, 
        tb_log_name="blue_v5", 
        reset_num_timesteps=False
        # callback=checkpoint_callback
    )
        
    model.save("../models/robot_strategy_b_v5_800K.zip")

if __name__ == "__main__":
    train()