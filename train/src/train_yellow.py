from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv
from game_env import RobotMatchEnv
import os

def make_env(rank: int, seed: int = 0):
    def _init():
        env = RobotMatchEnv(my_color="yellow")
        return env
    return _init

def train():
    # build basic env
    num_envs = 4  # 依照你的 CPU 核心數調整，釋放單核過載
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    # env.load_enemy_model("../models/robot_strategy_b_v4_800K.zip")
    
    # init MaskablePPO
    model = MaskablePPO(
        # "../models/robot_strategy_y_v4-4_800K",
        "MlpPolicy",
        env,
        verbose=1, 
        tensorboard_log="../logs",
        learning_rate=5e-4,
        clip_range=0.15,
        n_steps=4096,
        batch_size=512,
        gamma=0.995,
        ent_coef=0.06,
        device="cuda"  # or "cuda"
    )

    model.learn(total_timesteps=800000, tb_log_name="yellow_test", reset_num_timesteps=False)
    model.save("../models/robot_strategy_y_test.zip")

if __name__ == "__main__":
    train()