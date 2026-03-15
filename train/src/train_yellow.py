from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from game_env import RobotMatchEnv
import os

def mask_fn(env):
    return env.action_masks()

def train():
    # build basic env
    env = RobotMatchEnv(my_color="yellow")
    # use ActionMasker env
    # env.load_enemy_model("../models/robot_strategy_b_v2_500K.zip")
    env = ActionMasker(env, mask_fn)
    
    # init MaskablePPO
    model = MaskablePPO.load(
        "../models/robot_strategy_y_v2_500K", 
        # "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="../logs",
        learning_rate=1e-3,
        n_steps=4096,
        batch_size=512,
        gamma=0.995,
        ent_coef=0.01,
        device="cpu"  # or "cuda"
    )

    model.learn(total_timesteps=300000, tb_log_name="yellow", reset_num_timesteps=False)
    model.save("../models/robot_strategy_y_v2_800K")
    # model.save("../models/robot_strategy_y_v3_500K")

if __name__ == "__main__":
    train()