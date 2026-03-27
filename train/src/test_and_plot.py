import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # force PyTorch use CPU

import matplotlib
matplotlib.use('TkAgg') # force use TkAgg window
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from game_env import RobotMatchEnv

def main():
    model_path = "../models/robot_strategy_y_v3_1.8M.zip" # import trained model
    model = MaskablePPO.load(model_path, device="cpu")
    print("model import successfully, run with CPU")

    # initialize, set render as human and select my color
    env = RobotMatchEnv(render_mode="human", my_color="yellow")
    env.load_enemy_model("../models/robot_strategy_b_v3_1.3M.zip") # load enemy model to conduct competition
    obs, _ = env.reset()

    try:
        for step in range(3000):
            action_masks = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True) # model give the action
            obs, reward, terminated, truncated, info = env.step(action) # environment give the feedback
            env.render() # plot graph

            print(f"Step {step}: {info}")
            if terminated or truncated:
                print(f"round stop, steps: {step}")
                break
                
    except KeyboardInterrupt:
        print("test interruped by user")
    finally:
        print("test complete")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()