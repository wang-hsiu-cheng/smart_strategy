import os
# 強制 PyTorch 使用 CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib
matplotlib.use('TkAgg') # 指定使用 TkAgg 彈出視窗
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from game_env import RobotMatchEnv

def main():
    # import trained model
    model_path = "../models/robot_strategy_v1_800K.zip"
    if not os.path.exists(model_path):
        print(f"can't find the model named {model_path}")
        return
        
    model = PPO.load(model_path, device="cpu")
    print("model import successfully, run with CPU")

    # initialize, set render as human
    env = RobotMatchEnv(render_mode="human")
    obs, _ = env.reset()

    # testing loop
    try:
        for step in range(3000):
            # model give the action
            action, _ = model.predict(obs, deterministic=True)
            
            # environment give the feedback
            obs, reward, terminated, truncated, info = env.step(action)
            
            # plot graph
            env.render()
            
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