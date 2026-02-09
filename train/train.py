from stable_baselines3 import PPO
from game_env import RobotMatchEnv
import os

def train():
    # 建立環境
    env = RobotMatchEnv()
    # 設定 log 存放路徑
    logdir = "logs"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    # 初始化 PPO 模型
    # MlpPolicy 適合處理向量觀測資料
    model = PPO.load(
        "robot_strategy_v1_800K", 
        env, 
        verbose=1, 
        tensorboard_log=logdir,
        learning_rate=5e-5,
        n_steps=8192,
        ent_coef = 0.01,
        device="cpu"
    )
    # model = PPO.load("robot_strategy_v1", env=env, device="cpu")

    print("開始訓練...")
    model.learn(total_timesteps=300000, reset_num_timesteps=False)

    # 儲存模型
    model.save("robot_strategy_v1_1100K")
    print("模型已儲存")

if __name__ == "__main__":
    train()