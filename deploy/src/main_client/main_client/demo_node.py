import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
import numpy as np
from stable_baselines3 import PPO # 假設使用 stable-baselines3

class RLNav2Node(Node):
    def __init__(self):
        super().__init__('rl_nav2_node')
        
        # 1. 載入訓練好的模型
        self.model = PPO.load("robot_model_v1")
        self.get_logger().info("RL Model Loaded!")

        # 2. 定義 18 個動作對應的真實座標 (需與 Env 一致)
        self.target_coords = [
            [0.3, 0.5], [0.3, 1.0], [0.3, 1.5], [0.3, 2.0], # 0-3 蒐集區
            [1.7, 0.5], [1.7, 1.0], [1.7, 1.5], [1.7, 2.0], # 4-7 蒐集區
            [0.4, 0.2], [0.8, 0.2], [1.2, 0.2], [1.6, 0.2], [1.0, 0.5], # 8-12 放置區
            [0.4, 2.8], [0.8, 2.8], [1.2, 2.8], [1.6, 2.8], [1.0, 2.5]  # 13-17 放置區
        ]

        # 3. 初始化 Nav2 Action Client
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # 4. 啟動決策迴圈 (例如每秒做一次決策)
        self.timer = self.create_timer(1.0, self.make_decision)

    def make_decision(self):
        # --- A. 獲取並處理觀測值 ---
        # 實務上你需要從 Topic 訂閱目前的數量與座標
        # 這裡示範如何組建那 73 維向量
        raw_obs = self.gather_real_world_data() 
        norm_obs = self.normalize_observation(raw_obs)

        # --- B. 模型推論 ---
        """
        之後要改成監聽 Nav2 的 result 回傳
        只有當機器人抵達上一個目標後，才呼叫 model.predict() 決定下一個點
        否則 AI 可能會在一秒鐘內發送 30 個不同的導航請求，導致 Nav2 崩潰
        """
        action, _states = self.model.predict(norm_obs, deterministic=True)
        self.get_logger().info(f"RL Decision: Move to Area {action}")

        # --- C. 發送指令給 Nav2 ---
        self.send_nav2_goal(self.target_coords[action])

    def send_nav2_goal(self, coords):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        goal_msg.pose.pose.position.x = coords[0]
        goal_msg.pose.pose.position.y = coords[1]
        goal_msg.pose.pose.orientation.w = 1.0 # 暫不考慮轉向

        self._action_client.wait_for_server()
        self._action_client.send_goal_async(goal_msg)

    def normalize_observation(self, obs):
        # 務必使用與訓練時一模一樣的歸一化公式
        # 例如: (pos / 2.0) * 2 - 1
        return np.array(obs, dtype=np.float32)

    def gather_real_world_data(self):
        """
        Robot Pos: /odom 或 /tf
        Pantry Counts: 需要一個專門的 vision_node 辨識並發布剩餘數量的 Topic
        """
        # 這裡應從你的 Subscriber 獲取最新的數據
        # 回傳一個 73 維的 list
        return [0.0] * 73 

def main(args=None):
    rclpy.init(args=args)
    node = RLNav2Node()
    rclpy.spin(node)
    rclpy.shutdown()