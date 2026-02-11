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
        self.model = PPO.load("robot_model_v1")
        
        # 狀態追蹤
        self.is_moving = False
        self.current_goal_action = None  # 目前 Nav2 正在執行的 Action ID
        self.goal_handle = None          # 用於取消目標的 handle
        
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 啟動非同步監控定時器 (例如每 0.5 秒檢查一次環境)
        self.decision_timer = self.create_timer(0.5, self.async_monitor_logic)
        self.get_logger().info("RL Model Loaded!")

        # 2. 定義 18 個動作對應的真實座標 (需與 Env 一致)
        self.target_coords = [
            [0.3, 0.5], [0.3, 1.0], [0.3, 1.5], [0.3, 2.0], # 0-3 蒐集區
            [1.7, 0.5], [1.7, 1.0], [1.7, 1.5], [1.7, 2.0], # 4-7 蒐集區
            [0.4, 0.2], [0.8, 0.2], [1.2, 0.2], [1.6, 0.2], [1.0, 0.5], # 8-12 放置區
            [0.4, 2.8], [0.8, 2.8], [1.2, 2.8], [1.6, 2.8], [1.0, 2.5]  # 13-17 放置區
        ]

    def async_monitor_logic(self):
        # 1. 獲取觀測值
        raw_obs = self.gather_real_world_data()
        norm_obs = self.normalize_observation(raw_obs)

        # 2. 模型推論
        new_action, _ = self.model.predict(norm_obs, deterministic=True)

        # 3. 核心判斷邏輯
        if not self.is_moving:
            # 情況 A：目前閒置，直接出發
            self.get_logger().info(f"Idle. Starting Action {new_action}")
            self.execute_new_action(new_action)
        
        elif new_action != self.current_goal_action:
            self.change_request_count += 1
            if self.change_request_count > 3: # 必須連續 3 次檢查都覺得要換，才動手
                self.get_logger().warn(f"Decision changed! {self.current_goal_action} -> {new_action}. Re-routing...")
                self.cancel_and_reroute(new_action)
                self.change_request_count = 0
        else:
            self.change_request_count = 0

    def execute_new_action(self, action_index):
        self.is_moving = True
        self.current_goal_action = action_index
        
        goal_msg = NavigateToPose.Goal()
        coords = self.target_coords[action_index]
        goal_msg.pose.pose.position.x = coords[0]
        goal_msg.pose.pose.position.y = coords[1]
        goal_msg.pose.pose.orientation.w = 1.0

        self.get_logger().info(f"Sending Goal: Area {action_index}")
        
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def cancel_and_reroute(self, next_action):
        """取消當前目標並立即前往新目標"""
        if self.goal_handle is not None:
            # 發送取消請求
            self.goal_handle.cancel_goal_async()
            self.get_logger().info("Current goal cancellation requested.")
        
        # 立即執行新動作（Nav2 會處理 preempting，或者等待 cancel 完成）
        self.execute_new_action(next_action)

    def goal_response_callback(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.is_moving = False
            return

        self._get_result_future = self.goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        # 當導航結束時（成功、失敗或被取消）
        status = future.result().status
        # 注意：如果是被我們主動取消的，狀態會是 STATUS_CANCELED (4)
        if status == 4:
            self.get_logger().info("Goal was successfully cancelled for re-routing.")
        else:
            self.get_logger().info("Arrival or Failure. Waiting for next timer tick.")
            self.is_moving = False
            self.current_goal_action = None
            self.goal_handle = None
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