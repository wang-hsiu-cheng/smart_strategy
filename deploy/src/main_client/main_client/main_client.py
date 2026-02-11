import rclpy
from rclpy.node import Node
import numpy as np
from stable_baselines3 import PPO # 假設使用 stable-baselines3

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from std_msgs.msg import Int32
from std_msgs.msg import Float32
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose

class MainClient():
    def __init__(self):
        super().__init__('main_node')
        # declare robot veriable
        self.start_time = self.get_clock().now()
        self.timer_period = 0.1
        self.crate_robot = 0
        self.cursor_target.pose.position.x = 0.7
        self.cursor_target.pose.position.y = 0.0
        # declare map info
        self.field_width = 3.0
        self.field_height = 2.0
        self.max_collect_capacity = 4.0   # 假設訓練時蒐集區最大容量為 4
        self.max_pantry_capacity = 5.0  # 假設放置區單一顏色最大容量
        self.max_robot_capacity = 8.0    # 機器人最大負載
        self.collect_time_steps = 30 * 5     # 訓練時設定的等待步數
        self.place_time_steps = 30 * 5
        # declare map condition
        self.crate_nest = 0
        self.crate_collect = np.zeros(8)
        self.crate_blue = np.zeros(10)
        self.cursor_pose.pose.position.x = 1.3
        self.cursor_pose.pose.position.y = 0.0
        # declare puplisher
        self.time_pub = self.create_publisher(Float32, '/main/time', 10)
        self.crate_robot_pub = self.create_publisher(Int32, '/main/crate', 10)
        # declare subscriber
        self.pantry_blue_sub = self.create_subscription(Int32MultiArray, '/camera/pantry_blue', self.pantry_blue_callback, 10)
        self.pantry_yellow_sub = self.create_subscription(Int32MultiArray, '/camera/pantry_yellow', self.pantry_yellow_callback, 10)
        self.crate_nest_sub = self.create_subscription(Int32, '/camera/nest_crate', self.crate_nest_callback, 10)
        self.crate_collect_sub = self.create_subscription(Int32MultiArray, '/camera/collect_area_crate', self.crate_collect_callback, 10)
        self.cursor_pose_sub = self.create_subscription(PoseStamped, '/camera/cursor_pose', self.cursor_pose_callback, 10)
        # declare action
        # declare tf listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # start main program
        self.clock = self.create_timer(self.timer_period, self.time_publisher)
        self.timer = self.create_timer(0.5, self.brain)
    def normalizer():
        pass
    def sensor_callback():
        pass
    def brain(self):
        self.robot_pose = self.get_robot_pose('/robot/base_link')
        self.rival_pose = self.get_robot_pose('/rival/base_link')
        self.start_waiting = self.elapsed_seconds
        norm_obs = self.get_norm_obs()
        
        self.waiting_time = self.elapsed_seconds - self.start_waiting
        crate_robot_msg = Int32()
        crate_robot_msg.data = float(self.crate_robot)
        self.crate_robot_pub.publish(crate_robot_msg)
    def pantry_blue_callback(self, msg):
        self.pantry_blue = list(msg.data)
    def pantry_yellow_callback(self, msg):
        self.pantry_yellow = list(msg.data)
    def crate_nest_callback(self, msg):
        self.crate_nest = int(msg.data)
    def crate_collect_callback(self, msg):
        self.crate_collect = list(msg.data)
    def cursor_pose_callback(self, msg):
        self.cursor_pose = msg.data
    def get_robot_pose(self, target_frame):
        pose_msg = PoseStamped()
        try:
            # 1. 查詢最新的座標轉換 (TransformStamped)
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('map', target_frame, now)

            # 2. 複製 Header (包含 Timestamp 和 Frame_id)
            # 這確保了位姿的時間戳與 TF 是一致的
            pose_msg.header = trans.header

            # 3. 映射平移數據 (Translation -> Position)
            pose_msg.pose.position.x = trans.transform.translation.x
            pose_msg.pose.position.y = trans.transform.translation.y
            pose_msg.pose.position.z = trans.transform.translation.z

            # 4. 映射旋轉數據 (Rotation -> Orientation)
            pose_msg.pose.orientation.x = trans.transform.rotation.x
            pose_msg.pose.orientation.y = trans.transform.rotation.y
            pose_msg.pose.orientation.z = trans.transform.rotation.z
            pose_msg.pose.orientation.w = trans.transform.rotation.w

            return pose_msg

        except Exception as e:
            self.get_logger().error(f"TF 轉換失敗: {str(e)}")
            return None
    def time_publisher(self):
        # 計算從啟動到現在經過的總時間 (秒)
        now = self.get_clock().now()
        elapsed_time_duration = now - self.start_time
        
        # 轉換為浮點數秒數
        self.elapsed_seconds = elapsed_time_duration.nanoseconds / 1e9
        
        # 建立訊息並發布
        msg = Float32()
        msg.data = float(self.elapsed_seconds)
        self.time_pub.publish(msg)
    def get_norm_obs(self):
        # --- 1. 處理位置資料 (Robot & Rival) ---
        # 假設 self.latest_robot_pose 是 geometry_msgs.msg.PoseStamped
        robot_pos = [self.robot_pose.pose.position.x, 
                    self.robot_pose.pose.position.y]
        enemy_pos = [self.rival_pose.pose.position.x, 
                    self.rival_pose.pose.position.y]
        
        robot_norm = self._normalize_pos(robot_pos)
        enemy_norm = self._normalize_pos(enemy_pos)

        # --- 2. 處理等待計時器 (1維) ---
        # self.waiting_time.data 是 Float32
        max_steps = max(self.collect_time_steps, self.place_time_steps) * 2
        wait_val = (self.waiting_time.data / max_steps) * 2 - 1
        wait_timer_norm = np.array([wait_val], dtype=np.float32)

        # --- 3. 處理蒐集區狀態 (8維) ---
        # self.crate_collect 是 np.array(size=8)
        collect_info = [(c / self.max_collect_capacity) * 2 - 1 for c in self.crate_collect]

        # --- 4. 處理放置區狀態 (20維) ---
        pantry_info = []
        for i in range(10):
            # 黃色歸一化
            y_norm = (self.crate_yellow[i] / self.max_pantry_capacity) * 2 - 1
            pantry_info.append(y_norm)
            # 藍色歸一化
            b_norm = (self.crate_blue[i] / self.max_pantry_capacity) * 2 - 1
            pantry_info.append(b_norm)

        # --- 5. 機上物品數量 (1維) ---
        # self.crate_robot 是 Int32
        held_val = (self.crate_robot.data / self.max_robot_capacity) * 2 - 1
        held_info = np.array([held_val], dtype=np.float32)

        # --- 6. 合併為 34 維向量 ---
        obs = np.concatenate([
            robot_norm,      # 2
            enemy_norm,      # 2
            wait_timer_norm, # 1
            collect_info,    # 8
            pantry_info,     # 20 (10x2)
            held_info        # 1
        ]).astype(np.float32)
        return obs

    def _normalize_pos(self, pos):
        """將 [0,3]x[0,2] 映射至 [-1,1]"""
        nx = (pos[0] / self.field_width) * 2 - 1
        ny = (pos[1] / self.field_height) * 2 - 1
        return [nx, ny]

def main(args=None):
    rclpy.init(args=args)
    node = MainClient()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()