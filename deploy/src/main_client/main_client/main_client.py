import math
import rclpy
from rclpy.node import Node
import numpy as np

from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from threading import Lock

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

from navigation_client import Navigate
from navigation_client import Dock
from navigation_client import BackRotateForward
from mission_client import MissionClient

class MainClient():
    def __init__(self):
        super().__init__('main_node')
        self.model = PPO.load("~/rl_main/train/models/robot_strategy_v1_800K")
        # --- 1. 定義執行緒組 ---
        # 監控組：用於高頻率的 RL 推論
        self.monitor_group = ReentrantCallbackGroup()
        # 執行組：用於處理 Nav2/Dock 動作（一次只做一件事）
        self.execution_group = MutuallyExclusiveCallbackGroup()
        self.navigate = Navigate(self, self.action_group)
        self.dock = Dock(self, self.action_group)
        self.back_rotate_forward = BackRotateForward(self, self.action_group)
        # --- 2. 共享資源與鎖 (Thread Safety) ---
        self.data_lock = Lock()
        self.current_action = None
        self.current_task = None
        self.change_counter = 0
        self.is_busy = False
        # declare robot veriable
        self.start_time = self.get_clock().now()
        self.timer_period = 0.1
        self.crate_robot = np.zeros(2)
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
        for i in range(8):
            for side in ['north', 'south', 'center']:
                param_name = f'goal_areas.collect_{i}.{side}'
                self.declare_parameter(param_name, [0.0, 0.0])
        for i in range(10):
            for side in ['north', 'south', 'east', 'west', 'center']:
                param_name = f'goal_areas.pantry_{i}.{side}'
                self.declare_parameter(param_name, [0.0, 0.0])
        self.declare_parameter('goal_areas.nest.south', [0.0, 0.0])
        self.declare_parameter('goal_areas.cursor.north', [0.0, 0.0])
        # declare map condition
        self.crate_nest = 0
        self.crate_collect = np.full(8, 4)
        self.crate_pantry_blue = np.zeros(10)
        self.crate_pantry_yellow = np.zeros(10)
        self.cursor_pose.pose.position.x = 1.3
        self.cursor_pose.pose.position.y = 0.0
        # declare puplisher
        self.time_pub = self.create_publisher(Float32, '/main/time', 10)
        self.crate_robot_pub = self.create_publisher(Int32, '/main/crate', 10)
        # declare subscriber
        # self.pantry_blue_sub = self.create_subscription(Int32MultiArray, '/camera/crate_pantry_blue', self.pantry_blue_callback, 10)
        # self.pantry_yellow_sub = self.create_subscription(Int32MultiArray, '/camera/crate_pantry_yellow', self.pantry_yellow_callback, 10)
        # self.crate_nest_sub = self.create_subscription(Int32, '/camera/nest_crate', self.crate_nest_callback, 10)
        # self.crate_collect_sub = self.create_subscription(Int32MultiArray, '/camera/collect_area_crate', self.crate_collect_callback, 10)
        # self.cursor_pose_sub = self.create_subscription(PoseStamped, '/camera/cursor_pose', self.cursor_pose_callback, 10)
        # declare action
        # declare tf listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # start main program
        self.clock = self.create_timer(self.timer_period, self.time_publisher)
        self.monitor_timer = self.create_timer(0.5, self.get_model_action, callback_group=self.monitor_group)
    def get_model_action(self):
        """
        [執行緒 A]：不斷進行 RL 推論，並判斷是否需要中斷
        """
        self.robot_pose = self.get_robot_pose('/robot/base_link')
        self.rival_pose = self.get_robot_pose('/rival/base_link')
        norm_obs = self.get_norm_obs()
        new_action, _ = self.model.predict(norm_obs, deterministic=True)

        with self.data_lock:
            if self.is_busy:
                # 檢查「連續三次不同」邏輯
                if new_action != self.current_action:
                    self.change_counter += 1
                    if self.change_counter >= 3:
                        self.get_logger().warn("檢測到目標變更，準備中斷任務...")
                        self.trigger_preemption(new_action)
                else:
                    self.change_counter = 0
            else:
                # 閒置狀態，啟動新任務
                self.start_execution_thread(new_action)
    def execute_model_action(self, action):
        self.get_logger().info(f"Idle. Starting Action {action}")
        with self.data_lock:
            self.is_busy = True
            self.current_action = action
            self.change_counter = 0
        self.start_waiting = self.elapsed_seconds
        target_coords, side = self.get_best_entry_point(self.robot_pose.pose.position.x, self.robot_pose.pose.position.y, action)
        if action < 8:
            if self.crate_robot[0] < self.crate_robot[1]:
                load_side = 0
            else:
                load_side = 1
            self.current_task = None
            MissionClient.send_mission_id(f'take_prepare_{load_side}')
            self.current_task = self.navigate
            self.current_task.execute(target_coords, load_side, finished_cb=self.handle_task_event)
            self.current_task = self.dock
            self.current_task.execute(side, 0.07, finished_cb=self.handle_task_event)
            self.current_task = None
            MissionClient.send_mission_id(f'take_{load_side}')
            self.crate_collect[action] -= min(4 - self.crate_robot[load_side], self.crate_collect[action])
            if self.crate_collect[action] != 0 and self.crate_robot[0]+self.crate_robot[1] < self.max_robot_capacity:
                load_side = abs(load_side-1)
                MissionClient.send_mission_id(f'take_prepare_{load_side}')
                self.current_task = self.back_rotate_forward
                self.current_task.execute(side, 0.07, finished_cb=self.handle_task_event)
                self.current_task = None
                MissionClient.send_mission_id(f'take_{load_side}')
                self.crate_collect[action] -= min(4 - self.crate_robot[load_side], self.crate_collect[action])
        elif action < 18:
            need_flip = False
            need_put = False
            if not self.crate_pantry_blue[action - 8] == 0:
                MissionClient.send_mission_id(f'flip_prepare')
                need_flip = True
            if self.crate_robot[0]+self.crate_robot[1] != 0 and self.crate_pantry_blue[action - 8]+self.crate_pantry_yellow[action - 8] < self.max_pantry_capacity:
                MissionClient.send_mission_id(f'put_prepare_{release_side}')
                if not self.crate_robot[0] == 0:
                    release_side = 0
                elif not self.crate_robot[1] == 0:
                    release_side = 1
                need_put = True
            self.current_task = self.navigate
            self.current_task.execute(target_coords, release_side, finished_cb=self.handle_task_event)
            self.current_task = self.dock
            self.current_task.execute(side, 0.07, finished_cb=self.handle_task_event)
            self.current_task = None
            if need_flip:
                MissionClient.send_mission_id('flip')
                self.crate_pantry_yellow[action - 8] += self.crate_pantry_blue[action - 8]
                self.crate_pantry_blue[action - 8] = 0
            if need_put:
                MissionClient.send_mission_id(f'put_{release_side}')
                self.crate_pantry_yellow[action - 8] += min(self.max_pantry_capacity-self.crate_pantry_blue[action - 8]-self.crate_pantry_yellow[action - 8], self.crate_robot[release_side])
                release_side = abs(release_side-1)
                if self.crate_robot[release_side] != 0 and self.crate_pantry_blue[action - 8]+self.crate_pantry_yellow[action - 8] < self.max_pantry_capacity:
                    MissionClient.send_mission_id(f'put_prepare_{release_side}')
                    self.current_task = self.back_rotate_forward
                    self.current_task.execute(side, 0.07, finished_cb=self.handle_task_event)
                    self.current_task = None
                    MissionClient.send_mission_id(f'put_{release_side}')
                    self.crate_pantry_yellow[action - 8] += min(self.max_pantry_capacity-self.crate_pantry_blue[action - 8]-self.crate_pantry_yellow[action - 8], self.crate_robot[release_side])

        crate_robot_msg = Int32()
        crate_robot_msg.data = self.crate_robot[0] + self.crate_robot[1]
        self.crate_robot_pub.publish(crate_robot_msg)

        self.waiting_time = self.elapsed_seconds - self.start_waiting
        self.is_busy = False
    def trigger_preemption(self, next_action):
        """
        [執行緒 A - Monitor] 觸發搶佔邏輯
        """
        with self.data_lock:
            self.get_logger().warn(f"正在搶佔任務：準備切換至 Action {next_action}")
            # 1. 檢查目前是否有正在執行的 Action Handle
            # 注意：你需要維護 self.active_goal_handle，在發送 goal 成功後存入
            if self.current_task:
                # 2. 發送非同步取消請求
                self.current_task.cancel(
                    cancelled_cb=lambda: self.execute_model_action(next_action)
                )
    # def pantry_blue_callback(self, msg):
    #     self.crate_pantry_blue = list(msg.data)
    # def pantry_yellow_callback(self, msg):
    #     self.crate_pantry_yellow = list(msg.data)
    # def crate_nest_callback(self, msg):
    #     self.crate_nest = int(msg.data)
    # def crate_collect_callback(self, msg):
    #     self.crate_collect = list(msg.data)
    # def cursor_pose_callback(self, msg):
    #     self.cursor_pose = msg.data
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
            y_norm = (self.crate_pantry_yellow[i] / self.max_pantry_capacity) * 2 - 1
            pantry_info.append(y_norm)
            # 藍色歸一化
            b_norm = (self.crate_pantry_blue[i] / self.max_pantry_capacity) * 2 - 1
            pantry_info.append(b_norm)

        # --- 5. 機上物品數量 (1維) ---
        held_val = ((self.crate_robot[0]+self.crate_robot[1]) / self.max_robot_capacity) * 2 - 1
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
def get_best_entry_point(self, robot_x, robot_y, action_id):
        """
        根據機器人目前位置與目標中心點的角度，選擇四個邊中最適合的進入點
        """
        
        # 獲取該區域中心點
        center = self.get_parameter(f'{prefix}.center').value
        
        # 計算機器人相對於中心點的角度 (弧度)
        # atan2(dy, dx)
        dx = robot_x - center[0]
        dy = robot_y - center[1]
        angle = math.atan2(dy, dx) # 範圍 [-pi, pi]
        
        # 將弧度轉為角度以便判斷
        deg = math.degrees(angle)

        # 判斷象限 (選擇最靠近機器人那一側的進入點)
        # 注意：這是基於機器人相對於目標的位置
        if action_id == 2 or action_id == 3:
            prefix = f'goal_areas.collect_{action_id}'
            if 0 <= deg <= 180:
                side = 'north'
            elif -180 <= deg <= 0:
                side = 'south'
            target_point = self.get_parameter(f'{prefix}.{side}').value
        elif action_id < 8:
            prefix = f'goal_areas.collect_{action_id}'
            side = self.get_parameter(f'{prefix}.side').value
            target_point = self.get_parameter(f'{prefix}.pose').value
        elif action_id == 11 or action_id == 12 or action_id == 13:
            prefix = f'goal_areas.pantry_{action_id - 8}'
            if -45 <= deg <= 45:
                side = 'east'   # 機器人在目標右側
            elif 45 < deg <= 135:
                side = 'north'  # 機器人在目標上方
            elif -135 <= deg < -45:
                side = 'south'  # 機器人在目標下方
            else:
                side = 'west'   # 機器人在目標左側 (包含 >135 或 <-135)
        else:
            prefix = f'goal_areas.pantry_{action_id - 8}'
            side = self.get_parameter(f'{prefix}.side').value
            target_point = self.get_parameter(f'{prefix}.pose').value
        return target_point, side

def main(args=None):
    rclpy.init(args=args)
    node = MainClient()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()