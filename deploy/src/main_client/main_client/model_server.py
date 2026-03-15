import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float32MultiArray, Int32MultiArray
from geometry_msgs.msg import Pose2D
import onnxruntime as ort
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 推理引擎封裝 ---
class RobotInference:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
    
    def get_action(self, obs_41dim, action_mask):
        inputs = {self.session.get_inputs()[0].name: obs_41dim.reshape(1, 41).astype(np.float32)}
        logits = self.session.run(None, inputs)[0]
        # 套用遮罩：將不合法動作設為負無窮
        masked_logits = np.where(action_mask, logits, -1e10)
        return np.argmax(masked_logits)

# --- ROS 2 部署節點 ---
class ModelServerNode(Node):
    def __init__(self):
        super().__init__('model_server_node')
        
        # 1. 讀取機器人設定參數 (YAML)
        with open('robot_config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 2. 初始化推理與狀態
        self.infer = RobotInference("robot_policy.onnx")
        self.my_color = "yellow" # 可從參數讀取
        
        # 內部狀態快取 (Buffer)
        self.robot_pos = np.array([0.0, 0.0])
        self.enemy_pos = np.array([0.0, 0.0])
        self.wait_timer = 0
        self.collect_counts = np.zeros(8)
        self.pantry_yellow = np.zeros(10)
        self.pantry_blue = np.zeros(10)
        self.held_count = np.zeros(4)
        self.current_step = 0
        self.max_steps = 3000
        
        # RL 決策鏈狀態
        self.stage = 0
        self.selected_area = -1
        self.last_area_id = -1
        
        # 3. ROS 2 通訊設置
        # 訂閱來自 C++ 程式更新的狀態 (範例 Topic 名稱)
        self.create_subscription(Pose2D, '/robot/pose', self.pose_cb, 10)
        self.create_subscription(Pose2D, '/enemy/pose', self.enemy_cb, 10)
        self.create_subscription(Int32MultiArray, '/field/status', self.field_status_cb, 10)
        
        # 發佈決策動作
        self.action_pub = self.create_publisher(Int32, '/robot/action_cmd', 10)
        
        # 4. 定時器：30 FPS 決策循環
        self.create_timer(1/30, self.control_loop)
        
        # 可視化初始化
        self.init_render()

    # --- 數據回調函數 ---
    def pose_cb(self, msg): self.robot_pos = np.array([msg.x, msg.y])
    def enemy_cb(self, msg): self.enemy_pos = np.array([msg.x, msg.y])
    def field_status_cb(self, msg):
        # 假設資料格式：[8個Collect, 10個YellowPantry, 10個BluePantry, 4個Held, WaitTimer, Step]
        data = msg.data
        self.collect_counts = np.array(data[0:8])
        self.pantry_yellow = np.array(data[8:18])
        self.pantry_blue = np.array(data[18:28])
        self.held_count = np.array(data[28:32])
        self.wait_timer = data[32]
        self.current_step = data[33]

    # --- 核心邏輯：構建 41 維向量與 Mask ---
    def control_loop(self):
        # A. 構建 Observation (與 game_env.py 一致)
        obs = self.build_observation()
        
        # B. 構建 Action Mask
        mask = self.build_action_mask()
        
        # C. 模型推理
        if self.wait_timer == 0:
            action_idx = self.infer.get_action(obs, mask)
            
            # D. 內部 Stage 狀態機維護 (關鍵：需同步訓練時的邏輯)
            self.update_internal_stage(action_idx)
            
            # E. 獲取模型決策並傳送給 C++
            msg = Int32()
            msg.data = int(action_idx)
            self.action_pub.publish(msg)
        
        # F. 可視化
        self.update_render()

    def build_observation(self):
        # 實作 game_env.py 中的 _normalize_pos 與向量拼接
        def norm_p(p): return [(p[0]/3.0)*2-1, (p[1]/2.0)*2-1]
        
        p_info = []
        for i in range(10):
            p_info.extend([(self.pantry_yellow[i]/5.0)*2-1, (self.pantry_blue[i]/5.0)*2-1])
            
        obs = np.concatenate([
            norm_p(self.robot_pos), norm_p(self.enemy_pos),
            [(self.wait_timer/120)*2-1],
            [(c/4.0)*2-1 for c in self.collect_counts],
            p_info,
            [(h/4.0)*2-1 for h in self.held_count],
            [(self.stage/2.0)*2-1],
            [(self.selected_area/17.0)*2-1 if self.selected_area != -1 else -1.0],
            [(self.last_area_id/17.0)*2-1],
            [(self.current_step/self.max_steps)*2-1]
        ])
        return obs.astype(np.float32)

    def build_action_mask(self):
        mask = np.zeros(18, dtype=bool)
        if self.stage == 0:
            mask[:] = True
        elif self.stage == 1:
            mask[17] = True # Exit
            # 這裡應補齊 entry_points_config 的合法 id 判斷
        elif self.stage == 2:
            if self.selected_area >= 10: mask[0:5] = True
            else: mask[0:6] = True
        return mask

    def update_internal_stage(self, action):
        # 模擬 _robot_logic 中的 Stage 切換，以便下一幀計算 Mask
        if self.stage == 0:
            self.last_area_id = self.selected_area
            self.selected_area = action
            self.stage = 1
        elif self.stage == 1:
            if action == 17: self.stage = 0
            # 注意：這裡假設 C++ 程式會處理位移，並在抵達後透過 Topic 通知 Python 節點 Stage 變為 2
            # 或者 Python 端根據位置判斷自動進入 Stage 2
        elif self.stage == 2:
            exit_act = 5 if self.selected_area < 10 else 4
            if action == exit_act: self.stage = 0

    # --- Render 功能實作 ---
    def init_render(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        # 繪製靜態地圖邊界與區域 (參考 game_env.py)
        # ... (代碼略，與 game_env.py render 初期化一致)

    def update_render(self):
        # 更新機器人位置與文字資訊
        # ... (代碼略，與 game_env.py render 更新邏輯一致)
        plt.pause(0.001)

def main():
    rclpy.init()
    node = ModelServerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()