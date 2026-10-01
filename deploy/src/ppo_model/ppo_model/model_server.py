import rclpy
from rclpy.node import Node
import tf2_ros
from rclpy.time import Time
from std_msgs.msg import Int32, Int64, Float32MultiArray, Int32MultiArray
from geometry_msgs.msg import Pose2D
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- use static onnx model ---
class RobotInference:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
    
    def get_action(self, obs_41dim, action_mask):
        obs_input = obs_41dim.reshape(1, 41).astype(np.float32) # set 41 dimension observation input
        logits = self.session.run(None, {self.input_name: obs_input})[0]
        # add mask to static model output. filter invalid action
        logits = logits.flatten()
        masked_logits = np.where(action_mask, logits, -1e10)
        return int(np.argmax(masked_logits))

# --- ROS2 ---
class ModelServerNode(Node):
    def __init__(self):
        super().__init__('model_server')
        # --- get params from YAML ---
        # set params
        self.declare_parameter('is_simulation', True)
        self.declare_parameter('onnx_model_path', '')
        self.declare_parameter('robot_color', '')
        self.declare_parameter('radius', 0)
        self.declare_parameter('max_dir_capacity', 0)
        self.declare_parameter('max_robot_total_capacity', 0)
        self.declare_parameter('yellow_start_pos', np.array([0.0, 0.0]))
        self.declare_parameter('blue_start_pos', np.array([0.0, 0.0]))
        self.declare_parameter('max_pantry_capacity', 0)
        self.declare_parameter('max_collect_capacity', 0)
        # init array variables
        self.robot_pos = [0.0, 0.0]
        self.enemy_pos = [0.0, 0.0]
        self.pantry_pos = np.zeros((10, 2))
        self.collect_pos = np.zeros((8, 2))
        self.collect_sizes = np.zeros((8, 2))
        # set and get array params
        for i in range(10):
            param_name = f'pantry_pos.area_{i}'
            self.declare_parameter(param_name, rclpy.Parameter.Type.DOUBLE_ARRAY)
            self.pantry_pos[i] = self.get_parameter(param_name).value
        for i in range(8):
            param_name = f'collect_pos.area_{i}'
            self.declare_parameter(param_name, rclpy.Parameter.Type.DOUBLE_ARRAY)
            self.collect_pos[i] = self.get_parameter(param_name).value
            param_name = f'collect_size.area_{i}'
            self.declare_parameter(param_name, rclpy.Parameter.Type.DOUBLE_ARRAY)
            self.collect_sizes[i] = self.get_parameter(param_name).value
        self.entry_points_config = {}
        self.load_entry_config()
        # get params
        self.is_simulation = self.get_parameter('is_simulation').value
        onnx_model_path = self.get_parameter('onnx_model_path').value
        self.my_color = self.get_parameter('robot_color').value
        self.radius = self.get_parameter('radius').value
        self.max_dir_capacity = self.get_parameter('max_dir_capacity').value
        self.max_robot_total_capacity = self.get_parameter('max_robot_total_capacity').value
        self.max_pantry_capacity = self.get_parameter('max_pantry_capacity').value
        self.max_collect_capacity = self.get_parameter('max_collect_capacity').value
        self.pantry_size = self.get_parameter('pantry_size').value
        if self.my_color == 'yellow':
            self.robot_pos = self.get_parameter('yellow_start_pos').value
            self.enemy_pos = self.get_parameter('blue_start_pos').value
        else:
            self.robot_pos = self.get_parameter('blue_start_pos').value
            self.enemy_pos = self.get_parameter('yellow_start_pos').value
        # compute needed information from params
        self.pantry_rects_min = self.pantry_pos - (self.pantry_size / 2.0)
        self.pantry_rects_max = self.pantry_pos + (self.pantry_size / 2.0)
        self.collet_rects_min = self.collect_pos - (self.collect_sizes / 2.0)
        self.collect_rects_max = self.collect_pos + (self.collect_sizes / 2.0)
        # init object variables
        self.seq_num = 0
        self.waiting_for_ack = False
        self.action_id = 0
        self.encoded_msg = 0
        self.max_steps = 3000
        self.infer = RobotInference(onnx_model_path) # get model engine
        self.stage = 0
        self.selected_area = -1
        self.last_area_id = -1
        # init callback variables
        self.collect_area_counts = np.zeros(8)
        self.pantry_yellow_counts = np.zeros(10)
        self.pantry_blue_counts = np.zeros(10)
        self.held_count = np.zeros(4)
        self.wait_timer = 0
        self.current_step = 0
        # ROS2 msg
        self.create_subscription(Int32MultiArray, '/field/status', self.field_status_cb, 10)
        self.action_pub = self.create_publisher(Int64, '/robot/action_cmd', 10)
        self.ack_sub = self.create_subscription(Int64, '/robot/action_ack', self.ack_callback, 10)
        # init variable about tf2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # define frame id
        self.map_frame = "map"
        self.robot_frame = "robot/base_link"
        self.rival_frame = "rival/base_link"
        # start core loop
        self.create_timer(1/30, self.control_loop)
        # init matplotlib figure
        self.fig = None

    def load_entry_config(self):
        for i in range(18):
            param_name = f'entry_config.area_{i}'
            self.declare_parameter(param_name, rclpy.Parameter.Type.DOUBLE_ARRAY)
            # get 1D array
            raw_list = self.get_parameter(param_name).value
            # organize to desired structure: {entry_id: (x, y)}
            area_dict = {}
            for j in range(0, len(raw_list), 3):
                entry_id = int(raw_list[j])
                x = raw_list[j+1]
                y = raw_list[j+2]
                area_dict[entry_id] = (x, y)
            self.entry_points_config[i] = area_dict

    def encode_msg(self, seq, stage, action):
        return int(seq * 1000 + stage * 100 + action)
    
    # --- callback function ---
    # update field info from model_client BT node
    def field_status_cb(self, msg):
        # [Collect * 8, YellowPantry * 10, BluePantry * 10, Held * 4, WaitTimer, Step]
        data = msg.data
        self.collect_area_counts = np.array(data[0:8])
        self.pantry_yellow_counts = np.array(data[8:18])
        self.pantry_blue_counts = np.array(data[18:28])
        self.held_count = np.array(data[28:32])
        self.wait_timer = data[32]
        self.current_step = data[33]

    # receive ack from model_client BT node
    def ack_callback(self, msg):
        val = msg.data
        ack_seq = val // 1000
        ack_stage = (val % 1000) // 100
        ack_action = val % 100

        # check if ack success: seq_num + 1
        if ack_seq == self.seq_num + 1:
            self.get_logger().info(f"ACK Received: Seq {ack_seq}, Action {ack_action}")
            # special case: if get action=18 => navigation finish successfully
            if ack_action == 18:
                self.stage = 2 # navigation finish. change to stage2
                self.get_logger().info("Navigation Arrived: Stage changed to 2")
            if not self.waiting_for_ack: return # if model_client haven't receive current action. don't update 
            self.update_internal_stage(self.action_id) # update stage
            # update seq_num and reset variables
            self.seq_num += 1
            self.waiting_for_ack = False
            self.encoded_msg = 0

    def update_poses_from_tf(self):
        try:
            now = rclpy.time.Time()
            t = self.tf_buffer.lookup_transform(self.map_frame, self.robot_frame, now)
            self.robot_pos = [t.transform.translation.x, t.transform.translation.y]
            if self.tf_buffer.can_transform(self.map_frame, self.rival_frame, now):
                t2 = self.tf_buffer.lookup_transform(self.map_frame, self.rival_frame, now)
                self.enemy_pos = [t2.transform.translation.x, t2.transform.translation.y]
        except Exception as e:
            pass

    # --- core logic：build 41 dimension vector and mask ---
    def control_loop(self):
        self.update_poses_from_tf()
        if self.waiting_for_ack: # model_client haven't receive current action
            # keep sending same action & same seq_num
            retry_msg = Int64(data=self.encoded_msg)
            self.action_pub.publish(retry_msg)
            # still need to update render
            self.update_render(self.action_idx)
            return

        # generate new action
        if self.wait_timer == 0: # robot is not waiting
            # build observation & mask
            obs = self.build_observation()
            mask = self.build_action_mask()
            # get acction from onnx static model
            action = self.infer.get_action(obs, mask)
            self.action_id = action
            # encode action and seq_num and send to model_client
            encoded_val = self.encode_msg(self.seq_num, self.stage, self.action_id)
            self.encoded_msg = encoded_val
            msg = Int64(data=self.encoded_msg)
            self.action_pub.publish(msg)
            self.waiting_for_ack = True # wait for model_client receive
            self.update_render(self.action_id) # update render

    def build_observation(self):
        def norm_position(p): return [(p[0]/3.0)*2-1, (p[1]/2.0)*2-1]
        p_info = []
        for i in range(10):
            p_info.extend([(self.pantry_yellow_counts[i]/5.0)*2-1, (self.pantry_blue_counts[i]/5.0)*2-1])
        # 41 dimention: (Robot:2, Enemy:2, WaitTimer:1, Collection:8, Pantry:10*2, Held:4, Stage:1, Area:1, LastArea:1, Time:1)
        obs = np.concatenate([
            norm_position(self.robot_pos), 
            norm_position(self.enemy_pos),
            [(self.wait_timer/100)*2-1],
            [(c/4.0)*2-1 for c in self.collect_area_counts],
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
            mask[17] = True
            if self.selected_area in self.entry_points_config:
                available_entries = self.entry_points_config[self.selected_area].keys()
                for entry_id in available_entries:
                    if entry_id < 18:
                        mask[entry_id] = True
            else: mask[0] = True
        elif self.stage == 2:
            if self.selected_area >= 10: mask[0:5] = True 
            else: mask[0:6] = True 
        return mask

    # --- stage transition: only update when model_client receive current action ---
    def update_internal_stage(self, action):
        if self.stage == 0:
            self.last_area_id = self.selected_area
            self.selected_area = action
            self.stage = 1
        elif self.stage == 1:
            if action == 17: # re-select new area
                self.stage = 0
            # transition after navigation: update in callback function ( when receive ack_action == 18)
        elif self.stage == 2:
            exit_act = 5 if self.selected_area < 10 else 4
            if action == exit_act: self.stage = 0

    def update_render(self, action):
        # ------ initial ax and static variable (run one time) ------
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 7))
            self.ax.set_xlim(-0.1, 3.1)
            self.ax.set_ylim(-0.1, 2.1)
            self.ax.set_aspect('equal')
            self.collect_patches = []
            self.pantry_yellow_patches = []
            self.pantry_blue_patches = []
            # draw crates in collection area and pantry
            for c in range(8):
                self.ax.add_patch(patches.Rectangle(self.collet_rects_min[c], self.collect_sizes[c][0], self.collect_sizes[c][1], color='gray', fill=False))
                rect = patches.Rectangle(self.collet_rects_min[c], 0, self.collect_sizes[c][1], color='gray', alpha=0.5)
                self.collect_patches.append(rect)
                self.ax.add_patch(rect)
            for p in range(10):
                self.ax.add_patch(patches.Rectangle(self.pantry_rects_min[p], self.pantry_size, self.pantry_size, color='forestgreen', fill=False))
                y_rect = patches.Rectangle(self.pantry_rects_min[p], 0, self.pantry_size, color='yellow', alpha=0.8)
                self.pantry_yellow_patches.append(y_rect)
                self.ax.add_patch(y_rect)
                b_rect = patches.Rectangle(self.pantry_rects_min[p], 0, self.pantry_size, color='blue', alpha=0.8)
                self.pantry_blue_patches.append(b_rect)
                self.ax.add_patch(b_rect)
            # function: draw robot and crate on robot
            def create_robot_visuals(color, ec):
                patch = patches.Circle(self.robot_pos, self.radius, color=color, ec=ec, lw=2, zorder=10)
                self.ax.add_patch(patch)
                wedges = []
                angles = [(45, 135), (225, 315), (135, 225), (315, 405)] 
                colors = ["#EEA695", "#A5DDAF", "#AFB8E3", "#E6AEEA"] # use 4 color to represent crates in 4 directions
                for i in range(4):
                    w = patches.Wedge(self.robot_pos, 0, angles[i][0], angles[i][1], color=colors[i], alpha=0.7, zorder=11)
                    wedges.append(w)
                    self.ax.add_patch(w)
                return patch, wedges
            # create my robot (white) and enemy robot (gray)
            self.my_patch, self.my_wedges = create_robot_visuals('white', 'black')
            self.enemy_patch = patches.Circle(self.enemy_pos, self.radius, color='#E0E0E0', ec='red', lw=2, zorder=10)
            self.ax.add_patch(self.enemy_patch)
        # ------ update everything keep changing ------
        stage_names = ["Select Area", "Select Entry", "Execute"]
        # update title: my info and enemy info
        my_info = f"MY - Stage: {stage_names[self.stage]} | Act: {action}"
        self.ax.set_title(f"{my_info}", fontsize=10)
        # update objects on map
        for c in range(8):
            new_width = self.collect_sizes[c][0] * (self.collect_area_counts[c] / self.max_collect_capacity)
            self.collect_patches[c].set_width(new_width)
        for p in range(10):
            y_w = self.pantry_size * (self.pantry_yellow_counts[p] / self.max_pantry_capacity)
            self.pantry_yellow_patches[p].set_width(y_w)
            b_w = self.pantry_size * (self.pantry_blue_counts[p] / self.max_pantry_capacity)
            self.pantry_blue_patches[p].set_width(b_w)
            self.pantry_blue_patches[p].set_xy(self.pantry_rects_min[p] + np.array([y_w, 0]))
        # function: update robot position and crates condition on robot
        def update_robot_visuals(patch, wedges):
            patch.center = self.robot_pos
            for i in range(4):
                wedges[i].set_center(self.robot_pos)
                dynamic_radius = self.radius * (self.held_count[i] / self.max_dir_capacity)
                wedges[i].set_radius(dynamic_radius)
        update_robot_visuals(self.my_patch, self.my_wedges)
        self.enemy_patch.center = self.enemy_pos
        # update canvas
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

def main():
    rclpy.init()
    node = ModelServerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()