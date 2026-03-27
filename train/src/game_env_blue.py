import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class RobotMatchEnv(gym.Env):
    # define render params
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = 100 * self.metadata["render_fps"]
        self.current_step = 0
        # -------- params about game field ------
        # field size
        self.field_height = 2.0
        self.field_width = 3.0
        # params about collection area
        self.max_collect_capacity = 4.0
        self.collect_pos = np.array([
            [0.175, 1.2], [2.875, 1.2], [1.15, 0.8], [1.85, 0.8], 
            [0.175, 0.4], [2.875, 0.4], [1.1, 0.2], [1.9, 0.2]])
        self.collect_sizes = np.array([
            [0.15, 0.2], [0.15, 0.2], [0.2, 0.15], [0.2, 0.15],
            [0.15, 0.2], [0.15, 0.2], [0.2, 0.15], [0.2, 0.15]])
        self.collet_rects_min = self.collect_pos - (self.collect_sizes / 2.0)
        self.collect_rects_max = self.collect_pos + (self.collect_sizes / 2.0)
        # params about pantry area
        self.max_pantry_capacity = 5.0
        self.pantry_pos = np.array([
            [1.25, 1.45], [1.75, 1.45], [0.1, 0.8], [0.8, 0.8], [1.5, 0.8], 
            [2.2, 0.8],   [2.9, 0.8],   [0.7, 0.1], [1.5, 0.1], [2.3, 0.1]])
        self.pantry_size = 0.2
        self.pantry_rects_min = self.pantry_pos - (self.pantry_size / 2.0)
        self.pantry_rects_max = self.pantry_pos + (self.pantry_size / 2.0)
        # define entry points of each collection area and pantry
        self.entry_points_config = {
            # pantry
            0: {1: (1.25, 1.15)},
            1: {1: (1.75, 1.15)}, 
            2: {1: (0.4, 0.8)},
            3: {0: (0.8, 1.1), 1: (0.8, 0.5), 2: (1.1, 0.8), 3: (0.8, 0.5)},
            4: {0: (1.5, 1.1), 1: (1.5, 0.5), 2: (1.8, 0.8), 3: (1.2, 0.8)},
            5: {0: (2.2, 1.1), 1: (2.2, 0.5), 2: (2.5, 0.8), 3: (1.9, 0.8)},
            6: {3: (2.6, 0.8)}, 
            7: {0: (0.7, 0.4)}, 
            8: {0: (1.5, 0.4)}, 
            9: {0: (2.3, 0.4)},
            # collect
            10: {2: (0.45, 1.2)}, 
            11: {3: (2.55, 1.2)}, 
            12: {0: (1.15, 1.075), 1: (1.15, 0.525)},
            13: {0: (1.85, 1.075), 1: (1.85, 0.525)}, 
            14: {2: (0.45, 0.4)}, 
            15: {3: (2.55, 0.4)},
            16: {0: (1.1, 0.45)}, 
            17: {0: (1.9, 0.45)}
        }
        # -------- params about robot ---------
        self.max_dir_capacity = 4.0
        self.max_robot_total_capacity = 16.0
        self.max_speed = 0.8
        self.dt = 1/30
        self.robot_radius = 0.15
        # time spent for every mission
        self.collect_time_steps = 30 * 5
        self.place_time_steps = 30 * 3
        self.flip_time_steps = 30 * 2
        # ------- params about model -------
        # model action space
        self.action_space = spaces.Discrete(18)
        # model observation space
        # 40 dimention: (Robot:2, Enemy:2, WaitTimer:1, Collection:8, Pantry:10*2, Held:4, Stage:1, Area:1, LastArea:1, Time:1)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(41,), dtype=np.float32
        )
        self.reset()

    def _normalize_pos(self, pos):
        return [(pos[0] / self.field_width) * 2 - 1, (pos[1] / self.field_height) * 2 - 1]

    def _get_obs(self):
        robot_norm = self._normalize_pos(self.robot_pos) # robot pose (x, y)
        enemy_norm = self._normalize_pos(self.enemy_pos) # enemy pose (x, y)
        max_steps = max(self.collect_time_steps, self.place_time_steps)
        wait_timer_norm = np.array([(self.wait_timer / (max_steps * 2)) * 2 - 1], dtype=np.float32) # wait timer
        collection_info = [(c / self.max_collect_capacity) * 2 - 1 for c in self.collect_counts] # 8 collection
        pantry_info = [] # 10 blue pantries + 10 yellow pantries
        for i in range(10):
            pantry_info.append((self.pantry_yellow_counts[i] / self.max_pantry_capacity) * 2 - 1)
            pantry_info.append((self.pantry_blue_counts[i] / self.max_pantry_capacity) * 2 - 1)
        held_info = [(h / self.max_dir_capacity) * 2 - 1 for h in self.held_count] # crate held on robot
        time_progress = (self.current_step / self.max_steps) # game timer
        time_progress_norm = np.array([time_progress * 2 - 1], dtype=np.float32)
        # model stages and select area
        stage_norm = np.array([(self.stage / 2.0) * 2 - 1], dtype=np.float32)
        area_norm = np.array([(self.selected_area / 17.0) * 2 - 1 if self.selected_area != -1 else -1.0], dtype=np.float32)
        last_area_norm = np.array([(self.last_area_id / 17.0) * 2 - 1], dtype=np.float32)
        
        # 41 dimention in total: Robot(2), Enemy(2), WaitTimer(1), Collect(8), Pantry(20), Held(4), Stage(1), Area(1), LastArea(1), Time(1)
        obs = np.concatenate([
            robot_norm, enemy_norm, wait_timer_norm, collection_info, pantry_info, 
            held_info, stage_norm, area_norm, last_area_norm, time_progress_norm
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0 # init game timer
        self.robot_pos = np.array([2.7, 1.8])
        self.enemy_pos = np.array([0.3, 1.8])
        self.collect_counts = np.full(8, 4) # init 8 collection area with 4 crates
        self.pantry_yellow_counts = np.zeros(10) # init pantry to 0 crates
        self.pantry_blue_counts = np.zeros(10) # init pantry to 0 crates
        self.held_count = np.zeros(4, dtype=int) # use list to store crates on robot number
        self.wait_timer = 0 
        self.stage = 0  # 0:select area, 1:select entry point, 2:execute mission
        self.selected_area = -1
        self.last_area_id = -1
        self.selected_entry_id = -1
        self.target_pos = None
        self.prev_dist = np.zeros(1) # init previous dist of nav as 0
        self.action = 0    
        self.reward = 0.0
        self.fig = None # init render figure
        return self._get_obs(), {}

    def action_masks(self):
        """
        generate action masks for current stage
        return a boolean list, len=action_space=18
        True: valid action, False: invalid action
        """
        # init all action masks as false
        mask = np.zeros(self.action_space.n, dtype=bool)
        # Stage 0: select area
        if self.stage == 0:
            mask[0:18] = True # all 18 masks are valid
        # Stage 1: select entry point
        elif self.stage == 1:
            # determine how many entry point for this area 
            area_id = self.selected_area
            if area_id in self.entry_points_config:
                # only open id that really has entry point
                available_entries = self.entry_points_config[area_id].keys()
                for entry_id in available_entries:
                    # check entry point id < action_space number
                    if entry_id < self.action_space.n:
                        mask[entry_id] = True
            else:
                # if there's something wrong. need to open 1 mask at least
                mask[0] = True
        # Stage 2: execute mission
        elif self.stage == 2:
            area_id = self.selected_area
            if area_id >= 10: # condition 0：collection area (10-17)
                # action 0, 1, 2, 3: collect crate to robot from 4 direction
                # action 4: return to stage 0
                mask[0:5] = True
            else: # condition 1：pantry (0-9)
                # action 0: flip
                # action 1, 2, 3, 4: place crate with 4 direction of robot
                # action 5: return to stage 0
                mask[0:6] = True
        return mask

    def step(self, action):
        action = int(action)
        self.current_step += 1
        self.action = action
        reward = -0.005
        terminated = False
        truncated = False
        current_mask = self.action_masks()

        if not current_mask[action]: # make sure model doesn't choose invalid action mask. (Impossible)
            reward -= 1.0 # penalty
            # return. let model choose again
            self.reward = reward
            return self._get_obs(), reward, False, False, {"error": "Action Mask Violation"}

        # ------ Stage 0: select area ------
        if self.stage == 0:
            self.selected_area = action
            self.stage = 1
            self.reward = reward
            return self._get_obs(), reward, False, False, {}

        # ------ Stage 1: select entry point and enable navigation ------
        elif self.stage == 1:
            area_id = self.selected_area
            self.selected_entry_id = action
            self.target_pos = np.array(self.entry_points_config[area_id][action])
            # need to let navigation program run before return

        # ------ Stage 2: 執行任務決策 ------
        elif self.stage == 2:
            area_idx = self.selected_area
            is_pantry = area_idx < 10
            
            # 定義離開動作 (Pantry 是 5, Collection 是 4)
            exit_action = 5 if is_pantry else 4
            
            if action == exit_action:
                # 選擇離開：重置狀態回到 Stage 0
                self.stage = 0
                self.last_area_id = self.selected_area
                self.selected_area = -1
                reward += 0.2  # 正確選擇離開的微量獎勵
                return self._get_obs(), reward, terminated, truncated, {"action": "exit_area"}
            
            # --- 執行具體動作邏輯 ---
            if not is_pantry: # Collection Area
                robot_dir = action  # 0-3
                can_collect = self.collect_counts[area_idx-10] > 0 and self.held_count[robot_dir] < self.max_dir_capacity
                
                if can_collect:
                    count = min(self.collect_counts[area_idx-10], 
                                self.max_dir_capacity - self.held_count[robot_dir])
                    self.held_count[robot_dir] += count
                    self.collect_counts[area_idx-10] -= count
                    reward += 1.5 * count
                    self.wait_timer = self.collect_time_steps
                else:
                    reward -= 0.2 # 滿載或區空了還想採集的懲罰
            
            else: # Pantry Area
                if action == 0: # Flip
                    if self.pantry_yellow_counts[area_idx] > 0:
                        reward += 4.0 * self.pantry_yellow_counts[area_idx]
                        self.pantry_blue_counts[area_idx] += self.pantry_yellow_counts[area_idx]
                        self.pantry_yellow_counts[area_idx] = 0
                        self.wait_timer = self.flip_time_steps
                    else:
                        reward -= 0.3
                else: # Place (1-4 -> dir 0-3)
                    dir_idx = action - 1
                    capacity_left = self.max_pantry_capacity - (self.pantry_blue_counts[area_idx] + self.pantry_yellow_counts[area_idx])
                    if self.held_count[dir_idx] > 0 and capacity_left > 0:
                        count = min(self.held_count[dir_idx], capacity_left)
                        self.held_count[dir_idx] -= count
                        self.pantry_blue_counts[area_idx] += count
                        reward += 4.5 * count
                        self.wait_timer = self.place_time_steps
                    else:
                        reward -= 0.3

            # 重要：執行完動作後，保持在 Stage 2，讓模型決定下一步 (繼續執行或離開)
            self.reward = reward
            return self._get_obs(), reward, terminated, truncated, {"action_done": True}

        # ------ simple navigation program (activate after stage 1) ------
        if self.target_pos is not None:
            move_dir = self.target_pos - self.robot_pos
            dist = np.linalg.norm(move_dir)
            if dist > (self.max_speed * self.dt):
                self.robot_pos += (move_dir / dist) * (self.max_speed * self.dt)
            else:
                self.robot_pos = self.target_pos.copy()
                self.target_pos = None
                self.stage = 2 # turn to stage 2 and wait for next model action
                return self._get_obs(), reward + 0.5, False, False, {"arrived": True}
        
            # movement reward
            dist_to_goal = np.linalg.norm(self.robot_pos - (self.target_pos if self.target_pos is not None else self.robot_pos))
            reward += (self.prev_dist[0] - dist_to_goal) * 0.5
            self.prev_dist[0] = dist_to_goal

        # time reward
        time_ratio = self.current_step / self.max_steps
        if time_ratio > 0.9:  # if game time is running out
            # place all crates on robot as soon as possible
            if np.sum(self.held_count) > 0:
                reward -= 0.01 * np.sum(self.held_count)

        truncated = self.current_step >= self.max_steps
        if np.sum(self.pantry_blue_counts) > 16: terminated = True; reward += 50.0

        self.reward = reward
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human": return
        
        # ------ initialize ------
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.ax.set_xlim(-0.1, 3.1)
            self.ax.set_ylim(-0.1, 2.1)
            self.ax.set_aspect('equal')
            self.collect_patches = []
            self.pantry_yellow_patches = []
            self.pantry_blue_patches = []

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

            self.robot_patch = patches.Circle(self.robot_pos, self.robot_radius, color='white', ec='black', lw=2, zorder=10)
            self.ax.add_patch(self.robot_patch)
            # 0:(45-135 deg), 1:(225-315 deg), 2:(135-225 deg), 3:(315-45 deg)
            self.cargo_wedges = []
            angles = [(45, 135), (225, 315), (135, 225), (315, 405)] 
            colors = ["#EC6648", '#33FF57', '#3357FF', '#F333FF']
            
            for i in range(4):
                wedge = patches.Wedge(self.robot_pos, 0, angles[i][0], angles[i][1], 
                                      color=colors[i], alpha=0.7, zorder=11)
                self.cargo_wedges.append(wedge)
                self.ax.add_patch(wedge)
        # ------ update ------
        stage_names = ["Select Area", "Select Entry", "Execute Action"]
        self.ax.set_title(f"Stage: {stage_names[self.stage]} | Last Action: {self.action} | Reward: {self.reward:.3f}")

        for c in range(8):
            new_width = self.collect_sizes[c][0] * (self.collect_counts[c] / self.max_collect_capacity)
            self.collect_patches[c].set_width(new_width)
        for p in range(10):
            y_w = self.pantry_size * (self.pantry_yellow_counts[p] / self.max_pantry_capacity)
            self.pantry_yellow_patches[p].set_width(y_w)
            b_w = self.pantry_size * (self.pantry_blue_counts[p] / self.max_pantry_capacity)
            self.pantry_blue_patches[p].set_width(b_w)
            self.pantry_blue_patches[p].set_xy(self.pantry_rects_min[p] + np.array([y_w, 0]))

        self.robot_patch.center = self.robot_pos
        for i in range(4):
            self.cargo_wedges[i].set_center(self.robot_pos)
            dynamic_radius = self.robot_radius * (self.held_count[i] / self.max_dir_capacity)
            self.cargo_wedges[i].set_radius(dynamic_radius)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.00001)

class CollisionManager():
    @staticmethod
    def check_circle_collision(pos1, radius1, pos2, radius2):
        # robot vs robot
        distance = np.linalg.norm(pos1 - pos2)
        return distance <= (radius1 + radius2)

    @staticmethod
    def get_closest_point_on_rect(circle_pos, rect_min, rect_max):
        return np.clip(circle_pos, rect_min, rect_max)

    @staticmethod
    def check_circle_rect_collision(circle_pos, circle_radius, rect_min, rect_max):
        # robot vs collection area, pantry, nest
        closest_point = CollisionManager.get_closest_point_on_rect(circle_pos, rect_min, rect_max)
        distance = np.linalg.norm(circle_pos - closest_point)
        return distance <= circle_radius
    
    @staticmethod
    def handle_boundary_collision(pos, velocity, width, height, radius):
        new_pos = np.copy(pos)
        new_vel = np.copy(velocity)
        
        # check x-axis
        if new_pos[0] - radius < 0:
            new_pos[0] = radius
            new_vel[0] *= -0.5
        elif new_pos[0] + radius > width:
            new_pos[0] = width - radius
            new_vel[0] *= -0.5
            
        # check y-axis
        if new_pos[1] - radius < 0:
            new_pos[1] = radius
            new_vel[1] *= -0.5
        elif new_pos[1] + radius > height:
            new_pos[1] = height - radius
            new_vel[1] *= -0.5
        return new_pos, new_vel
    