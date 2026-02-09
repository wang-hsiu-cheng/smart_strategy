import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class RobotMatchEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = 100 * self.metadata["render_fps"]  # 100秒
        self.current_step = 0

        # setup constant parameter
        # --- game filed ---
        self.field_height = 2.0
        self.field_width = 3.0
        # ---- robot ----
        self.max_robot_capacity = 8.0
        self.max_speed = 0.8    # m/s
        self.dt = 1/30          # 假設 30 FPS
        self.robot_radius = 0.15
        self.collect_time_steps = 30 * 5
        self.place_time_steps = 30 * 3
        self.flip_time_steps = 30 * 2

        # ---- collect area ----
        self.max_collect_capacity = 4.0
        self.collect_pos = np.array([
            [0.2, 1.2], [2.8, 1.2], [1.15, 0.8], [1.85, 0.8],
            [0.2, 0.4], [2.8, 0.4], [1.1, 0.2], [1.9, 0.2]
        ])
        self.collect_sizes = np.array([
            [0.15, 0.2], [0.15, 0.2], [0.2, 0.15], [0.2, 0.15],
            [0.15, 0.2], [0.15, 0.2], [0.2, 0.15], [0.2, 0.15]
        ])
        half_size = self.collect_sizes / 2.0
        self.collet_rects_min = self.collect_pos - half_size
        self.collect_rects_max = self.collect_pos + half_size

        # ---- pantry ----
        self.max_pantry_capacity = 5.0
        self.pantry_pos = np.array([
            [1.25, 1.45], [1.75, 1.45], [0.1, 0.8], [0.8, 0.8], [1.5, 0.8],
            [2.2, 0.8], [2.9, 0.8], [0.7, 0.1], [1.5, 0.1], [2.3, 0.1]
        ])
        self.pantry_size = 0.2
        half_size = self.pantry_size / 2.0
        self.pantry_rects_min = self.pantry_pos - half_size
        self.pantry_rects_max = self.pantry_pos + half_size

        # --- action and observasion space ---
        # action: 8(collect) + 10(place) = 18 discrete target
        self.action_space = spaces.Discrete(18)
        # observasion: 34維 (Robot:2, Enemy:2, Timer:1, Collection:8, Pantry:10*2, Held:1)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(34,), dtype=np.float32
        )

        self.reset()

    def _normalize_pos(self, pos):
        """將 [0,3]x[0,2] 映射至 [-1,1]"""
        nx = (pos[0] / self.field_width) * 2 - 1
        ny = (pos[1] / self.field_height) * 2 - 1
        return [nx, ny]

    def _get_obs(self):
        # 機器人位置 (2維)
        robot_norm = self._normalize_pos(self.robot_pos)
        # 敵人位置 (2維)
        enemy_norm = self._normalize_pos(self.enemy_pos)

        # 等待計時器 (1維)
        max_steps = max(self.collect_time_steps, self.place_time_steps)
        wait_val = (self.wait_timer / (max_steps * 2)) * 2 - 1
        wait_timer_norm = np.array([wait_val], dtype=np.float32)

        # 蒐集區狀態 (8維): 數量 0~4 -> -1~1
        collect_info = [(c / self.max_collect_capacity) * 2 - 1 for c in self.collect_counts]

        # 放置區狀態 (20維):
        pantry_info = []
        for i in range(10):
            # 該區數量歸一化 (2維)
            yellow_count_norm = (self.pantry_yellow_counts[i] / self.max_pantry_capacity) * 2 - 1
            pantry_info.append(yellow_count_norm)
            blue_count_norm = (self.pantry_blue_counts[i] / self.max_pantry_capacity) * 2 - 1
            pantry_info.append(blue_count_norm)

        # 機上物品數量 (1維): 0->-1, 8->1
        held_val = (self.held_count / self.max_robot_capacity) * 2 - 1
        held_info = np.array([held_val], dtype=np.float32)

        # 合併為 34 維向量
        obs = np.concatenate([
            robot_norm, enemy_norm, wait_timer_norm, collect_info, pantry_info, held_info
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.robot_pos = np.array([0.3, 1.8])
        self.enemy_pos = np.array([2.7, 1.8])
        self.collect_counts = np.full(8, 4)
        self.pantry_yellow_counts = np.zeros(10)
        self.pantry_blue_counts = np.zeros(10)
        self.held_count = 0
        self.wait_timer = 0  # 初始化計時器
        initial_target = self.collect_pos[0]
        self.prev_dist = np.zeros(2)
        self.prev_dist[0] = np.linalg.norm(self.robot_pos - initial_target)
        
        self.fig = None # 重置渲染
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        terminated = False
        truncated = False
        reward = -0.005 # time cost
        old_pos = self.robot_pos.copy()

        if self.wait_timer > 0:
            self.wait_timer -= 1
            # 在等待期間，機器人不移動，直接回傳當前觀察值
            return self._get_obs(), reward, False, False, {}
        self.wait_timer = 0

        # convert action number into position
        if action < 8:
            if self.held_count == self.max_robot_capacity: # 拿滿了還想去拿，扣分
                reward -= 0.2
            if self.collect_counts[action] == 0:
                reward -= 0.5
            target = self.collect_pos[action]
        else:
            if self.held_count == 0 and self.pantry_blue_counts[action - 8] == 0: # 檢查是否有藍色可以翻，如果連藍色都沒有，空手去放置區就是浪費時間
                reward -= 0.5
            target = self.pantry_pos[action - 8]

        # robot move to the target in constant speed
        move_dir = target - self.robot_pos
        dist = np.linalg.norm(move_dir)
        step_dist = self.max_speed * self.dt
        if dist > step_dist:
            self.robot_pos += (move_dir / dist) * step_dist
        else:
            self.robot_pos = target
        
        # --- 4. 獎勵計算：決策一致性與引導獎勵 ---
        dist_to_goal = np.linalg.norm(self.robot_pos - target)
        
        # 確保 last_action 屬性存在
        if not hasattr(self, 'last_action'): 
            self.last_action = action

        if action != self.last_action:
            # (B) 決策一致性檢查：防止在不同放置點間抖動
            if action >= 8 and self.last_action >= 8:
                reward -= 0.1  # 針對 Pantry 間橫跳的重罰
            else:
                reward -= 0.02 # 一般切換輕微扣分
                
            # 換目標時重置距離基準，讓 diff = 0，避免產生錯誤的距離獎勵/懲罰
            self.prev_dist[0] = dist_to_goal 
            diff = 0
        else:
            # 目標一致時，計算靠近獎勵 (引導獎勵)
            diff = self.prev_dist[0] - dist_to_goal
            reward += diff * 0.5  # 鼓勵朝向目標移動
        
        # 更新舊距離
        self.prev_dist[0] = dist_to_goal
        self.last_action = action

        # --- 5. 獎勵計算：物理位移檢查 (靜態懲罰) ---
        dist_moved = np.linalg.norm(self.robot_pos - old_pos)
        # 判定門檻：至少要達到理論移動速度的一半
        actual_speed_threshold = (self.max_speed * self.dt) * 0.5 
        
        if dist_moved < actual_speed_threshold:
            # 根據是否持物加重處罰 (拿著東西發呆最傷)
            idle_penalty = -0.05 if self.held_count > 0 else -0.02
            reward += idle_penalty
        
        # if robot pass by the area not the target
        for c in range(8):
            if CollisionManager.check_circle_rect_collision(
                self.robot_pos, self.robot_radius, self.collet_rects_min[c], self.collect_rects_max[c]
            ) and c != action:
                reward -= 0.06
        for p in range(10):
            if CollisionManager.check_circle_rect_collision(
                self.robot_pos, self.robot_radius, self.pantry_rects_min[p], self.pantry_rects_max[p]
            ) and p+8 != action:
                reward -= 0.06

        # if robot reach the target
        if dist < 0.05:
            if action < 8: # collect
                if self.collect_counts[action] > 0 and self.held_count != 8:
                    self.wait_timer = self.collect_time_steps
                    collect_count = min(self.collect_counts[action], self.max_robot_capacity - self.held_count)
                    self.collect_counts[action] -= collect_count
                    self.held_count += collect_count
                    reward += 1.0 * collect_count
                    return self._get_obs(), reward, False, False, {}
            else: # place
                if (self.pantry_yellow_counts[action - 8] + self.pantry_blue_counts[action - 8]) < self.max_pantry_capacity and self.held_count > 0:
                    self.wait_timer += self.place_time_steps
                    place_count = min(self.max_pantry_capacity - (self.pantry_yellow_counts[action - 8] + self.pantry_blue_counts[action - 8]), self.held_count)
                    self.pantry_yellow_counts[action - 8] += place_count
                    self.held_count -= place_count
                    reward += 4.0 * place_count
                if self.pantry_blue_counts[action - 8] <= 0:
                    return self._get_obs(), reward, False, False, {}
                else:
                    self.wait_timer = self.flip_time_steps
                    self.pantry_yellow_counts[action - 8] += self.pantry_blue_counts[action - 8]
                    reward += 4.0 * self.pantry_blue_counts[action - 8]
                    self.pantry_blue_counts[action - 8] = 0
                    return self._get_obs(), reward, False, False, {}

        # 結束條件
        truncated = self.current_step >= self.max_steps

        if np.sum(self.pantry_yellow_counts) > 16:
            terminated = True
            reward += 50.0

        self.action = action
        self.reward = reward

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human": return
        
        # 1. 初始化階段：建立畫布與所有 Patch
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(7, 5))
            self.ax.set_xlim(-0.1, 3.1)
            self.ax.set_ylim(-0.1, 2.1)
            self.ax.set_aspect('equal')
            self.score_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            # 儲存 patch 的清單，以便後續更新
            self.collect_patches = []
            self.pantry_yellow_patches = []
            self.pantry_blue_patches = []

            # 畫出 8 個蒐集區 (背景 + 填充層)
            for c in range(8):
                # 框線背景
                self.ax.add_patch(patches.Rectangle(self.collet_rects_min[c], self.collect_sizes[c][0], self.collect_sizes[c][1], color='gray', fill=False))
                # 實際數量填充層 (寬度會隨數量變化)
                rect = patches.Rectangle(self.collet_rects_min[c], 0, self.collect_sizes[c][1], color='gray', alpha=0.5)
                self.collect_patches.append(rect)
                self.ax.add_patch(rect)

            # 畫出 10 個放置區
            for p in range(10):
                # 框線背景
                self.ax.add_patch(patches.Rectangle(self.pantry_rects_min[p], self.pantry_size, self.pantry_size, color='forestgreen', fill=False))
                # 黃色填充層
                y_rect = patches.Rectangle(self.pantry_rects_min[p], 0, self.pantry_size, color='yellow', alpha=0.8)
                self.pantry_yellow_patches.append(y_rect)
                self.ax.add_patch(y_rect)
                # 藍色填充層
                b_rect = patches.Rectangle(self.pantry_rects_min[p], 0, self.pantry_size, color='blue', alpha=0.8)
                self.pantry_blue_patches.append(b_rect)
                self.ax.add_patch(b_rect)

            # 機器人基礎底層 (圓形)
            self.robot_patch = patches.Circle(self.robot_pos, self.robot_radius, color='lightgray', ec='black', zorder=10)
            # 載物進度條 (扇形)
            self.cargo_wedge = patches.Wedge(self.robot_pos, self.robot_radius, 0, 0, color='orange', zorder=11)
            
            self.ax.add_patch(self.robot_patch)
            self.ax.add_patch(self.cargo_wedge)

        # 2. 更新階段：僅修改現有 Patch 的數據，不建立新物件
        
        self.ax.set_title(f"Action: {self.action} | Reward: {self.reward:.3f}")
        # self.score_text.set_text(f"Total Reward: {self.total_reward:.2f}")
        # 更新蒐集區數量顯示
        for c in range(8):
            new_width = self.collect_sizes[c][0] * (self.collect_counts[c] / self.max_collect_capacity)
            self.collect_patches[c].set_width(new_width)

        # 更新放置區數量顯示 (修正了你原本的索引 c 錯誤)
        for p in range(10):
            # 黃色部分
            y_w = self.pantry_size * (self.pantry_yellow_counts[p] / self.max_pantry_capacity)
            self.pantry_yellow_patches[p].set_width(y_w)
            
            # 藍色部分 (起點需位移，接在黃色後面)
            b_w = self.pantry_size * (self.pantry_blue_counts[p] / self.max_pantry_capacity)
            self.pantry_blue_patches[p].set_width(b_w)
            new_xy = self.pantry_rects_min[p] + np.array([y_w, 0])
            self.pantry_blue_patches[p].set_xy(new_xy)

        # 更新機器人位置與載物狀態
        self.robot_patch.center = self.robot_pos
        self.cargo_wedge.set_center(self.robot_pos)
        # 根據載物量更新扇形角度 (0~360度)
        angle = (self.held_count / self.max_robot_capacity) * 360
        self.cargo_wedge.set_theta2(angle)

        # 重新整理畫布
        self.fig.canvas.draw_idle() # 比 draw() 更節省資源
        self.fig.canvas.flush_events()
        plt.pause(0.00001) # 縮短暫停時間增加流暢度

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
        """
        處理場地邊界碰撞：限制位置並反轉速度 (彈回效果)
        """
        new_pos = np.copy(pos)
        new_vel = np.copy(velocity)
        
        # X 軸檢查
        if new_pos[0] - radius < 0:
            new_pos[0] = radius
            new_vel[0] *= -0.5  # 撞牆後動能損耗
        elif new_pos[0] + radius > width:
            new_pos[0] = width - radius
            new_vel[0] *= -0.5
            
        # Y 軸檢查
        if new_pos[1] - radius < 0:
            new_pos[1] = radius
            new_vel[1] *= -0.5
        elif new_pos[1] + radius > height:
            new_pos[1] = height - radius
            new_vel[1] *= -0.5
            
        return new_pos, new_vel
    