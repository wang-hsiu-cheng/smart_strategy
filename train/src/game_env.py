import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class RobotState:
    def __init__(self, start_pos, is_enemy=False):
        self.pos = np.array(start_pos, dtype=np.float32)
        self.prev_pos = np.array(start_pos, dtype=np.float32)
        self.color = None               # this robot's color
        self.stage = 0
        self.steps_in_stage = 0         # count the navigation steps
        self.last_stage = -1
        self.last_finished_area = -1    # area that robot really do sth last time
        self.selected_area = -1
        self.last_area_id = -1
        self.selected_entry_id = -1
        self.wait_timer = 0
        self.held_count = np.zeros(4, dtype=int)
        self.target_pos = None
        self.prev_dist = np.zeros(1)    # init previous dist of nav as 0
        self.action = 0
        self.need_leave = 0
        self.made_contribution = False  # flag: did robot really do sth in an area
        self.is_enemy = is_enemy

    def reset(self, start_pos, is_enemy=False):
        self.pos = np.array(start_pos, dtype=np.float32)
        self.prev_pos = np.array(start_pos, dtype=np.float32)
        self.stage = 0
        self.steps_in_stage = 0
        self.last_stage = -1
        self.last_finished_area = -1
        self.selected_area = -1
        self.last_area_id = -1
        self.selected_entry_id = -1
        self.wait_timer = 0
        self.held_count.fill(0)
        self.target_pos = None
        self.prev_dist = np.zeros(1)    # init previous dist of nav as 0
        self.action = 0
        self.need_leave = 0
        self.made_contribution = False  # init flag as false
        self.is_enemy = is_enemy

class RobotMatchEnv(gym.Env):
    # define matadata params
    metadata = {"my_color": ["yellow", "blue"], "render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(self, render_mode=None, my_color=None):
        super().__init__()
        self.render_mode = render_mode
        self.my_color = my_color
        self.max_steps = 100 * self.metadata["render_fps"]
        self.current_step = 0
        # -------- params about game field ------
        self.YELLOW_START_POSE = np.array([0.3, 1.8])
        self.BLUE_START_POSE = np.array([2.7, 1.8])
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
        # ------ params about robot ------
        # init 2 robots' color and start position
        if self.my_color == "yellow":
            self.my_robot = RobotState(self.YELLOW_START_POSE)
            self.enemy_robot = RobotState(self.BLUE_START_POSE, is_enemy=True)
            self.my_robot.color = "yellow"
            self.enemy_robot.color = "blue"
        else:
            self.my_robot = RobotState(self.BLUE_START_POSE)
            self.enemy_robot = RobotState(self.YELLOW_START_POSE, is_enemy=True)
            self.my_robot.color = "yellow"
            self.enemy_robot.color = "blue"
        self.max_dir_capacity = 4.0
        self.max_robot_total_capacity = 16.0
        self.max_speed = 0.5
        self.dt = 1/30
        self.robot_radius = 0.15
        # time spent for every mission
        self.collect_time_steps = 30 * 2
        self.place_time_steps = 30 * 2
        self.flip_time_steps = 30 * 1
        # ------ params about enemy ------
        self.enemy_model = None
        # ------ params about model ------
        # model action space
        self.action_space = spaces.Discrete(18)
        # model observation space
        # 41 dimention: (Robot:2, Enemy:2, WaitTimer:1, Collection:8, Pantry:10*2, Held:4, Stage:1, Area:1, LastArea:1, Time:1)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(41,), dtype=np.float32
        )
        self.reset()

    def load_enemy_model(self, model_path):
        self.enemy_model = MaskablePPO.load(model_path, device="cpu")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0 # init game timer
        if self.my_color == "yellow":
            self.my_robot.reset(self.YELLOW_START_POSE)
            self.enemy_robot.reset(self.BLUE_START_POSE)
            self.my_robot.color = "yellow"
            self.enemy_robot.color = "blue"
        else:
            self.my_robot.reset(self.BLUE_START_POSE)
            self.enemy_robot.reset(self.YELLOW_START_POSE)
            self.my_robot.color = "blue"
            self.enemy_robot.color = "yellow"
        self.collect_counts = np.full(8, 4)      # init 8 collection area with 4 crates
        self.pantry_yellow_counts = np.zeros(10) # init pantry to 0 crates
        self.pantry_blue_counts = np.zeros(10)   # init pantry to 0 crates
        self.reward = 0.0
        self.fig = None                          # init render figure
        return self._get_obs(self.my_robot), {}
    
    def _normalize_pos(self, pos):
        return [(pos[0] / self.field_width) * 2 - 1, (pos[1] / self.field_height) * 2 - 1]

    def _get_obs(self, robot):
        opponent = self.enemy_robot if not robot.is_enemy else self.my_robot                       # identity swap
        robot_norm = self._normalize_pos(robot.pos)
        enemy_norm = self._normalize_pos(opponent.pos)
        max_steps = max(self.collect_time_steps, self.place_time_steps)
        wait_timer_norm = np.array([(robot.wait_timer / (max_steps * 2)) * 2 - 1], dtype=np.float32) # wait timer
        collection_info = [(c / self.max_collect_capacity) * 2 - 1 for c in self.collect_counts]   # 8 collection
        pantry_info = [] # 10 blue pantries + 10 yellow pantries
        for i in range(10):
            pantry_info.append((self.pantry_yellow_counts[i] / self.max_pantry_capacity) * 2 - 1)
            pantry_info.append((self.pantry_blue_counts[i] / self.max_pantry_capacity) * 2 - 1)
        held_info = [(h / self.max_dir_capacity) * 2 - 1 for h in robot.held_count]                # crate held on robot
        time_progress = (self.current_step / self.max_steps)                                       # game timer
        time_progress_norm = np.array([time_progress * 2 - 1], dtype=np.float32)
        # model stages and select area
        stage_norm = np.array([(robot.stage / 2.0) * 2 - 1], dtype=np.float32)
        area_norm = np.array([(robot.selected_area / 17.0) * 2 - 1 if robot.selected_area != -1 else -1.0], dtype=np.float32)
        last_area_norm = np.array([(robot.last_area_id / 17.0) * 2 - 1], dtype=np.float32)
        # 41 dimention: (Robot:2, Enemy:2, WaitTimer:1, Collection:8, Pantry:10*2, Held:4, Stage:1, Area:1, LastArea:1, Time:1)
        obs = np.concatenate([
            robot_norm, enemy_norm, wait_timer_norm, collection_info, pantry_info, 
            held_info, stage_norm, area_norm, last_area_norm, time_progress_norm
        ]).astype(np.float32)
        return obs

    # public function only call from outside of class
    def action_masks(self):
        return self._get_action_masks(self.my_robot)       # retrun my robot's action mask
    
    def _get_action_masks(self, robot):
        """
        generate action masks for current stage
        return a boolean list, len=action_space=18
        True: valid action, False: invalid action
        """
        mask = np.zeros(self.action_space.n, dtype=bool)   # init all action masks as false
        # --- identify enemy pantry color ---
        total_held = np.sum(robot.held_count)
        enemy_pantry_counts = self.pantry_blue_counts if robot.color == "yellow" else self.pantry_yellow_counts

        # ------ Stage 0: select area ------
        if robot.stage == 0:
            for i in range(18):
                if i < 10:                                                                   # pantry
                    pantry_total = self.pantry_yellow_counts[i] + self.pantry_blue_counts[i] # total crates in pantries
                    can_place = (total_held > 0 and pantry_total < self.max_pantry_capacity) # can place
                    can_flip = (enemy_pantry_counts[i] > 0)                                  # can flip
                    if can_place or can_flip:
                        mask[i] = True
                else:                                                                        # collection area
                    if total_held < self.max_robot_total_capacity and self.collect_counts[i-10] > 0: # can collect
                        mask[i] = True
            if not np.any(mask): mask[0] = True                                              # open on area to prevent no one can choose
        # ------ Stage 1: select entry point ------
        elif robot.stage == 1:
            # --- an action that can exit current stage. back to stage 0 ---
            if robot.steps_in_stage >= 15:  # can exit only when navigate over 0.5sec
                mask[17] = True 
            else:
                mask[17] = False            # forbid exit: preventing hesitation 
            # --- determine how many entry point for this area ---
            if robot.selected_area in self.entry_points_config:
                available_entries = self.entry_points_config[robot.selected_area].keys() # only open id that really has entry point
                for entry_id in available_entries:
                    # check entry point id < action_space number
                    if entry_id < self.action_space.n:
                        mask[entry_id] = True
            else:
                mask[0] = True              # if there's something wrong. need to open 1 mask at least
        # ------ Stage 2: execute mission ------
        elif robot.stage == 2:
            area_idx = robot.selected_area
            is_pantry = area_idx < 10
            exit_action = 5 if is_pantry else 4
            # --- check if current area has anything to do ---
            if not is_pantry:
                has_work = self.collect_counts[area_idx-10] > 0
            else:
                enemy_pantry = self.pantry_blue_counts if robot.color == "yellow" else self.pantry_yellow_counts
                my_pantry = self.pantry_blue_counts if robot.color == "blue" else self.pantry_yellow_counts
                has_work = (np.sum(robot.held_count) > 0 and my_pantry[area_idx] < self.max_pantry_capacity) or (enemy_pantry[area_idx] > 0)
            # --- decide if robot can exit or keep stay in current area ---
            if has_work:                       # stay
                if is_pantry: mask[0:6] = True # 0-5
                else: mask[0:5] = True         # 0-4
            else:                              # must exit
                mask[exit_action] = True
        return mask

    def _update_navigation(self, robot):
        # ------ physical movement ------
        reward = 0
        if robot.target_pos is not None:
            # --- check if robot overlap any area ---
            for i in range(18):
                # get the size of each area
                if i < 10:                       # pantry
                    r_min, r_max = self.pantry_rects_min[i], self.pantry_rects_max[i]
                    has_items = (self.pantry_yellow_counts[i] + self.pantry_blue_counts[i]) > 0
                else:                            # collect
                    r_min, r_max = self.collet_rects_min[i-10], self.collect_rects_max[i-10]
                    has_items = self.collect_counts[i-10] > 0
                #if there's anything in the area. and robot pass through
                if has_items and CollisionManager.check_circle_rect_collision(
                    robot.pos, self.robot_radius, r_min, r_max):
                    reward -= 0.02               # get punishment
            # --- simple navigation ---
            if robot.prev_dist == 0:
                robot.prev_dist = np.linalg.norm(robot.pos - robot.target_pos)
            move_dir = robot.target_pos - robot.pos
            dist = np.linalg.norm(move_dir)
            if dist > (self.max_speed * self.dt): # navigating
                robot.pos += (move_dir / dist) * (self.max_speed * self.dt)
            else:                                # navigation finished
                robot.pos = robot.target_pos.copy()
                robot.target_pos = None
                robot.last_stage = robot.stage
                robot.stage = 2
                robot.steps_in_stage = 0         # init steps counter of navigation
                robot.prev_dist = 0
                reward += 0.1                    # reward after arrival
            # --- give a litte reward when moving. according to moving distance ---
            if not robot.is_enemy:
                dist_to_goal = np.linalg.norm(robot.pos - (robot.target_pos if robot.target_pos is not None else robot.pos))
                reward += (robot.prev_dist - dist_to_goal) * 0.5
                robot.prev_dist = dist_to_goal
        return reward
    
    def _robot_logic(self, robot, action):
        """
        - stage change
        - execution in stages
        - return reward and info of executions
        """
        reward = 0 # record total rewrad in this loop of robot logic
        if robot.wait_timer > 0:
            return 0, {"status": "waiting"}
    
        robot.steps_in_stage += 1
        # ------ Stage 0: select area ------
        if robot.stage == 0:
            robot.last_area_id = robot.selected_area
            robot.selected_area = action
            # --- get the position of current target area ---
            if action < 10:
                target_center = self.pantry_pos[action]
            else:
                target_center = self.collect_pos[action-10]
            # --- punishment after choosing area ---
            dist_to_target = np.linalg.norm(robot.pos - target_center)                # distance between robot and target
            dist_cost = dist_to_target * 0.1                                          # declare a dist cost (if max dist=3.6. then dist cost is 0.36)
            if robot.selected_area == robot.last_finished_area:                       # tell robot not to choose area that have done sth last time
                reward -= 0.6
            if robot.last_stage == 1 and robot.selected_area != robot.last_area_id:   # need re-select. but give a little punishment
                reward -= 0.1
            elif robot.last_stage == 2 and robot.selected_area != robot.last_area_id: # should select area as near as possible
                reward -= dist_cost
            # --- competitive strategy ---
            area_has_items = False
            if action < 10:
                area_has_items = (self.pantry_yellow_counts[action] + self.pantry_blue_counts[action]) < self.max_pantry_capacity
            else:
                area_has_items = self.collect_counts[action-10] > 0
            if area_has_items:
                dist_enemy_to_target = np.linalg.norm(self.enemy_robot.pos - target_center)
                if dist_enemy_to_target < (dist_to_target - 0.5):                     # enemy is close to target than my_robot
                    competition_penalty = (dist_to_target - dist_enemy_to_target) * 0.07 # punishment is proportional to delta distance
                    reward -= min(competition_penalty, 0.4)                           # set upper limit
                elif dist_enemy_to_target > (dist_to_target + 0.5):
                    reward += 0.1
            # --- update stage of robot ---
            robot.last_stage = robot.stage
            robot.stage = 1
            robot.steps_in_stage = 0
            return reward, {"status": f"area_selected area: {action}"}

        # ------ Stage 1: select entry point. do physical movement ------
        elif robot.stage == 1:
            if action == 17:                             # decide to re-select the target area
                robot.last_stage = 1
                robot.stage = 0
                robot.steps_in_stage = 0
                return -0.05, {"status": "re-selecting"} # need to give a little punishment. shouldn't change frequently.
            robot.target_pos = np.array(self.entry_points_config[robot.selected_area][action])
            nav_reward = self._update_navigation(robot)
            reward += nav_reward
            robot.need_leave = 0
            return reward, {"status": f"navigating entry point: {action}"}

        # ------ Stage 2: execute mission ------
        elif robot.stage == 2:
            area_idx = robot.selected_area
            is_pantry = area_idx < 10
            exit_action = 5 if is_pantry else 4
            enemy_pantry = self.pantry_blue_counts if robot.color == "yellow" else self.pantry_yellow_counts
            my_pantry = self.pantry_blue_counts if robot.color == "blue" else self.pantry_yellow_counts

            # --- check if current area has anything to do: must same as mask function ---
            if not is_pantry:
                has_work = self.collect_counts[area_idx-10] > 0
            else:
                has_work = (np.sum(robot.held_count) > 0 and my_pantry[area_idx] < self.max_pantry_capacity) or (enemy_pantry[area_idx] > 0)
            
            # --- exit logic ---
            if action == exit_action:
                # --- calculate enemy threat ---
                has_contribute = robot.made_contribution            # temp save made_contribution value
                dist_to_enemy = np.linalg.norm(robot.pos - self.enemy_robot.pos)
                is_danger = dist_to_enemy < (self.robot_radius * 3.5)
                if has_contribute:
                    robot.last_finished_area = robot.selected_area  # update last area that robot really do sth
                    robot.made_contribution = False                 # reset flag
                # reset other veriables
                robot.last_stage = robot.stage
                robot.stage = 0
                robot.steps_in_stage = 0
                robot.need_leave = 0
                # --- reward and panelty of exit_action ---
                if has_contribute and is_danger:
                    return 0.5, {"status": "tactical_escape"}       # case 1：finish mission under danger
                elif not has_work:
                    return 0.3, {"status": "correct_exit"}          # case 2：finish mission
                else:
                    return -1.0, {"status": "waste_exit"}           # case 3：didn't do everything can do
            # --- mission execution logic
            if not is_pantry:                                       # collection area 10-17
                robot_dir = action                                  # 4 direction of robot: 0-3
                if self.collect_counts[area_idx-10] > 0 and robot.held_count[robot_dir] < self.max_dir_capacity # has sth and this dir of robot has space
                    count = min(self.collect_counts[area_idx-10], self.max_dir_capacity - robot.held_count[robot_dir])
                    robot.held_count[robot_dir] += count
                    self.collect_counts[area_idx-10] -= count
                    robot.made_contribution = True                  # flag: record robot really do sth
                    robot.wait_timer = self.collect_time_steps
                    reward += 1.5 * count
            else:                                                   # pantry 0-9
                if action == 0:                                     # flip crates: action 0
                    if enemy_pantry[area_idx] > 0:
                        count = enemy_pantry[area_idx]
                        reward += 4.0 * count
                        my_pantry[area_idx] += count
                        enemy_pantry[area_idx] = 0
                        robot.made_contribution = True
                        robot.wait_timer = self.flip_time_steps
                    else:                                           # nothing to flip but flip
                        reward -= 0.3
                else:                                               # place crates: action 1-4 -> robot dir 0-3
                    dir_idx = action - 1
                    cap_left = self.max_pantry_capacity - (my_pantry[area_idx] + enemy_pantry[area_idx])
                    if robot.held_count[dir_idx] > 0 and cap_left > 0:
                        count = min(robot.held_count[dir_idx], cap_left)
                        robot.held_count[dir_idx] -= count
                        my_pantry[area_idx] += count
                        robot.made_contribution = True
                        robot.wait_timer = self.place_time_steps
                        reward += 4.5 * count
            return reward, {"action_done": f"area_{area_idx}_act_{action}"}
        return 0, {}

    def step(self, action):
        # ------ initial ------
        self.current_step += 1                             # step timer
        terminated = False                                 # model condition
        truncated = False                                  # model condition
        total_reward = -0.005                              # time punishment
        # ------ update robot previous pose ------
        self.my_robot.prev_pos = self.my_robot.pos.copy()
        self.enemy_robot.prev_pos = self.enemy_robot.pos.copy()
        # ------ update wait timer ------
        if self.my_robot.wait_timer > 0: self.my_robot.wait_timer -= 1
        if self.enemy_robot.wait_timer > 0: self.enemy_robot.wait_timer -= 1
        # ------ update enemy decision ------
        if self.enemy_model is not None and self.enemy_robot.wait_timer == 0:                      # make sure enemy model loaded
            e_obs = self._get_obs(self.enemy_robot)
            e_mask = self._get_action_masks(self.enemy_robot)
            e_action, _ = self.enemy_model.predict(e_obs, action_masks=e_mask, deterministic=True) # get enemy action from enemy PPO model
            # execute enemy action
            _ = self._robot_logic(self.enemy_robot, int(e_action))
        # ------ update my robot decision ------
        # --- prevent invalid action that doesn't masked. (theoretically impossible) ---
        current_mask = self._get_action_masks(self.my_robot)
        action = int(action)                               # get action and change np.array into integer
        self.action = action
        if not current_mask[action]:
            total_reward -= 1.0                            # give heavy punishment when choosing invalid action
            self.reward = total_reward
            return self._get_obs(self.my_robot), total_reward, False, False, {"error": "Action Mask Violation"}
        # --- execute my robot action ---
        step_reward, info = self._robot_logic(self.my_robot, action)
        total_reward += step_reward
        # --- collision avoidance logic ---
        dist_now = np.linalg.norm(self.my_robot.pos - self.enemy_robot.pos)
        dist_before = np.linalg.norm(self.my_robot.prev_pos - self.enemy_robot.prev_pos)
        collision_limit = self.robot_radius * 2.2
        if dist_now < collision_limit:                     # collid happened
            total_reward -= 0.5
        elif dist_now < collision_limit * 2:               # my_robot is in danger
            if dist_now > dist_before:                     # it's leaving danger: give reward
                total_reward += 0.08 
            else:                                          # it's going into danger: give panelty
                total_reward -= 0.05
        # ------ time pressure ------
        time_ratio = self.current_step / self.max_steps
        if time_ratio > 0.9 and np.sum(self.my_robot.held_count) > 0:
            total_reward -= 0.01 * np.sum(self.my_robot.held_count)
        # ------ end game condition ------
        truncated = self.current_step >= self.max_steps
        if np.sum(self.pantry_yellow_counts) >= 17: 
            terminated = True
            total_reward += 50.0
        self.reward = total_reward                         # record reward to class public variable
        return self._get_obs(self.my_robot), total_reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human": return
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
            # --- draw crates in collection area and pantry ---
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
            def create_robot_visuals(robot, color, ec):
                patch = patches.Circle(robot.pos, self.robot_radius, color=color, ec=ec, lw=2, zorder=10)
                self.ax.add_patch(patch)
                wedges = []
                angles = [(45, 135), (225, 315), (135, 225), (315, 405)] 
                colors = ["#EEA695", "#A5DDAF", "#AFB8E3", "#E6AEEA"] # use 4 color to represent crates in 4 directions
                for i in range(4):
                    w = patches.Wedge(robot.pos, 0, angles[i][0], angles[i][1], 
                                      color=colors[i], alpha=0.7, zorder=11)
                    wedges.append(w)
                    self.ax.add_patch(w)
                return patch, wedges
            # --- create my robot (white) and enemy robot (gray) ---
            self.my_patch, self.my_wedges = create_robot_visuals(self.my_robot, 'white', 'black')
            self.enemy_patch, self.enemy_wedges = create_robot_visuals(self.enemy_robot, '#E0E0E0', 'red')
        # ------ update everything keep changing ------
        stage_names = ["Select Area", "Select Entry", "Execute"]
        # --- update title: my info and enemy info ---
        my_info = f"MY - Stage: {stage_names[self.my_robot.stage]} | Act: {self.action} | R: {self.reward:.2f}"
        enemy_info = f"ENEMY - Stage: {stage_names[self.enemy_robot.stage]} | Act: {self.enemy_robot.action}"
        self.ax.set_title(f"{my_info}\n{enemy_info}", fontsize=10)
        # --- update objects on map ---
        for c in range(8):
            new_width = self.collect_sizes[c][0] * (self.collect_counts[c] / self.max_collect_capacity)
            self.collect_patches[c].set_width(new_width)
        for p in range(10):
            y_w = self.pantry_size * (self.pantry_yellow_counts[p] / self.max_pantry_capacity)
            self.pantry_yellow_patches[p].set_width(y_w)
            b_w = self.pantry_size * (self.pantry_blue_counts[p] / self.max_pantry_capacity)
            self.pantry_blue_patches[p].set_width(b_w)
            self.pantry_blue_patches[p].set_xy(self.pantry_rects_min[p] + np.array([y_w, 0]))
        # function: update robot position and crates condition on robot
        def update_robot_visuals(robot, patch, wedges):
            patch.center = robot.pos
            for i in range(4):
                wedges[i].set_center(robot.pos)
                dynamic_radius = self.robot_radius * (robot.held_count[i] / self.max_dir_capacity)
                wedges[i].set_radius(dynamic_radius)
        update_robot_visuals(self.my_robot, self.my_patch, self.my_wedges)
        update_robot_visuals(self.enemy_robot, self.enemy_patch, self.enemy_wedges)
        # --- update canvas ---
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

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