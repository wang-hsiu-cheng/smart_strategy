import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from robot_state import RobotState
from collision_manager import CollisionManager
import numpy as np

class RobotMatchEnv(gym.Env):
    # define matadata params
    metadata = {"my_color": ["yellow", "blue"], "render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(self, render_mode=None, my_color=None):
        super().__init__()
        self.render_mode = render_mode
        self.my_color = my_color
        self.max_steps = 100 * self.metadata["render_fps"]
        self.current_step = 0
        self.is_game_end = False                 # flag: true if robot arrived home
        # -------- params about game field ------
        self.YELLOW_START_POSE = np.array([0.3, 1.8, 0.0])
        self.BLUE_START_POSE = np.array([2.7, 1.8, 2.0])
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
            0: {3: (1.25, 1.15)},
            1: {3: (1.75, 1.15)}, 
            2: {0: (0.4, 0.8)},
            3: {0: (1.1, 0.8), 1: (0.8, 1.1), 2: (0.8, 0.5), 3: (0.5, 0.8)},
            4: {0: (1.8, 0.8), 1: (1.5, 1.1), 2: (1.2, 0.8), 3: (1.5, 0.5)},
            5: {0: (2.5, 0.8), 1: (2.2, 1.1), 2: (1.9, 0.8), 3: (2.2, 0.5)},
            6: {2: (2.6, 0.8)}, 
            7: {1: (0.7, 0.4)}, 
            8: {1: (1.5, 0.4)}, 
            9: {1: (2.3, 0.4)},
            # collect
            10: {0: (0.45, 1.2)}, 
            11: {2: (2.55, 1.2)}, 
            12: {1: (1.15, 1.075), 3: (1.15, 0.525)},
            13: {1: (1.85, 1.075), 3: (1.85, 0.525)}, 
            14: {0: (0.45, 0.4)}, 
            15: {2: (2.55, 0.4)},
            16: {1: (1.1, 0.45)}, 
            17: {1: (1.9, 0.45)}
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
            self.my_robot.color = "blue"
            self.enemy_robot.color = "yellow"
        self.max_dir_capacity = 4.0
        self.max_robot_total_capacity = 16.0
        self.max_speed = 0.4
        self.dt = 1/30
        self.robot_radius = 0.15
        # time spent for every mission
        self.collect_time_steps = 30 * 2
        self.place_time_steps = 30 * 2
        self.flip_time_steps = 30 * 1
        self.rotate_time_steps = 30 * 2
        # ------ params about enemy ------
        self.enemy_model = None
        # ------ params about model ------
        # model action space
        self.action_space = spaces.Discrete(19)
        # model observation space
        # 43 dimention: (Robot:3, Enemy:3, WaitTimer:1, Collection:8, Pantry:10*2, Held:4, Stage:1, Area:1, LastArea:1, Time:1)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(43,), dtype=np.float32
        )
        self.reset()

    def load_enemy_model(self, model_path):
        self.enemy_model = MaskablePPO.load(model_path, device="cpu")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0 # init game timer
        self.is_game_end = False
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
        return self._get_obs(self.my_robot), {}
    
    def _normalize_pos(self, pos):
        return [(pos[0] / self.field_width) * 2 - 1, (pos[1] / self.field_height) * 2 - 1, (pos[2] / 3.0) * 2 - 1]

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
        # 43 dimention: (Robot:3, Enemy:3, WaitTimer:1, Collection:8, Pantry:10*2, Held:4, Stage:1, Area:1, LastArea:1, Time:1)
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
            # --- go home: action 18 ---
            if self.current_step > 2000:                                                     # can go home only when game time > 66sec
                mask[18] = True
            else:
                mask[18] = False
            if not np.any(mask): mask[0] = True                                              # open on area to prevent no one can choose
        # ------ Stage 1: select entry point ------
        elif robot.stage == 1:
            # if robot.nav_fail:
            #     if robot.steps_in_stage >= 15:  # can exit only when navigate over 0.5sec
            #         mask[17] = True
            #     else:
            #         pass
            # --- an action that can exit current stage. back to stage 0 ---
            if robot.steps_in_stage >= 15:  # can exit only when navigate over 0.5sec
                mask[17] = True
            else:
                mask[17] = False            # forbid exit: preventing hesitation 
            # --- determine how many entry point for this area ---
            if robot.selected_area in self.entry_points_config:
                available_entry_ids = self.entry_points_config[robot.selected_area].keys() # only open id that really has entry point
                for action_id in range(16): # use action_id to represent entry_point(action_id / 4) and robot direction(action_id % 4)
                    entry_id = action_id // 4
                    target_dir = action_id % 4
                    # check entry point id < action_space number
                    if entry_id in available_entry_ids:
                        # --- 核心修正：預判抵達後的 mission_dir 是否有意義 ---
                        mission_dir = int(((entry_id + 2) - target_dir + 4) % 4)
                        if self._check_dir_validation(robot, mission_dir) is not None:
                            mask[action_id] = True
                        else:
                            mask[action_id] = False # 不准選一個沒事做的面出發
            if not np.any(mask[:16]):
                mask[17] = True
            # else:
            #     mask[0] = True              # if there's something wrong. need to open 1 mask at least
        # ------ Stage 2: execute mission ------
        elif robot.stage == 2:
            area_idx = robot.selected_area
            is_pantry = area_idx < 10
            exit_action = 4
            # --- check if current area has anything to do ---
            if not is_pantry:
                has_work = self.collect_counts[area_idx-10] > 0
            else:
                enemy_pantry = self.pantry_blue_counts if robot.color == "yellow" else self.pantry_yellow_counts
                my_pantry = self.pantry_blue_counts if robot.color == "blue" else self.pantry_yellow_counts
                has_work = (np.sum(robot.held_count) > 0 and my_pantry[area_idx] < self.max_pantry_capacity) or (enemy_pantry[area_idx] > 0 and np.any(robot.held_count == 0))
            # --- decide if robot can exit or keep stay in current area ---
            if has_work:                       # stay
                mission_dir = int(((robot.entry_id + 2) - robot.pos[2] + 4) % 4)
                if robot.last_stage == 1:
                    if self._check_dir_validation(robot, mission_dir) is not None:
                        mask[mission_dir] = True
                    else:
                        # 否則，開放 0-4 讓它有機會旋轉或撤退
                        for dir in range(4):
                            if self._check_dir_validation(robot, dir) is not None:
                                mask[dir] = True
                        mask[exit_action] = True
                else:
                    for dir in range(4):
                        if not self._check_dir_validation(robot, dir) == None:
                            mask[dir] = True           # 0-4
                    mask[exit_action] = True
            else:                              # must exit
                mask[exit_action] = True
        return mask

    def _collid_crate_check(self, robot):
        reward = 0
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
                robot.pos[:2], self.robot_radius, r_min, r_max):
                reward -= 0.2                # get punishment
        return reward
    def _update_navigation(self, robot):
        # ------ physical movement ------
        reward = 0
        if robot.target_pos is not None:
            # --- simple navigation ---
            if robot.prev_dist == 0:
                robot.prev_dist = np.linalg.norm(robot.pos[:2] - robot.target_pos[:2])
            move_dir = robot.target_pos[:2] - robot.pos[:2]
            dist = np.linalg.norm(move_dir)
            if dist > (self.max_speed * self.dt): # navigating
                robot.pos[:2] += (move_dir / dist) * (self.max_speed * self.dt)
                robot.pos[2] = robot.target_pos[2].copy()
            else:                                 # navigation finished
                robot.pos = robot.target_pos.copy()
                robot.target_pos = None
                robot.last_stage = robot.stage
                robot.stage = 2
                robot.mission_fail = False
                robot.steps_in_stage = 0          # init steps counter of navigation
                robot.prev_dist = 0
                if robot.is_going_home:
                    self.is_game_end = True
                reward += 0.1                     # reward after arrival
            # --- give a litte reward when moving. according to moving distance ---
            if not robot.is_enemy:
                dist_to_goal = np.linalg.norm(robot.pos[:2] - (robot.target_pos[:2] if robot.target_pos is not None else robot.pos[:2]))
                reward += (robot.prev_dist - dist_to_goal) * 0.5
                robot.prev_dist = dist_to_goal
        return reward
    
    def _check_dir_validation(self, robot, mission_dir):
        mission_type = None
        area_idx = robot.selected_area
        is_pantry = area_idx < 10
        enemy_pantry = self.pantry_blue_counts if robot.color == "yellow" else self.pantry_yellow_counts
        my_pantry = self.pantry_blue_counts if robot.color == "blue" else self.pantry_yellow_counts
        
        if not is_pantry:                                       # collection area 10-17
            if self.collect_counts[area_idx-10] > 0 and robot.held_count[mission_dir] < self.max_dir_capacity: # has sth and this dir of robot has space
                mission_type = "collect"
        else:                                                   # pantry 0-9
            if robot.held_count[mission_dir] == 0 and enemy_pantry[area_idx] > 0 :
                mission_type = "flip"
            elif robot.held_count[mission_dir] > 0 and (self.max_pantry_capacity - (my_pantry[area_idx] + enemy_pantry[area_idx])) > 0: # place crates: action 0-3
                mission_type = "place"
        return mission_type
                
    def _robot_logic(self, robot, action):
        """
        - stage change
        - execution in stages
        - return reward and info of executions
        """
        reward = 0 # record total rewrad in this loop of robot logic
        reward += self._collid_crate_check(robot) # check if robot overlap any area
        if robot.wait_timer > 0:
            return reward, {"status": "waiting"}
    
        robot.steps_in_stage += 1
        # ------ Stage 0: select area ------
        if robot.stage == 0:
            robot.last_area_id = robot.selected_area
            robot.selected_area = action
            # --- get the position of current target area ---
            if action == 18:
                # chooose home of the robot
                robot.target_pos = self.YELLOW_START_POSE.copy() if robot.color == "yellow" else self.BLUE_START_POSE.copy()
                robot.is_going_home = True
                robot.stage = 1
                robot.nav_fail = False
                robot.steps_in_stage = 0
                return 0, {"status": "heading_home"}
            if action < 10:
                target_center = self.pantry_pos[action]
            else:
                target_center = self.collect_pos[action-10]
            # --- punishment after choosing area ---
            dist_to_target = np.linalg.norm(robot.pos[:2] - target_center)            # distance between robot and target
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
                dist_enemy_to_target = np.linalg.norm(self.enemy_robot.pos[:2] - target_center)
                if dist_enemy_to_target < (dist_to_target - 0.5):                     # enemy is close to target than my_robot
                    competition_penalty = (dist_to_target - dist_enemy_to_target) * 0.07 # punishment is proportional to delta distance
                    reward -= min(competition_penalty, 0.4)                           # set upper limit
                elif dist_enemy_to_target > (dist_to_target + 0.5):
                    reward += 0.1
            # --- update stage of robot ---
            robot.last_stage = robot.stage
            robot.stage = 1
            robot.nav_fail = False
            robot.steps_in_stage = 0
            return reward, {"status": f"area_selected area: {action}"}

        # ------ Stage 1: select entry point. do physical movement ------
        elif robot.stage == 1:
            if getattr(robot, 'is_going_home', False):   # need to go home
                nav_reward = self._update_navigation(robot) # do navigation directly
                reward += nav_reward
                return reward, {"status": "navigating_home"}
            if action == 17:                             # decide to re-select the target area
                robot.last_stage = 1
                robot.stage = 0
                robot.steps_in_stage = 0
                return -0.05, {"status": "re-selecting"} # need to give a little punishment. shouldn't change frequently.
            # get entry point and robot direction from action_id
            if robot.last_stage == 0: # cannot change entry point and robot direction unless re-select target area
                robot.last_stage = 1
                robot.entry_id = action // 4
                robot_dir = action % 4
                mission_dir = int(((robot.entry_id + 2) - robot_dir + 4) % 4)
                robot.target_pos = np.concatenate([self.entry_points_config[robot.selected_area][robot.entry_id], [float(robot_dir)]])
                mission_type = self._check_dir_validation(robot, mission_dir)
                if mission_type == None:
                    robot.nav_fail = True
                    reward -= 3.0
            nav_reward = self._update_navigation(robot) # keep navigating
            reward += nav_reward
            return reward, {"status": f"navigating entry point: {robot.entry_id} robot dir: {robot.pos[2]}"}

        # ------ Stage 2: execute mission ------
        elif robot.stage == 2:
            robot.wait_timer = 0
            area_idx = robot.selected_area
            is_pantry = area_idx < 10
            exit_action = 4
            enemy_pantry = self.pantry_blue_counts if robot.color == "yellow" else self.pantry_yellow_counts
            my_pantry = self.pantry_blue_counts if robot.color == "blue" else self.pantry_yellow_counts

            # --- check if current area has anything to do: must same as mask function ---
            if not is_pantry:
                has_work = self.collect_counts[area_idx-10] > 0
            else:
                has_work = (np.sum(robot.held_count) > 0 and my_pantry[area_idx] < self.max_pantry_capacity) or (enemy_pantry[area_idx] > 0 and np.any(robot.held_count == 0))
            # --- exit logic ---
            if action == exit_action:
                # --- calculate enemy threat ---
                has_contribute = robot.made_contribution            # temp save made_contribution value
                dist_to_enemy = np.linalg.norm(robot.pos[:2] - self.enemy_robot.pos[:2])
                is_danger = dist_to_enemy < (self.robot_radius * 3.5)
                if has_contribute:
                    robot.last_finished_area = robot.selected_area  # update last area that robot really do sth
                    robot.made_contribution = False                 # reset flag
                # reset other veriables
                robot.last_stage = robot.stage
                robot.stage = 0
                robot.steps_in_stage = 0
                # --- reward and panelty of exit_action ---
                if has_contribute and is_danger:
                    return 0.8, {"status": "do & escape"}           # case 1: done mission under danger
                elif has_contribute and has_work:
                    return 0.2, {"status": "do & exit"}             # case 2: done mission, but didn't finish all missions
                elif has_contribute and not has_work:
                    return 0.5, {"status": "finish & exit"}         # case 3： finish all missions
                else:
                    return -1.0, {"status": "wrong exit"}           # case 4： didn't do any mission
            # --- determine mission direction ---                  
            mission_dir = action
            # 2. 計算目前物理上對準目標的面
            aligned_face = int(((robot.entry_id + 2) - robot.pos[2] + 4) % 4)
            # 3. 判斷是否需要旋轉
            if mission_dir != aligned_face:
                # 機器人選的面沒對準，需要原地旋轉
                robot_dir = ((robot.entry_id + 2) - mission_dir + 4) % 4
                robot.wait_timer += abs((robot_dir - robot.pos[2] + 2) % 4 - 2) * self.rotate_time_steps
                robot.pos[2] = robot_dir
            robot.last_stage = robot.stage    
            # --- mission execution logic ---
            mission_type = self._check_dir_validation(robot, mission_dir)
            if mission_type == "collect":                            # has sth and this dir of robot has space
                count = min(self.collect_counts[area_idx-10], self.max_dir_capacity - robot.held_count[mission_dir])
                robot.held_count[mission_dir] += count
                self.collect_counts[area_idx-10] -= count
                robot.made_contribution = True                          # flag: record robot really do sth
                robot.wait_timer += self.collect_time_steps
                reward += (0.2 * count) if robot.mission_fail else (1.0 * count)
                robot.mission_fail = False
                return reward, {"collect success": f" reward:{reward},area:{area_idx},dir:{mission_dir}"}
            elif mission_type == "flip":                            # flip crates
                count = enemy_pantry[area_idx]
                reward += (0.8 * count) if robot.mission_fail else (3.0 * count)
                my_pantry[area_idx] += count
                enemy_pantry[area_idx] = 0
                robot.made_contribution = True
                robot.wait_timer += self.flip_time_steps
                robot.mission_fail = False
                return reward, {"flip success": f" reward:{reward},area:{area_idx},dir:{mission_dir}"}
            elif mission_type == "place":                          # place crates
                cap_left = self.max_pantry_capacity - (my_pantry[area_idx] + enemy_pantry[area_idx])
                count = min(robot.held_count[mission_dir], cap_left)
                robot.held_count[mission_dir] -= count
                my_pantry[area_idx] += count
                robot.made_contribution = True
                robot.wait_timer += self.place_time_steps
                reward += (1.0 * count) if robot.mission_fail else (4.0 * count)
                robot.mission_fail = False
                return reward, {"place success": f" reward:{reward},area:{area_idx},dir:{mission_dir}"}
            else:
                robot.mission_fail = True
                reward -= 1.5
                return reward, {"exception": f"{robot.last_stage} reward:{reward},area:{area_idx},dir:{mission_dir}"}
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
        dist_now = np.linalg.norm(self.my_robot.pos[:2] - self.enemy_robot.pos[:2])
        dist_before = np.linalg.norm(self.my_robot.prev_pos[:2] - self.enemy_robot.prev_pos[:2])
        collision_limit = self.robot_radius * 2.2
        if dist_now < collision_limit:                     # collid happened
            total_reward -= 10.0
        elif dist_now < collision_limit * 2:               # my_robot is in danger
            if dist_now > dist_before:                     # it's leaving danger: give reward
                total_reward += 0.08 
            else:                                          # it's going into danger: give panelty
                total_reward -= 0.05
        # --- defense my pantry ---
        my_pantry_counts = self.pantry_blue_counts if self.my_robot.color == "blue" else self.pantry_yellow_counts
        for i in range(10):
            if my_pantry_counts[i] > 0:  # 只防守有自己方塊的區域
                p_pos = self.pantry_pos[i]
                if np.linalg.norm(self.my_robot.pos[:2] - p_pos) < 0.3 and np.linalg.norm(self.enemy_robot.pos[:2] - p_pos) < 0.5:
                    total_reward += 0.03
        # --- time pressure ---
        time_ratio = self.current_step / self.max_steps
        if time_ratio > 0.8 and np.sum(self.my_robot.held_count) > 0:
            total_reward -= 0.001 * np.sum(self.my_robot.held_count)
        # --- end game condition ---
        if self.current_step >= self.max_steps:
            truncated = True
            total_reward -= 20.0
        if self.is_game_end:
            terminated = True
            # --- calculate left time bonus ---
            remaining_steps = self.max_steps - self.current_step
            time_bonus = remaining_steps * 0.03            # get 0.03 points for each step
            # (basic point(10) + time bonus)* time_ratio^n
            total_reward += (5.0 + time_bonus) * np.power(time_ratio, 3)
            # $$Weight = 0.5 \times (1 + \tanh(k \times (Progress - Offset)))$$
            if np.sum(my_pantry_counts) >= 16:
                total_reward += 20.0
            elif np.sum(my_pantry_counts) >= 20:
                total_reward += 50.0
        self.reward = total_reward                         # record reward to class public variable
        return self._get_obs(self.my_robot), total_reward, terminated, truncated, info