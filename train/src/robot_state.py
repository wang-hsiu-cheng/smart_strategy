import numpy as np

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
        self.entry_id = -1
        self.wait_timer = 0
        self.held_count = np.zeros(4, dtype=int)
        self.target_pos = None
        self.prev_dist = np.zeros(1)    # init previous dist of nav as 0
        self.action = 0
        self.nav_fail = False
        self.is_going_home = False      # flag: true if robot is going home
        self.made_contribution = False  # flag: did robot really do sth in an area
        self.mission_fail = False
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
        self.entry_id = -1
        self.wait_timer = 0
        self.held_count.fill(0)
        self.target_pos = None
        self.prev_dist = np.zeros(1)    # init previous dist of nav as 0
        self.action = 0
        self.nav_fail = False
        self.is_going_home = False
        self.made_contribution = False  # init flag as false
        self.mission_fail = False
        self.is_enemy = is_enemy