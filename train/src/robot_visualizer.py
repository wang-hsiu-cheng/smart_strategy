import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class RobotVisualizer:
    def __init__(self, config):
        """
        config params: 
        pantry_rects_min, collet_rects_min, collect_sizes, 
        robot_radius, max_collect_capacity, max_pantry_capacity, 
        max_dir_capacity, pantry_size
        """
        self.__dict__.update(config)
        self.fig, self.ax = None, None
        self.stage_names = ["Select Area", "Select Entry", "Execute"]

    # ------ initial ax and static variable (run one time) ------
    def init_plot(self):
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
        # --- create my robot (white) and enemy robot (gray) ---
        self.my_patch, self.my_wedges, self.head_marker = self._create_robot_visuals('white', 'black')
        self.enemy_patch, self.enemy_wedges, self.enemy_marker = self._create_robot_visuals('#E0E0E0', 'red')
    
    # function: draw robot and crate on robot
    def _create_robot_visuals(self, color, ec):
        patch = patches.Circle((0, 0), self.robot_radius, color=color, ec=ec, lw=2, zorder=10)
        self.ax.add_patch(patch)
        head_marker = patches.Circle((0, 0), self.robot_radius*0.2, color='black', zorder=12) # init head of robot
        self.ax.add_patch(head_marker)
        wedges = []
        angles = [(315, 405), (45, 135), (135, 225), (225, 315)]
        colors = ["#EEA695", "#A5DDAF", "#AFB8E3", "#E6AEEA"]
        for i in range(4):
            w = patches.Wedge((0, 0), 0, angles[i][0], angles[i][1], color=colors[i], alpha=0.7, zorder=11)
            wedges.append(w)
            self.ax.add_patch(w)
        return patch, wedges, head_marker

    # ------ update everything changing according to data dictionary------
    def update(self, data):
        # --- update title: my info and enemy info ---
        my_info = f"MY - Stage: {self.stage_names[data['my_stage']]} | Act: {data['my_act']} | R: {data['reward']:.2f}"
        enemy_info = f"ENEMY - Stage: {self.stage_names[data['en_stage']]} | Act: {data['en_act']}"
        self.ax.set_title(f"{my_info}\n{enemy_info}", fontsize=10)
        # --- update objects on map ---
        for c in range(8):
            w = self.collect_sizes[c][0] * (data['collect_counts'][c] / self.max_collect_capacity)
            self.collect_patches[c].set_width(w)
        for p in range(10):
            y_w = self.pantry_size * (data['pantry_y'][p] / self.max_pantry_capacity)
            self.pantry_yellow_patches[p].set_width(y_w)
            b_w = self.pantry_size * (data['pantry_b'][p] / self.max_pantry_capacity)
            self.pantry_blue_patches[p].set_width(b_w)
            self.pantry_blue_patches[p].set_xy(self.pantry_rects_min[p] + np.array([y_w, 0]))
        # update robots pose and direction
        self._update_robot(data['my_pos'], data['my_held'], self.my_patch, self.my_wedges, self.head_marker)
        self._update_robot(data['en_pos'], data['en_held'], self.enemy_patch, self.enemy_wedges, self.enemy_marker)
        # --- update canvas ---
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # function: update robot position and crates condition on robot
    def _update_robot(self, pos, held, patch, wedges, head_marker):
        patch.center = pos[:2]
        heading_deg = pos[2] * 90.0 # update head of robot
        rad = np.radians(heading_deg)
        marker_pos = pos[:2] + np.array([np.cos(rad), np.sin(rad)]) * (self.robot_radius * 0.7)
        head_marker.center = marker_pos
        for i in range(4):
            wedges[i].set_center(pos[:2])
            center_angle = heading_deg + (i * 90.0)
            wedges[i].theta1 = center_angle - 45.0
            wedges[i].theta2 = center_angle + 45.0
            dynamic_radius = self.robot_radius * (held[i] / self.max_dir_capacity)
            wedges[i].set_radius(dynamic_radius)