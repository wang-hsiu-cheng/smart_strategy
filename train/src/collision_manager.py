import numpy as np

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