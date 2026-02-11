import rclpy
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from threading import Lock

from opennav_docking_msgs.action import DockRobot
from geometry_msgs.msg import PoseStamped

class BaseNavAction:
    def __init__(self, node, action_type, action_name, callback_group):
        self.node = node
        self.action_name = action_name
        self.client = ActionClient(node, action_type, action_name, callback_group=callback_group)
        
        self.active_handle = None
        self.is_running = False
        self.lock = Lock()
        self.on_finished_callback = None 

    def execute(self, goal_msg, finished_cb=None):
        with self.lock:
            if self.is_running:
                return False
            self.is_running = True
            self.on_finished_callback = finished_cb

        if not self.client.wait_for_server(timeout_sec=1.0):
            self.node.get_logger().error(f"[{self.action_name}] 伺服器不可用")
            self._cleanup()
            return False

        future = self.client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_callback)
        return True

    def cancel(self, cancelled_cb=None):
        with self.lock:
            if not self.active_handle:
                if cancelled_cb: cancelled_cb()
                return False
            
            # 中斷不觸發 finished_cb，而是觸發專用的 cancelled_cb
            self.on_finished_callback = None 
            future = self.active_handle.cancel_goal_async()
            future.add_done_callback(lambda f: self._cancel_done(f, cancelled_cb))
            return True

    def _goal_response_callback(self, future):
        handle = future.result()
        if not handle.accepted:
            self._cleanup()
            return
        self.active_handle = handle
        handle.get_result_async().add_done_callback(self._get_result_callback)

    def _get_result_callback(self, future):
        status = future.result().status
        # 關鍵：先提取回呼函式，再清理內部狀態
        cb = self.on_finished_callback
        self._cleanup()
        
        # 通知主程式：動作已結束，並帶上狀態
        if cb:
            cb(status)

    def _cancel_done(self, future, cb):
        self._cleanup()
        if cb: cb()

    def _cleanup(self):
        with self.lock:
            self.active_handle = None
            self.is_running = False
            self.on_finished_callback = None

class Navigate(BaseNavAction):
    def __init__(self, node, cb_group):
            super().__init__(node, DockRobot, 'dock_robot', cb_group)

    def execute(self, x, y, theta=1.0, finished_cb=None):
        goal = DockRobot.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.w = theta
        return super().execute(goal, finished_cb)

class Dock(BaseNavAction):
    def __init__(self, node, cb_group):
        super().__init__(node, DockRobot, 'dock_robot', cb_group)

    def execute(self, finished_cb=None):
        goal = DockRobot.Goal()
        goal.use_dock_id = True
        goal.dock_id = "pantry_dock" # 範例參數
        return super().execute(goal, finished_cb)

class BackRotateForward(BaseNavAction):
    def __init__(self, node, cb_group):
        super().__init__(node, DockRobot, 'dock_robot', cb_group)

    def execute(self, finished_cb=None):
        goal = DockRobot.Goal()
        goal.use_dock_id = True
        goal.dock_id = "pantry_dock" # 範例參數
        return super().execute(goal, finished_cb)