
class SensorManager():
    def __init__(self):
        super().__init__('sensor_node')
        # declare tf listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
def main(args=None):
    rclpy.init(args=args)
    node = SensorManager()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()