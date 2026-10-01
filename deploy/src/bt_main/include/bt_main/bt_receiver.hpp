#pragma once

// Use C++ libraries
#include <filesystem>
#include <fstream>
#include <deque>
#include <bitset>

// Use behavior tree
#include <behaviortree_ros2/bt_action_node.hpp>
#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/behavior_tree.h>

// Use ROS
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logger.hpp>

// Use ros message
#include <geometry_msgs/msg/pose_stamped.hpp>

// tf2
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/impl/utils.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/exceptions.h>

using namespace BT;

/***************/
/* LocReceiver */
/***************/
// receive the robot pose and rival pose from localization team
class LocReceiver
{
public:
  LocReceiver(const RosNodeParams& params)
    : node_(params.nh.lock()), tf_buffer_(node_->get_clock()), listener_(tf_buffer_)
  {
    node_->get_parameter("frame_id", frame_id_);
  }
  static bool UpdateRobotPose(geometry_msgs::msg::PoseStamped &robot_pose_, tf2_ros::Buffer &tf_buffer_, std::string frame_id_);
  static bool UpdateRivalPose(geometry_msgs::msg::PoseStamped &rival_pose_, tf2_ros::Buffer &tf_buffer_, std::string frame_id_);
private:
  std::shared_ptr<rclcpp::Node> node_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener listener_;
  std::string frame_id_;

  geometry_msgs::msg::PoseStamped robot_pose_;
  geometry_msgs::msg::PoseStamped rival_pose_;
};