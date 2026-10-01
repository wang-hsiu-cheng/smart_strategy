#pragma once

// Use C++ libraries
#include <filesystem>
#include <fstream>
#include <deque>
#include <bitset>
#include <vector>
#include <math.h> 

// Use behavior tree
#include <behaviortree_ros2/bt_action_node.hpp>
#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/behavior_tree.h>

// Use ROS
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logger.hpp>

// Use ros message
#include <std_msgs/msg/int32.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>

// tf2
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>

using namespace BT;
using namespace std;

namespace BT {
    template <> inline geometry_msgs::msg::PoseStamped convertFromString(StringView str);
    template <> inline int convertFromString(StringView str);
    template <> inline std::deque<int> convertFromString(StringView str);
    template <> inline std::deque<double> convertFromString(StringView str);
}

class Mission : public BT::StatefulActionNode
{
public:
  Mission(const std::string& name, const BT::NodeConfig& config, const RosNodeParams& params, BT::Blackboard::Ptr blackboard)
    : BT::StatefulActionNode(name, config), node_(params.nh.lock()), blackboard_(blackboard), rate_(30) {
    publisher_ = node_->create_publisher<std_msgs::msg::Int32>("mission_type", 10);
    subscription_ = node_->create_subscription<std_msgs::msg::Int32>("mission_status", 10, std::bind(&Mission::mission_callback, this, std::placeholders::_1));
  }

  /* Node remapping function */
  static BT::PortsList providedPorts();

  /* Start and running function */
  BT::NodeStatus onStart() override;
  BT::NodeStatus onRunning() override;

  void mission_callback(const std_msgs::msg::Int32::SharedPtr sub_msg);
  BT::NodeStatus stopStep();
  /* Halt function */
  void onHalted() override;

private:
  std::shared_ptr<rclcpp::Node> node_;
  BT::Blackboard::Ptr blackboard_;
  rclcpp::Rate rate_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr publisher_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr subscription_;

  std_msgs::msg::Int32 pub_msg;
  int mission_progress_ = 0;
  int missionType_ = 0;
  int mission_status_ = 0;
  int mission_stamp_ = 0;
  bool mission_received_ = false;
  bool mission_finished_ = false;
};