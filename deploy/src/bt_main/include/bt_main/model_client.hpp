#pragma once

#include <behaviortree_ros2/bt_action_node.hpp>
#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/behavior_tree.h>

#include <std_msgs/msg/int64.hpp>
#include <std_msgs/msg/float32.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>

using namespace BT;
using namespace std;

namespace BT {
    template <> inline geometry_msgs::msg::PoseStamped convertFromString(StringView str);
    template <> inline int convertFromString(StringView str);
    template <> inline std::deque<int> convertFromString(StringView str);
}

class ModelClient : public DecoratorNode
{
public:
    ModelClient(const string& name, const NodeConfig& config, const RosNodeParams& params, Blackboard::Ptr blackboard)
        : BT::DecoratorNode(name, config), node_(params.nh.lock()), blackboard_(blackboard)
    {
        node_->get_parameter("collect_time", collectTime_);
        node_->get_parameter("place_time", placeTime_);
        node_->get_parameter("flip_time", flipTime_);
        node_->get_parameter("docking_time", dockingTime_);

        ackPub_ = node_->create_publisher<std_msgs::msg::Int64>("/robot/action_ack", 10);
        actionSub_ = node_->create_subscription<std_msgs::msg::Int64>(
            "/robot/action_cmd", 10, std::bind(&ModelClient::OnActionReceived, this, std::placeholders::_1));
        timeSub_ = node_->create_subscription<std_msgs::msg::Float32>(
            "/robot/game_time", 10, std::bind(&ModelClient::OnTimeReceived, this, std::placeholders::_1));
    }
    static PortsList providedPorts();
    NodeStatus tick() override;

private:
    rclcpp::Node::SharedPtr node_;
    BT::Blackboard::Ptr blackboard_;
    double gameTime_ = 0, lastGameTime_ = 0;
    double collectTime_, placeTime_, flipTime_, dockingTime_;
    int lastSeq_ = 0, stage_ = -1, lastStage_ = -1, action_ = -1, selectedArea_ = -1, entryPointId_ = -1, subTreeId_ = -1;
    double waitTimer_ = 0;
    bool isNavigating_ = false, newMsgFlag_ = false;
    std::map<int, std::map<int, std::pair<double, double>>> entryPointsConfig_; // area_id: <entry_id: <x, y>>
    geometry_msgs::msg::PoseStamped entryPoint_;

    rclcpp::Publisher<std_msgs::msg::Int64>::SharedPtr ackPub_;
    rclcpp::Subscription<std_msgs::msg::Int64>::SharedPtr actionSub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr timeSub_;

    void SetOutput();
    void OnTimeReceived(const std_msgs::msg::Float32::SharedPtr msg);
    void OnActionReceived(const std_msgs::msg::Int64::SharedPtr msg);
    void SendAck(int act_override = -1);
    BT::NodeStatus HandleNewActivation();
    void HandleChildCompletion(BT::NodeStatus status);
    void LoadEntryPoints();
};