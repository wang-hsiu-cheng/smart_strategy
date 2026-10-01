#include <bt_main/bt_action_nav.hpp>
#include <bt_main/bt_receiver.hpp>

using namespace BT;
using namespace std;

template <> inline geometry_msgs::msg::PoseStamped BT::convertFromString(StringView str) {
    auto parts = splitString(str, ',');
    if (parts.size() != 3) {
        throw RuntimeError("invalid input)");
    }
    else {
        geometry_msgs::msg::PoseStamped output;
        output.pose.position.x = convertFromString<double>(parts[0]);
        output.pose.position.y = convertFromString<double>(parts[1]);
        output.pose.position.z = convertFromString<double>(parts[2]);
        return output;
    }
}

template <> inline int BT::convertFromString(StringView str) {
    auto value = convertFromString<double>(str);
    return (int) value;
}

template <> inline std::deque<int> BT::convertFromString(StringView str) {
    auto parts = splitString(str, ',');
    std::deque<int> output;
    for (int i = 0; i < (int)parts.size(); i++) {
        output.push_back(convertFromString<int>(parts[i]));
    }
    return output;
}

double inline calculateDistance(const geometry_msgs::msg::Pose &pose1, const geometry_msgs::msg::Pose &pose2) {
    tf2::Vector3 position1(pose1.position.x, pose1.position.y, 0);
    tf2::Vector3 position2(pose2.position.x, pose2.position.y, 0);
    double dist = position1.distance(position2);
    // RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "distance: " << dist);
    return dist;
}

double inline calculateAngleDifference(const geometry_msgs::msg::Pose &pose1, const geometry_msgs::msg::Pose &pose2) {
    tf2::Quaternion orientation1, orientation2;
    tf2::fromMsg(pose1.orientation, orientation1);
    tf2::fromMsg(pose2.orientation, orientation2);
    double yaw1 = tf2::impl::getYaw(orientation1);
    double yaw2 = tf2::impl::getYaw(orientation2);
    return std::fabs(yaw1 - yaw2);
}

geometry_msgs::msg::PoseStamped inline ConvertPoseFormat(geometry_msgs::msg::PoseStamped pose_) {
    tf2::Quaternion quaternion;
    tf2::fromMsg(pose_.pose.orientation, quaternion);
    pose_.pose.position.z = tf2::impl::getYaw(quaternion) * 2 / PI;
    return pose_;
}

PortsList Navigation::providedPorts() {
    return {
        OutputPort<geometry_msgs::msg::PoseStamped>("goal"), // get entry_point
        OutputPort<geometry_msgs::msg::PoseStamped>("final_pose") // give final pose
    };
}

bool Navigation::setGoal(RosActionNode::Goal& goal) {
    auto goalPose = getInput<geometry_msgs::msg::PoseStamped>("goal");

    rclcpp::Time now = this->now();
    goal_.header.stamp = now;
    goal_.header.frame_id = "map";
    goal_.pose.position.x = goalPose.value().pose.position.x;
    goal_.pose.position.y = goalPose.value().pose.position.y;
    tf2::Quaternion q;
    q.setRPY(0, 0, goalPose.value().pose.position.z * PI / 2);
    goal_.pose.orientation.x = q.x();
    goal_.pose.orientation.y = q.y();
    goal_.pose.orientation.z = q.z();
    goal_.pose.orientation.w = q.w();
    goal_.pose.position.z = 0;

    goal.use_dock_id = false; // set use dock id
    goal.dock_pose = this->goal_; // send goal pose
    goal.dock_type = "dock_linearBoost_precise";
    goal.max_staging_time = 1000.0; // set max staging time
    goal.navigate_to_staging_pose = 1;  // if it's pure docking, then don't need to navigate to staging pose

    RCLCPP_INFO(logger(), "Start Nav to (%f, %f)", goal.dock_pose.pose.position.x, goal.dock_pose.pose.position.y);
    return true;
}

NodeStatus Navigation::onFeedback(const std::shared_ptr<const Feedback> feedback) {
    // nav_recov_times_ = feedback->number_of_recoveries;
    nav_recov_times_ = feedback->num_retries;
    if (nav_recov_times_ > 2) {
        // check the correctness of the final pose
        return goalErrorDetect();
    }
    // RCLCPP_INFO_STREAM(logger(), "current_pose: " << current_pose_.pose.position.x << ", " << current_pose_.pose.position.y);
    return NodeStatus::RUNNING;
}

NodeStatus Navigation::goalErrorDetect() {
    double nav_dist_error_ = node_->get_parameter("nav_dist_error").as_double();
    double nav_ang_error_ = node_->get_parameter("nav_ang_error").as_double();

    // check the correctness of the final pose
    LocReceiver::UpdateRobotPose(robot_pose_, tf_buffer_, frame_id_);
    if (calculateDistance(robot_pose_.pose, goal_.pose) < nav_dist_error_ && calculateAngleDifference(robot_pose_.pose, goal_.pose) < nav_ang_error_) {
        RCLCPP_INFO_STREAM(logger(), "success! final_pose: " << robot_pose_.pose.position.x << ", " << robot_pose_.pose.position.y << ", " << ConvertPoseFormat(robot_pose_).pose.position.z);
        RCLCPP_INFO_STREAM(logger(), "-----------------");
        setOutput<geometry_msgs::msg::PoseStamped>("final_pose", ConvertPoseFormat(robot_pose_));
        return NodeStatus::SUCCESS;
    } else {
        nav_error_ = true;
        RCLCPP_INFO_STREAM(logger(), "fail! final_pose: " << robot_pose_.pose.position.x << ", " << robot_pose_.pose.position.y << ", " << ConvertPoseFormat(robot_pose_).pose.position.z);
        RCLCPP_INFO_STREAM(logger(), "z" << ConvertPoseFormat(goal_).pose.position.z);
        RCLCPP_INFO_STREAM(logger(), "-----------------");
        setOutput<geometry_msgs::msg::PoseStamped>("final_pose", ConvertPoseFormat(goal_));
        return NodeStatus::SUCCESS;
    }
}

NodeStatus Navigation::onResultReceived(const WrappedResult& wr) {
    nav_finished_ = true;
    switch (wr.code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
        break;
    case rclcpp_action::ResultCode::ABORTED:
        nav_error_ = true;
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Goal was aborted");
    case rclcpp_action::ResultCode::CANCELED:
        nav_error_ = true;
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Goal was canceled");
        return NodeStatus::FAILURE;
    default:
        nav_error_ = true;
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Unknown result code");
        return NodeStatus::FAILURE;
    }
    return goalErrorDetect();
}

BT::PortsList Docking::providedPorts() {
    return { 
        InputPort<int>("direction"),                              // get entry_point_id as offset direction
        InputPort<geometry_msgs::msg::PoseStamped>("base"),       // get entry_point as starting point
        OutputPort<geometry_msgs::msg::PoseStamped>("final_pose") // give final pose
    };
}

bool Docking::setGoal(RosActionNode::Goal& goal) {
    auto basePose = getInput<geometry_msgs::msg::PoseStamped>("base"); // starting point for docking
    getInput<int>("direction", this->direction_);               // 0: 270deg, 1: 90deg, 2: 180deg, 3: 0deg
    blackboard_->set<bool>("Timeout", false);

    rclcpp::Time now = this->now();                            // get current time
    goal_.header.stamp = now;                                  // set header time
    goal_.header.frame_id = "map";                             // set header frame
    goal_.pose = basePose.value().pose;                               // calculate goal pose

    switch (this->direction_)
    {
    case 0:
        goal_.pose.position.y -= this->dockingOffset_;
        break;
    case 1:
        goal_.pose.position.y += this->dockingOffset_;
        break;
    case 2:
        goal_.pose.position.x -= this->dockingOffset_;
        break;
    case 3:
        goal_.pose.position.x += this->dockingOffset_;
        break;
    default:
        break;
    }
    goal_.pose.position.z = this->dockingOffset_;       // set offset distance
    tf2::Quaternion q;                                  // declare Quaternion
    q.setRPY(0, 0, basePose.value().pose.position.z * PI / 2); // change degree-z into Quaternion
    goal_.pose.orientation.x = q.x();
    goal_.pose.orientation.y = q.y();
    goal_.pose.orientation.z = q.z();
    goal_.pose.orientation.w = q.w();
    goal.use_dock_id = false;                           // set use dock id
    goal.dock_pose = this->goal_;                             // send goal pose
    goal.dock_type = "dock_linearBoost_precise";        // determine the docking direction (x or y)
    goal.max_staging_time = 1000.0;                     // set max staging time
    goal.navigate_to_staging_pose = 0;                  // if it's pure docking, then don't need to navigate to staging pose

    RCLCPP_INFO(logger(), "Start Docking to (%f, %f)", goal.dock_pose.pose.position.x, goal.dock_pose.pose.position.y);
    return true;
}

NodeStatus Docking::onFeedback(const std::shared_ptr<const Feedback> feedback) {
    nav_recov_times_ = feedback->num_retries;
    if (nav_recov_times_ > 2) {
        LocReceiver::UpdateRobotPose(robot_pose_, tf_buffer_, frame_id_);
        RCLCPP_INFO_STREAM(logger(), "success! final_pose: " << robot_pose_.pose.position.x << ", " << robot_pose_.pose.position.y << ", " << ConvertPoseFormat(robot_pose_).pose.position.z);
        RCLCPP_INFO_STREAM(logger(), "-----------------");
        setOutput<geometry_msgs::msg::PoseStamped>("final_pose", ConvertPoseFormat(robot_pose_));
        return NodeStatus::SUCCESS;
    }
    return NodeStatus::RUNNING;
}

NodeStatus Docking::goalErrorDetect() {
    double nav_dist_error_ = node_->get_parameter("nav_dist_error").as_double();
    double nav_ang_error_ = node_->get_parameter("nav_ang_error").as_double();
    blackboard_->set<bool>("enable_vision_check", true);

    // check the correctness of the final pose
    LocReceiver::UpdateRobotPose(robot_pose_, tf_buffer_, frame_id_);
    if (calculateDistance(robot_pose_.pose, goal_.pose) < nav_dist_error_ && calculateAngleDifference(robot_pose_.pose, goal_.pose) < nav_ang_error_) {
        RCLCPP_INFO_STREAM(logger(), "success! final_pose: " << robot_pose_.pose.position.x << ", " << robot_pose_.pose.position.y << ", " << ConvertPoseFormat(robot_pose_).pose.position.z);
        RCLCPP_INFO_STREAM(logger(), "-----------------");
        setOutput<geometry_msgs::msg::PoseStamped>("final_pose", ConvertPoseFormat(robot_pose_));
        return NodeStatus::SUCCESS;
    } else {
        nav_error_ = true;
        RCLCPP_INFO_STREAM(logger(), "fail! final_pose: " << robot_pose_.pose.position.x << ", " << robot_pose_.pose.position.y << ", " << ConvertPoseFormat(robot_pose_).pose.position.z);
        RCLCPP_INFO_STREAM(logger(), "-----------------");
        setOutput<geometry_msgs::msg::PoseStamped>("final_pose", ConvertPoseFormat(goal_));
        return NodeStatus::SUCCESS;
    }
}

NodeStatus Docking::onResultReceived(const WrappedResult& wr) {
    RCLCPP_INFO_STREAM(node_->get_logger(), "get dock result");
    nav_finished_ = true;
    if (!wr.result->success) {
        nav_error_ = true;
        nav_finished_ = true;
        if (wr.result->error_code == 905)
            blackboard_->set<bool>("Timeout", true);
        blackboard_->set<bool>("enable_vision_check", true);
        LocReceiver::UpdateRobotPose(robot_pose_, tf_buffer_, frame_id_);
        RCLCPP_INFO_STREAM(logger(), "error code: " << wr.result->error_code << " RETURN FAILURE! final_pose: " << robot_pose_.pose.position.x << ", " << robot_pose_.pose.position.y << ", " << robot_pose_.pose.position.z);
        RCLCPP_INFO_STREAM(logger(), "-----------------");
        return NodeStatus::FAILURE;
    }
    return goalErrorDetect();
}

BT::PortsList Rotation::providedPorts() {
    return {
        InputPort<geometry_msgs::msg::PoseStamped>("goal"),       // get entry point
        OutputPort<geometry_msgs::msg::PoseStamped>("final_pose") // update entry point
    };
}

bool Rotation::setGoal(RosActionNode::Goal& goal) {
    auto goalPose = getInput<geometry_msgs::msg::PoseStamped>("goal");// receive entry point, get z as the destinate direction
    double rad = goalPose.value().pose.position.z * PI / 2;
    
    // Get current actual robot pose to avoid error accumulation
    LocReceiver::UpdateRobotPose(robot_pose_, tf_buffer_, frame_id_);
    // Normalize angle to [-π, π] range
    while (rad > PI) rad -= 2 * PI;
    while (rad < -PI) rad += 2 * PI;

    rclcpp::Time now = this->now();
    goal_.header.stamp = now;
    goal_.header.frame_id = "map";
    goal_.pose = robot_pose_.pose;                                // Use current actual position
    goal_.pose.position.z = 0;                                    // set offset distance as 0
    tf2::Quaternion q;                                            // declare Quaternion
    q.setRPY(0, 0, rad);                                          // Use calculated target angle
    goal_.pose.orientation.x = q.x();
    goal_.pose.orientation.y = q.y();
    goal_.pose.orientation.z = q.z();
    goal_.pose.orientation.w = q.w();
    goal.use_dock_id = false;                                     // set use dock id
    goal.dock_pose = this->goal_;                                       // send goal pose
    goal.dock_type = "dock_slow_precise";                         // determine the docking direction (x or y)
    goal.max_staging_time = 2000.0;                               // set max staging time
    goal.navigate_to_staging_pose = 1;                            // if it's pure docking, then don't need to navigate to staging pose

    RCLCPP_INFO(logger(), "Start Rotating to angle: %f deg)", rad * 180 / PI);
    return true;
}

NodeStatus Rotation::goalErrorDetect() {
    double rotate_dist_error_ = node_->get_parameter("rotate_dist_error").as_double();
    double rotate_ang_error_ = node_->get_parameter("rotate_ang_error").as_double();

    LocReceiver::UpdateRobotPose(robot_pose_, tf_buffer_, frame_id_);
    // check the correctness of the final pose
    if (calculateDistance(robot_pose_.pose, goal_.pose) < rotate_dist_error_ && calculateAngleDifference(robot_pose_.pose, goal_.pose) < rotate_ang_error_) {
        nav_finished_ = true;
        RCLCPP_INFO(logger(), "success! final_direction: (%f, %f)", robot_pose_.pose.orientation.w, robot_pose_.pose.orientation.z);
        // RCLCPP_INFO_STREAM(logger(), "z" << ConvertPoseFormat(robot_pose_).pose.position.z);
        RCLCPP_INFO_STREAM(logger(), "-----------------");
        setOutput<geometry_msgs::msg::PoseStamped>("final_pose", ConvertPoseFormat(robot_pose_));
        return NodeStatus::SUCCESS;
    } else {
        nav_error_ = true;
        RCLCPP_INFO(logger(), "fail! final_direction: (%f, %f)", robot_pose_.pose.orientation.w, robot_pose_.pose.orientation.z);
        // RCLCPP_INFO_STREAM(logger(), "z" << ConvertPoseFormat(goal_).pose.position.z);
        RCLCPP_INFO_STREAM(logger(), "-----------------");
        setOutput<geometry_msgs::msg::PoseStamped>("final_pose", ConvertPoseFormat(goal_));
        return NodeStatus::SUCCESS;
    }
}

NodeStatus Rotation::onFeedback(const std::shared_ptr<const Feedback> feedback) {
    nav_recov_times_ = feedback->num_retries;
    if (nav_recov_times_ > 2) {
        return goalErrorDetect();       // check the correctness of the final pose
    }
    return NodeStatus::RUNNING;
}

NodeStatus Rotation::onResultReceived(const WrappedResult& wr) {
    if (wr.result->success)
        nav_finished_ = true;
    // check the correctness of the final pose
    return goalErrorDetect();
}