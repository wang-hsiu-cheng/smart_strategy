#include <bt_main/bt_action_firmware.hpp>

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

template <> inline std::deque<double> BT::convertFromString(StringView str) {
    auto parts = splitString(str, ',');
    std::deque<double> output;
    for (int i = 0; i < (int)parts.size(); i++) {
        output.push_back(convertFromString<double>(parts[i]));
    }
    return output;
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

PortsList Mission::providedPorts() {
    return {
        InputPort<int>("mission_type"),     // get sub_tree_id as mission type
        OutputPort<int>("mission_status")   // return mission status
    };
}

BT::NodeStatus Mission::stopStep() {
    if (mission_status_ == 1 && mission_stamp_ == missionType_) {
        // RCLCPP_INFO(node_->get_logger(), "Mission success");
        blackboard_->set<int>("mission_progress", ++mission_progress_);
        setOutput<int>("mission_status", mission_status_);
        // mission_received_ = false;
        mission_status_ = 0;
        subscription_.reset();
        return BT::NodeStatus::SUCCESS;
    } else if (mission_status_ == 0 || (mission_status_ == 1 && mission_stamp_ != missionType_)) {
        return BT::NodeStatus::RUNNING;
    } else if (mission_status_ == -1) {
        // RCLCPP_INFO(node_->get_logger(), "Mission failed");
        setOutput<int>("mission_status", mission_status_);
        // mission_received_ = false;
        subscription_.reset();
        return BT::NodeStatus::FAILURE;
    } else {
        // RCLCPP_INFO(node_->get_logger(), "Unknown status code, Stop mission");
        setOutput<int>("mission_status", -1);
        subscription_.reset();
        return BT::NodeStatus::FAILURE;
    }
}
void Mission::mission_callback(const std_msgs::msg::Int32::SharedPtr sub_msg) {
    if (missionType_)
    {
        // if (sub_msg->data == 0)
            // mission_received_ = true;
        int temp_ = sub_msg->data;
        mission_stamp_ = temp_ / 10;
        mission_status_ = temp_ % 10;
        // RCLCPP_INFO(node_->get_logger(), "mission type: %d, return: %d, status: %d", missionType_, mission_stamp_, mission_status_);
    }
}

BT::NodeStatus Mission::onStart() {
    // RCLCPP_INFO(node_->get_logger(), "Node start");
    getInput<int>("mission_type", missionType_);
    if (!blackboard_->get<int>("mission_progress", mission_progress_)) {
        throw std::runtime_error("blackboard variable not found!");
    }
    // RCLCPP_INFO(node_->get_logger(), "mission_type: %d", missionType_);
    // RCLCPP_INFO(node_->get_logger(), "-----------------");
    return BT::NodeStatus::RUNNING;
}

BT::NodeStatus Mission::onRunning() {
    pub_msg.data = missionType_;
    publisher_->publish(pub_msg);
    // RCLCPP_INFO(node_->get_logger(), "mission_progress: %d", mission_progress_);
    // RCLCPP_INFO(node_->get_logger(), "mission_type: %d", missionType_);
    rate_.sleep();
    return stopStep();
    // **failure test**
    // if (mission_progress_ != 5)
    //    blackboard_->set<int>("mission_progress", ++mission_progress_);
    // return BT::NodeStatus::SUCCESS;
}

void Mission::onHalted() {
    // RCLCPP_INFO(node_->get_logger(), "Testing Node halted");
    setOutput<int>("mission_status", -1);
    return;
}