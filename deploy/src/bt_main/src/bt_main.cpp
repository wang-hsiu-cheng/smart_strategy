#include <map>
#include <vector>
#include <string>

#include <behaviortree_ros2/bt_action_node.hpp>
#include "behaviortree_cpp/bt_factory.h"
#include "behaviortree_cpp/decorators/loop_node.h"
#include "behaviortree_cpp/xml_parsing.h"
#include "behaviortree_cpp/loggers/groot2_publisher.h"
#include "behaviortree_cpp/utils/shared_library.h"
#include "behaviortree_cpp/blackboard.h"

#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>

#include <bt_main/model_client.hpp>
#include <bt_main/bt_action_nav.hpp>
#include <bt_main/bt_action_firmware.hpp>
#include <bt_main/bt_receiver.hpp>

using namespace BT;
using namespace std;

class MainClient : public rclcpp::Node {
public:
    MainClient() : Node("main_client"), rate(100) {}
    void PublishGameTimer() {
        std_msgs::msg::Float32 time_msg;
        this->game_time = this->get_clock()->now().seconds() -  this->start_time;
        time_msg.data = this->game_time;
        time_pub_->publish(time_msg);
    }
    std::shared_ptr<rclcpp::Node> GetNode() {
        this->node_ = shared_from_this(); 
        return shared_from_this();       // Get a shared pointer to this node
    }
    void InitParam() { 
        collect_area_counts_.data = {4, 4, 4, 4, 4, 4, 4, 4};
        pantry_yellow_counts_.data = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        pantry_blue_counts_.data = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        held_count_.data = {0, 0, 0, 0};
        // Create a shared blackboard
        blackboard = BT::Blackboard::create();
        blackboard->set<std_msgs::msg::Int32MultiArray>("collect_area_counts", collect_area_counts_);
        blackboard->set<std_msgs::msg::Int32MultiArray>("pantry_yellow_counts", pantry_yellow_counts_);     
        blackboard->set<std_msgs::msg::Int32MultiArray>("pantry_blue_counts", pantry_blue_counts_);         
        blackboard->set<std_msgs::msg::Int32MultiArray>("held_count", held_count_);      
        blackboard->set<bool>("Timeout", false);
        blackboard->set<double>("wait_timer", wait_timer_);
        blackboard->set<double>("current_step", current_step_);  
        status_pub_ = node_->create_publisher<std_msgs::msg::Int32MultiArray>("/field/status", 10);
        time_pub_ = node_->create_publisher<std_msgs::msg::Float32>("/robot/game_time", 10);
        // Read parameters
        this->declare_parameter<std::string>("tree_file_path", "home/ted/rl_main/deploy/src/bt_main/bt_config/tree.xml");
        this->declare_parameter<std::string>("bt_nodes_file_path", "home/ted/rl_main/deploy/src/bt_main/bt_config/bt_nodes.xml");
        this->declare_parameter<std::string>("tree_name", "MainTree");
        this->declare_parameter<std::string>("frame_id", "base_link");
        this->declare_parameter<double>("nav_dist_error", 0.005);
        this->declare_parameter<double>("nav_ang_error", 0.4);
        this->declare_parameter<double>("rotate_dist_error", 0.005);
        this->declare_parameter<double>("rotate_ang_error", 0.4);
        this->declare_parameter<double>("safety_dist", 0.33);
        this->declare_parameter<double>("collect_time", 0);
        this->declare_parameter<double>("place_time", 0);
        this->declare_parameter<double>("flip_time", 0);
        this->declare_parameter<double>("docking_time", 0);
        this->declare_parameter<double>("docking_offset", 0);
        for (int i = 0; i < 18; ++i) {
            std::string param_name = "entry_config.area_" + std::to_string(i);
            this->declare_parameter<std::vector<double>>(param_name, std::vector<double>{});
        }
        // get parameters
        this->get_parameter("tree_file_path", tree_file_path);
        this->get_parameter("bt_nodes_file_path", bt_nodes_file_path);
        this->get_parameter("tree_name", tree_name);

        this->start_time = this->get_clock()->now().seconds();
        this->game_timer = this->create_wall_timer(50ms, std::bind(&MainClient::PublishGameTimer, this));
    }

    void CreateTreeNodes() {
        params.nh = this->node_;
        factory.registerNodeType<ModelClient>("ModelClient", params, blackboard);
        params.default_port_value = "dock_robot";  // ros action name for all three navigation BT actions
        factory.registerNodeType<Navigation>("Navigation", params);
        factory.registerNodeType<Docking>("Docking", params, blackboard);
        factory.registerNodeType<Rotation>("Rotation", params);
        factory.registerNodeType<Mission>("Mission", params, blackboard);   // need to update FieldState when success
    }

    void CreatTree() {
        RCLCPP_INFO_STREAM(this->get_logger(), "--Loading XML--");
        // register tree xml
        factory.registerBehaviorTreeFromFile(tree_file_path);                  // translate new tree nodes into xml language
        // add new tree nodes into xml
        xml_models = BT::writeTreeNodesModelXML(factory);
        std::ofstream file(bt_nodes_file_path);                                // open the xml that store the tree nodes
        file << xml_models;
        file.close();
        // create tree
        RCLCPP_INFO(this->node_->get_logger(), "--Create tree--");
        tree = factory.createTree(tree_name, blackboard);
    }

    void RunTheTree() {
        BT::NodeStatus status = BT::NodeStatus::RUNNING;
        RCLCPP_INFO(this->get_logger(), "[BT Application]: Behavior Tree start running!");
        do {
            status = tree.rootNode()->executeTick();
            rate.sleep();
            PublishFieldStatus();
        } while (rclcpp::ok() && status != BT::NodeStatus::FAILURE && this->game_time <= 100);
    }

private:
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr time_pub_;
    rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr status_pub_;
    BT::Blackboard::Ptr blackboard;
    std::shared_ptr<rclcpp::Node> node_;
    BT::Tree tree;
    rclcpp::Rate rate;
    std::string xml_models; // new tree nodes string
    // Behavior Tree Factory
    BT::BehaviorTreeFactory factory;
    BT::RosNodeParams params;
    // ROS msg
    std_msgs::msg::Int32MultiArray collect_area_counts_, pantry_yellow_counts_, pantry_blue_counts_, held_count_;
    rclcpp::TimerBase::SharedPtr game_timer;
    double wait_timer_, current_step_;
    double start_time, game_time;
    // Parameters
    std::string tree_file_path;
    std::string bt_nodes_file_path;
    std::string tree_name;

    void PublishFieldStatus() {
        auto msg = std_msgs::msg::Int32MultiArray();
        msg.data.resize(34);
        blackboard->get<std_msgs::msg::Int32MultiArray>("collect_area_counts", collect_area_counts_);   // *8
        blackboard->get<std_msgs::msg::Int32MultiArray>("pantry_yellow_counts", pantry_yellow_counts_); // *10
        blackboard->get<std_msgs::msg::Int32MultiArray>("pantry_blue_counts", pantry_blue_counts_);     // *10
        blackboard->get<std_msgs::msg::Int32MultiArray>("held_count", held_count_);                     // *4
        blackboard->get<double>("wait_timer", this->wait_timer_);                                       // *1
        // pass latest field info into ros msg
        if (collect_area_counts_.data.size() >= 8) 
            std::copy(collect_area_counts_.data.begin(), collect_area_counts_.data.begin() + 8, msg.data.begin() + 0);
        if (pantry_yellow_counts_.data.size() >= 10) 
            std::copy(pantry_yellow_counts_.data.begin(), pantry_yellow_counts_.data.begin() + 10, msg.data.begin() + 8);
        if (pantry_blue_counts_.data.size() >= 10) 
            std::copy(pantry_blue_counts_.data.begin(), pantry_blue_counts_.data.begin() + 10, msg.data.begin() + 18);
        if (held_count_.data.size() >= 4) 
            std::copy(held_count_.data.begin(), held_count_.data.begin() + 4, msg.data.begin() + 28);
        // convert sec. to 30Hz steps
        msg.data[32] = static_cast<int32_t>(this->wait_timer_ * 30.0);
        msg.data[33] = static_cast<int32_t>(this->game_time * 30.0);
        status_pub_->publish(msg);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::Rate rate(100);
    auto node = std::make_shared<MainClient>();
    node->GetNode();
    node->InitParam();
    node->CreateTreeNodes();
    std::thread spin_thread([&]() { rclcpp::spin(node); });
    node->CreatTree();
    node->RunTheTree();
    rclcpp::shutdown();
    return 0;
}