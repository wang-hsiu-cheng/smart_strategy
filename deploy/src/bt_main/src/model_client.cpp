#include <bt_main/model_client.hpp>

using namespace BT;
using namespace std;

PortsList ModelClient::providedPorts() {
    return {
        OutputPort<bool>("is_navigating"),
        OutputPort<geometry_msgs::msg::PoseStamped>("entry_point"),
        OutputPort<int>("entry_point_id"),
        OutputPort<int>("sub_tree_id")
    };
}
void ModelClient::SetOutput() {
    setOutput<bool>("is_navigating", this->isNavigating_);  // output and then use if-else condition node to active navigation node
    setOutput<int>("entry_point_id", this->entryPointId_); // the direction of area                 0: 270deg, 1: 90deg, 2: 180deg, 3: 0deg
    setOutput<int>("sub_tree_id", this->subTreeId_);       // the direction of robot to do mission  0: 0deg,   1: 90deg, 2: 180deg, 3: 270deg
    // refactor entry point id to same representation as subtree id
    int idRefactored = (this->entryPointId_ == 0 || this->entryPointId_ == 3) ? 
                        abs(3 - this->entryPointId_) : this->entryPointId_;                     // 0: 0deg,   1: 90deg, 2: 180deg, 3: 270deg
    entryPoint_.pose.position.x = entryPointsConfig_[this->selectedArea_][this->entryPointId_].first;
    entryPoint_.pose.position.y = entryPointsConfig_[this->selectedArea_][this->entryPointId_].second;
    // the direction robot need to rotate to
    entryPoint_.pose.position.z = ((idRefactored - this->subTreeId_) + 4) % 4;                   // 0: 0deg,   1: 90deg, 2: 180deg, 3: 270deg
    setOutput<geometry_msgs::msg::PoseStamped>("entry_point", this->entryPoint_);
}

BT::NodeStatus ModelClient::tick() {
    // update timer every frame
    if (this->lastGameTime_ > 0 && this->waitTimer_ > 0) {
        double delta = this->gameTime_ - this->lastGameTime_;
        this->waitTimer_ = max(0.0, this->waitTimer_ - delta);
        blackboard_->set("wait_timer", this->waitTimer_);
    }
    this->lastGameTime_ = this->gameTime_;
    // if need stop navigation (action_ == 0), tick child to stop immediatly
    if (this->stage_ == 1 && this->action_ == 0 && this->isNavigating_) {
        blackboard_->set("cancel_nav", true);
        SetOutput(); // update output port before executing child node
        auto status = child_node_->executeTick();
        if (status != BT::NodeStatus::RUNNING) {
            HandleChildCompletion(status);
            child_node_->haltNode();
            return status;
        }
        return BT::NodeStatus::RUNNING;
    }
    // execute new action: when get new action from model_server, or every child node are idling
    if (this->newMsgFlag_ || child_node_->status() == BT::NodeStatus::IDLE) {
        if (!this->newMsgFlag_ && child_node_->status() == BT::NodeStatus::IDLE) { // means model_client is waiting for new action
            return BT::NodeStatus::RUNNING; 
        }
        this->newMsgFlag_ = false; // have received latest action from model server
        BT::NodeStatus status = HandleNewActivation(); // handle new action according to different stages
        // return to root if BT is not running
        // BT will restart from root
        if (status != BT::NodeStatus::RUNNING) {
            return status; 
        }
    }

    // execute current child node every frame
    SetOutput(); // update output port before executing child node
    auto status = child_node_->executeTick();
    // stop child node if needed
    if (status == BT::NodeStatus::SUCCESS || status == BT::NodeStatus::FAILURE) {
        HandleChildCompletion(status);
        child_node_->haltNode();
    }
    return status;
}

// timer callback
void ModelClient::OnTimeReceived(const std_msgs::msg::Float32::SharedPtr msg) { 
    this->gameTime_ = msg->data;
}
// model server callback
void ModelClient::OnActionReceived(const std_msgs::msg::Int64::SharedPtr msg) {
    long val = msg->data;
    int incomingSeq = val / 1000;

    // receive new action
    if (incomingSeq == this->lastSeq_ + 1) {
        this->lastSeq_ = incomingSeq;
        this->lastStage_ = this->stage_;
        this->stage_ = (val % 1000) / 100;
        this->action_ = val % 100;
        this->newMsgFlag_ = true; // tell ticker that we have new action to execute
    } 
    // receive current action
    else if (incomingSeq == this->lastSeq_) {
        this->SendAck(); // re-send ack to model server again
    }
}

// send ack packet to model servr: {lastSeq_ + 1}{stage_}{action_}
void ModelClient::SendAck(int act_override) {
    std_msgs::msg::Int64 ackMsg;
    int send_act = (act_override != 18) ? this->action_ : act_override;
    // 編碼：(目前序號+1)*1000 + 階段*100 + 動作
    ackMsg.data = (this->lastSeq_ + 1) * 1000 + this->stage_ * 100 + send_act;
    ackPub_->publish(ackMsg);
}

BT::NodeStatus ModelClient::HandleNewActivation() {
    if (stage_ == 0) {
        this->selectedArea_ = this->action_;
        SendAck(); // tell model server to give next action immediatly
        return BT::NodeStatus::SUCCESS;
    } 
    else if (this->stage_ == 1) {
        if (this->action_ != 0) {
            if (!isNavigating_) {
                blackboard_->set("cancel_nav", false);
                this->entryPointId_ = this->action_;
                isNavigating_ = true;
            }
            SendAck();
            SetOutput();
            return child_node_->executeTick(); 
        } else {
            blackboard_->set("cancel_nav", true);
            SetOutput();
            return child_node_->executeTick();
        }
    }
    else if (stage_ == 2) {
        bool isPantry = this->selectedArea_ < 10;
        int exitAction = (isPantry) ? 5 : 4;
        if (this->action_ == exitAction) {
            SendAck();
            return BT::NodeStatus::SUCCESS;
        }
        if (!isPantry) {
            this->waitTimer_ = this->collectTime_ + this->dockingTime_ * 2;
            this->subTreeId_ = 20 + this->action_;
        } else if (this->action_ == 0) {
            this->waitTimer_ = this->flipTime_ + this->dockingTime_ * 2;
            this->subTreeId_ = 10 + 4;
        } else {
            this->waitTimer_ = this->placeTime_ + this->dockingTime_ * 2;
            this->subTreeId_ = 10 + (this->action_-1);
        }
        SendAck();
        SetOutput();
        return child_node_->executeTick();
    }
    return BT::NodeStatus::FAILURE;
}

void ModelClient::HandleChildCompletion(BT::NodeStatus status) {
    if (this->stage_ == 1) {
        if (this->action_ == 0) { // navigation canceled
            SendAck();
            isNavigating_ = false;
        } else if (status == BT::NodeStatus::SUCCESS) { // navigation finished
            isNavigating_ = false;
            SendAck(18); // force model server change to stage2
        }
    } else if (stage_ == 2) {
        this->waitTimer_ = 0;
        blackboard_->set("wait_timer", this->waitTimer_);
        SendAck(); // mission finished. change to stage0
    }
}

void ModelClient::LoadEntryPoints() {
    for (int i = 0; i < 18; ++i) {
        std::string paramName = "entry_config.area_" + std::to_string(i);
        std::vector<double> rawList = node_->get_parameter(paramName).as_double_array();
        std::map<int, std::pair<double, double>> areaDict;
        for (size_t j = 0; j < rawList.size(); j += 3) {
            int entryId = static_cast<int>(rawList[j]);
            double x = rawList[j + 1];
            double y = rawList[j + 2];
            areaDict[entryId] = std::make_pair(x, y);
        }
        entryPointsConfig_[i] = areaDict;
    }
}