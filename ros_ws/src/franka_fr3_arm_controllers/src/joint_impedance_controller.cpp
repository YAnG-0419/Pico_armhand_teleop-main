// Copyright (c) 2025 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <Eigen/Eigen>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <franka_fr3_arm_controllers/joint_impedance_controller.hpp>
#include <string>
#include <unordered_map>

using std::placeholders::_1;

namespace franka_fr3_arm_controllers {

controller_interface::InterfaceConfiguration
JointImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
JointImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" + std::to_string(i) +
                           "/position");
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" + std::to_string(i) +
                           "/velocity");
  }
  return config;
}

controller_interface::return_type JointImpedanceController::update(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {
  updateJointStates_();
  Vector7d q_goal;
  Vector7d tau_d_calculated;

  const auto* command = command_buffer_.readFromRT();
  if (command->sequence != consumed_command_sequence_) {
    gello_position_values_ = command->positions;
    validateGelloPositions_(command->source_stamp);
    // Stepping q_goal straight to the new target applies a torque step of
    // k_gains * delta at every command; ramping over one command interval
    // keeps the commanded torque continuous. Starting from the goal that was
    // actually applied (not the previous target) stays smooth when commands
    // arrive late or a ramp was still in flight.
    interp_from_ = applied_goal_valid_ ? applied_goal_ : q_;
    for (int i = 0; i < num_joints; ++i) {
      interp_to_(i) = gello_position_values_[i];
    }
    interp_duration_ = std::clamp(
        (command->receive_time - last_command_receive_time_).seconds(), 0.001, 0.02);
    interp_started_at_ = command->receive_time;
    last_command_receive_time_ = command->receive_time;
    consumed_command_sequence_ = command->sequence;
  }

  if (!motion_generator_initialized_) {
    motion_generator_initialized_ = initializeMotionGenerator_();

    if (!motion_generator_initialized_) {
      q_goal = hold_position_;
    }
  }

  if (motion_generator_initialized_ &&
      (!gello_position_values_valid_ ||
       (get_node()->now() - last_command_receive_time_).seconds() > command_timeout_)) {
    gello_position_values_valid_ = false;
    hold_position_ = q_;
    motion_generator_initialized_ = false;
    initial_target_rejection_logged_ = false;
    move_to_start_position_finished_ = false;
    motion_generator_.reset();
    q_goal = hold_position_;
  }

  if (motion_generator_initialized_ && !move_to_start_position_finished_) {
    // We have received valid joint states and initialized the motion generator
    // Now we move smoothly to the first joint position received from the input topic
    auto trajectory_time = this->get_node()->now() - start_time_;
    auto motion_generator_output = motion_generator_->getDesiredJointPositions(trajectory_time);
    move_to_start_position_finished_ = motion_generator_output.second;

    q_goal = motion_generator_output.first;
  }

  if (move_to_start_position_finished_) {
    // After reaching the start position we follow the joint position from the input topic
    // This is the normal operation mode of the controller
    const double elapsed = (get_node()->now() - interp_started_at_).seconds();
    const double alpha = std::clamp(elapsed / interp_duration_, 0.0, 1.0);
    q_goal = interp_from_ + alpha * (interp_to_ - interp_from_);
  }

  applied_goal_ = q_goal;
  applied_goal_valid_ = true;

  tau_d_calculated = calculateTauDGains_(q_goal);

  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_d_calculated(i));
  }

  return controller_interface::return_type::OK;
}

void JointImpedanceController::jointStateCallback_(const sensor_msgs::msg::JointState msg) {
  if (msg.name.size() != msg.position.size()) {
    RCLCPP_WARN(get_node()->get_logger(), "Rejected joint state with mismatched arrays.");
    return;
  }

  std::unordered_map<std::string, double> positions;
  for (size_t i = 0; i < msg.name.size(); ++i) {
    if (!std::isfinite(msg.position[i]) ||
        !positions.emplace(msg.name[i], msg.position[i]).second) {
      RCLCPP_WARN(get_node()->get_logger(), "Rejected non-finite or duplicate joint command.");
      return;
    }
  }
  std::array<double, 7> candidate{};
  for (int i = 0; i < num_joints; ++i) {
    const auto it = positions.find(arm_id_ + "_joint" + std::to_string(i + 1));
    if (it == positions.end()) {
      RCLCPP_WARN(get_node()->get_logger(), "Rejected incomplete joint command.");
      return;
    }
    candidate[i] = it->second;
  }

  JointCommand command;
  command.positions = candidate;
  command.source_stamp = msg.header.stamp;
  command.receive_time = get_node()->now();
  command.sequence = ++callback_sequence_;
  command_buffer_.writeFromNonRT(command);
}

CallbackReturn JointImpedanceController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "");
    auto_declare<std::vector<double>>("k_gains", {});
    auto_declare<std::vector<double>>("d_gains", {});
    auto_declare<double>("command_timeout", 0.25);
    auto_declare<double>("max_initial_target_delta", 0.35);
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  namespace_prefix_ = get_node()->get_namespace();
  if (namespace_prefix_ == "/" || namespace_prefix_.empty()) {
    namespace_prefix_.clear();
  } else {
    // Remove leading slash and add trailing underscore
    namespace_prefix_ = namespace_prefix_.substr(1) + "_";
  }

  auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  auto k_alpha = get_node()->get_parameter("k_alpha").as_double();
  command_timeout_ = get_node()->get_parameter("command_timeout").as_double();
  max_initial_target_delta_ = get_node()->get_parameter("max_initial_target_delta").as_double();

  if (!validateGains_(k_gains, "k_gains") || !validateGains_(d_gains, "d_gains")) {
    return CallbackReturn::FAILURE;
  }

  for (int i = 0; i < num_joints; ++i) {
    d_gains_(i) = d_gains.at(i);
    k_gains_(i) = k_gains.at(i);
  }

  if (k_alpha < 0.0 || k_alpha > 1.0) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_alpha should be in the range [0, 1]");
    return CallbackReturn::FAILURE;
  }

  k_alpha_ = k_alpha;

  dq_filtered_.setZero();

  auto parameters_client =
      std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");
  parameters_client->wait_for_service();

  auto future = parameters_client->get_parameters({"robot_description"});
  auto result = future.get();
  if (!result.empty()) {
    robot_description_ = result[0].value_to_string();
  } else {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
  }

  joint_state_subscriber_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
      "gello/joint_states", 1,
      [this](const sensor_msgs::msg::JointState& msg) { jointStateCallback_(msg); });

  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  updateJointStates_();
  if (!q_.allFinite() || !dq_.allFinite()) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Cannot activate controller with non-finite robot joint state.");
    return CallbackReturn::ERROR;
  }
  hold_position_ = q_;
  last_joint_state_time_ = get_node()->now();
  last_command_receive_time_ = get_node()->now();
  consumed_command_sequence_ = command_buffer_.readFromNonRT()->sequence;
  dq_filtered_.setZero();
  motion_generator_initialized_ = false;
  move_to_start_position_finished_ = false;
  gello_position_values_valid_ = false;
  applied_goal_valid_ = false;
  motion_generator_.reset();

  return CallbackReturn::SUCCESS;
}

auto JointImpedanceController::calculateTauDGains_(const Vector7d& q_goal) -> Vector7d {
  dq_filtered_ = (1 - k_alpha_) * dq_filtered_ + k_alpha_ * dq_;
  Vector7d tau_d_calculated;
  tau_d_calculated = k_gains_.cwiseProduct(q_goal - q_) + d_gains_.cwiseProduct(-dq_filtered_);

  // Clamp torques to FR3 hardware limits to prevent safety reflex / brake triggering
  tau_d_calculated = tau_d_calculated.cwiseMax(-tau_max_).cwiseMin(tau_max_);

  return tau_d_calculated;
}

bool JointImpedanceController::validateGains_(const std::vector<double>& gains,
                                              const std::string& gains_name) {
  if (gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "%s parameter not set", gains_name.c_str());
    return false;
  }

  if (gains.size() != static_cast<uint>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(), "%s should be of size %d but is of size %ld",
                 gains_name.c_str(), num_joints, gains.size());
    return false;
  }

  return true;
}

void JointImpedanceController::validateGelloPositions_(const rclcpp::Time& source_stamp) {
  const double max_time_diff = 0.5;
  auto current_time = get_node()->now();
  auto time_since_last_joint_state = (current_time - last_joint_state_time_).seconds();
  auto time_since_msg_stamp = (current_time - source_stamp).seconds();
  gello_position_values_valid_ =
      (time_since_last_joint_state < max_time_diff && time_since_msg_stamp >= 0.0 &&
       time_since_msg_stamp < max_time_diff);
  if (!gello_position_values_valid_) {
    RCLCPP_WARN(get_node()->get_logger(),
                "Gello position values are not valid. Time since last joint state: %f // Time "
                "since message stamp: %f",
                time_since_last_joint_state, time_since_msg_stamp);
  }
  last_joint_state_time_ = source_stamp;
}

void JointImpedanceController::updateJointStates_() {
  for (auto i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);

    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");

    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

bool JointImpedanceController::initializeMotionGenerator_() {
  if (!gello_position_values_valid_) {
    // Only send a warning once every 10 seconds in order not to spam the log
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 10 * 1000,
                         "Waiting for valid joint states...");
    return false;
  }

  Vector7d q_goal;
  updateJointStates_();
  for (int i = 0; i < num_joints; ++i) {
    q_goal(i) = gello_position_values_[i];
    if (std::abs(q_goal(i) - q_(i)) > max_initial_target_delta_) {
      if (!initial_target_rejection_logged_) {
        RCLCPP_ERROR(get_node()->get_logger(),
                     "Rejected initial target: joint %d delta %.3f exceeds %.3f rad.", i + 1,
                     std::abs(q_goal(i) - q_(i)), max_initial_target_delta_);
        initial_target_rejection_logged_ = true;
      }
      gello_position_values_valid_ = false;
      return false;
    }
  }
  const double motion_generator_speed_factor = 0.2;
  initial_target_rejection_logged_ = false;
  motion_generator_ = std::make_unique<MotionGenerator>(motion_generator_speed_factor, q_, q_goal);
  start_time_ = get_node()->now();
  return true;
}

}  // namespace franka_fr3_arm_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_fr3_arm_controllers::JointImpedanceController,
                       controller_interface::ControllerInterface)
