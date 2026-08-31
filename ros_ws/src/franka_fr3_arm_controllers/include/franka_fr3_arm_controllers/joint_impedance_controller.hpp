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

#pragma once

#include <Eigen/Eigen>
#include <array>
#include <controller_interface/controller_interface.hpp>
#include <cstdint>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <string>

#include "franka_fr3_arm_controllers/motion_generator.hpp"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_fr3_arm_controllers {

/**
 * Controller to move the robot to a desired joint position.
 */
class JointImpedanceController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  struct JointCommand {
    std::array<double, 7> positions{};
    rclcpp::Time source_stamp;
    rclcpp::Time receive_time;
    std::uint64_t sequence{0};
  };

  std::string arm_id_;
  std::string namespace_prefix_;
  std::string robot_description_;
  const int num_joints = 7;
  Vector7d q_;
  Vector7d dq_;
  Vector7d dq_filtered_;
  Vector7d hold_position_;
  Vector7d k_gains_;
  Vector7d d_gains_;
  double k_alpha_;
  double command_timeout_{0.25};
  double max_initial_target_delta_{0.35};
  // FR3 joint torque limits [N·m]: joints 1-4 @ 87 N·m, joints 5-7 @ 12 N·m
  const Vector7d tau_max_ = (Vector7d() << 87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0).finished();
  bool move_to_start_position_finished_{false};
  bool motion_generator_initialized_{false};
  bool initial_target_rejection_logged_{false};
  rclcpp::Time start_time_;
  std::unique_ptr<MotionGenerator> motion_generator_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscriber_ = nullptr;
  realtime_tools::RealtimeBuffer<JointCommand> command_buffer_;
  std::uint64_t callback_sequence_{0};
  std::uint64_t consumed_command_sequence_{0};
  bool gello_position_values_valid_ = false;
  std::array<double, 7> gello_position_values_{0, 0, 0, 0, 0, 0, 0};
  rclcpp::Time last_joint_state_time_;
  rclcpp::Time last_command_receive_time_;
  // First-order-hold interpolation of the command stream: q_goal ramps from
  // the goal applied when a command arrived (interp_from_) to that command's
  // target (interp_to_) over interp_duration_, evaluated each 1 kHz cycle.
  Vector7d interp_from_;
  Vector7d interp_to_;
  rclcpp::Time interp_started_at_;
  double interp_duration_{0.01};
  Vector7d applied_goal_;
  bool applied_goal_valid_{false};

  Vector7d calculateTauDGains_(const Vector7d& q_goal);
  bool validateGains_(const std::vector<double>& gains, const std::string& gains_name);
  bool initializeMotionGenerator_();
  void updateJointStates_();
  void validateGelloPositions_(const rclcpp::Time& source_stamp);
  void jointStateCallback_(const sensor_msgs::msg::JointState msg);
};

}  // namespace franka_fr3_arm_controllers
