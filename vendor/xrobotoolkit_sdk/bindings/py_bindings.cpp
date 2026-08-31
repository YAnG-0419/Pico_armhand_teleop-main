#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "PXREARobotSDK.h"


using json = nlohmann::json;

std::array<double, 7> LeftControllerPose;
std::array<double, 7> RightControllerPose;
std::array<double, 7> HeadsetPose;

std::array<std::array<double, 7>, 26> LeftHandTrackingState;
double LeftHandScale = 1.0;
int LeftHandIsActive = 0;
std::array<std::array<double, 7>, 26> RightHandTrackingState;
double RightHandScale = 1.0;
int RightHandIsActive = 0;

// Whole body motion data - 24 joints for body tracking
std::array<std::array<double, 7>, 24> BodyJointsPose;  // Position and rotation for each joint
std::array<std::array<double, 6>, 24> BodyJointsVelocity;  // Velocity and angular velocity for each joint
std::array<std::array<double, 6>, 24> BodyJointsAcceleration;  // Acceleration and angular acceleration for each joint
std::array<int64_t, 24> BodyJointsTimestamp;  // IMU timestamp for each joint
int64_t BodyTimeStampNs = 0;  // Body data timestamp
bool BodyDataAvailable = false;  // Flag to indicate if body data is available

std::vector<std::array<double, 7>> MotionTrackerPose;
std::vector<std::array<double, 6>> MotionTrackerVelocity;
std::vector<std::array<double, 6>> MotionTrackerAcceleration;
std::vector<std::string> MotionTrackerSerialNumbers;
int64_t MotionTimeStampNs = 0;  // Motion data timestamp
int NumMotionDataAvailable = 0;  // number of motion trackers
uint64_t MotionFrameSequence = 0;  // increments for every received Motion object
std::atomic<uint64_t> CallbackErrorCount{0};


bool LeftMenuButton;
double LeftTrigger;
double LeftGrip;
std::array<double, 2> LeftAxis{0.0, 0.0};
bool LeftAxisClick;
bool LeftPrimaryButton;
bool LeftSecondaryButton;

bool RightMenuButton;
double RightTrigger;
double RightGrip;
std::array<double, 2> RightAxis{0.0, 0.0};
bool RightAxisClick;
bool RightPrimaryButton;
bool RightSecondaryButton;

int64_t TimeStampNs;

std::mutex leftMutex;
std::mutex rightMutex;
std::mutex headsetPoseMutex;
std::mutex timestampMutex;
std::mutex leftHandMutex;
std::mutex rightHandMutex;
std::mutex bodyMutex;  // Mutex for body tracking data
std::mutex motionMutex;



template <std::size_t Size>
std::array<double, Size> stringToArray(const std::string& text) {
    std::array<double, Size> result{};
    std::stringstream ss(text);
    std::string value;
    std::size_t index = 0;
    while (std::getline(ss, value, ',')) {
        if (index == Size) {
            throw std::invalid_argument("too many comma-separated values");
        }
        std::size_t consumed = 0;
        const double parsed = std::stod(value, &consumed);
        while (consumed < value.size() && std::isspace(
                   static_cast<unsigned char>(value[consumed]))) {
            ++consumed;
        }
        if (consumed != value.size() || !std::isfinite(parsed)) {
            throw std::invalid_argument("invalid comma-separated number");
        }
        result[index++] = parsed;
    }
    if (index != Size) {
        throw std::invalid_argument("wrong comma-separated value count");
    }
    return result;
}

std::array<double, 7> stringToPoseArray(const std::string& poseStr) {
    return stringToArray<7>(poseStr);
}

std::array<double, 6> stringToVelocityArray(const std::string& velocityStr) {
    return stringToArray<6>(velocityStr);
}

void parseMotion(const json& value, int64_t frameTimeStampNs) {
    if (!value.contains("Motion")) {
        return;
    }
    const auto& motion = value["Motion"];
    const int64_t timestamp = motion.contains("timeStampNs")
        ? motion["timeStampNs"].get<int64_t>()
        : frameTimeStampNs;

    std::vector<std::array<double, 7>> poses;
    std::vector<std::array<double, 6>> velocities;
    std::vector<std::array<double, 6>> accelerations;
    std::vector<std::string> serials;
    if (motion.contains("joints") && motion["joints"].is_array()) {
        const auto& joints = motion["joints"];
        int count = static_cast<int>(joints.size());
        if (motion.contains("len")) {
            count = std::min(count, std::max(motion["len"].get<int>(), 0));
        }
        poses.reserve(count);
        velocities.reserve(count);
        accelerations.reserve(count);
        serials.reserve(count);
        for (int index = 0; index < count; ++index) {
            const auto& joint = joints[index];
            try {
                if (!joint.contains("p") || !joint.contains("sn")) {
                    continue;
                }
                const auto pose =
                    stringToPoseArray(joint["p"].get<std::string>());
                const auto serial = joint["sn"].get<std::string>();
                if (serial.empty()) {
                    continue;
                }
                std::array<double, 6> velocity{};
                std::array<double, 6> acceleration{};
                try {
                    if (joint.contains("va")) {
                        velocity = stringToVelocityArray(
                            joint["va"].get<std::string>());
                    }
                } catch (const std::exception&) {
                    ++CallbackErrorCount;
                }
                try {
                    if (joint.contains("wva")) {
                        acceleration = stringToVelocityArray(
                            joint["wva"].get<std::string>());
                    }
                } catch (const std::exception&) {
                    ++CallbackErrorCount;
                }
                poses.push_back(pose);
                velocities.push_back(velocity);
                accelerations.push_back(acceleration);
                serials.push_back(serial);
            } catch (const std::exception&) {
                ++CallbackErrorCount;
            }
        }
    }

    std::lock_guard<std::mutex> lock(motionMutex);
    MotionTimeStampNs = timestamp;
    MotionTrackerPose = std::move(poses);
    MotionTrackerVelocity = std::move(velocities);
    MotionTrackerAcceleration = std::move(accelerations);
    MotionTrackerSerialNumbers = std::move(serials);
    NumMotionDataAvailable = static_cast<int>(MotionTrackerPose.size());
    ++MotionFrameSequence;
}

void OnPXREAClientCallback(void* context, PXREAClientCallbackType type, int status, void* userData)
{
    switch (type)
    {
    case PXREAServerConnect:
        std::cout << "\nserver connect" << std::endl;
        break;
    case PXREAServerDisconnect:
        std::cout << "\nserver disconnect" << std::endl;
        break;
    case PXREADeviceFind:
        std::cout << "\ndevice found\n" << (const char*)userData << std::endl;
        break;
    case PXREADeviceMissing:
        std::cout << "\ndevice missing\n" << (const char*)userData << std::endl;
        break;
    case PXREADeviceConnect:
        std::cout << "\ndevice connect\n" << (const char*)userData << status << std::endl;
        break;
    case PXREADeviceStateJson:
        auto& dsj = *((PXREADevStateJson*)userData);


        try {
            json data = json::parse(dsj.stateJson);
            if (data.contains("value")) {
                auto value = json::parse(data["value"].get<std::string>());
                int64_t frameTimeStampNs = 0;
                if (value.contains("timeStampNs")) {
                    frameTimeStampNs = value["timeStampNs"].get<int64_t>();
                    std::lock_guard<std::mutex> lock(timestampMutex);
                    TimeStampNs = frameTimeStampNs;
                }
                // Arm-control data is independent of optional controller,
                // hand, and body fields. Publish it first so corruption in an
                // unrelated field cannot suppress a valid tracker frame.
                parseMotion(value, frameTimeStampNs);
                if (value["Controller"].contains("left")) {
                    auto& left = value["Controller"]["left"];
                    {
                        std::lock_guard<std::mutex> lock(leftMutex);
                        LeftControllerPose = stringToPoseArray(left["pose"].get<std::string>());
                        LeftTrigger = left["trigger"].get<double>();
                        LeftGrip = left["grip"].get<double>();
                        LeftMenuButton = left["menuButton"].get<bool>();
                        LeftAxis[0] = left["axisX"].get<double>();
                        LeftAxis[1] = left["axisY"].get<double>();
                        LeftAxisClick = left["axisClick"].get<bool>();
                        LeftPrimaryButton = left["primaryButton"].get<bool>();
                        LeftSecondaryButton = left["secondaryButton"].get<bool>();
                    }
                }
                if (value["Controller"].contains("right")) {
                    auto& right = value["Controller"]["right"];
                    {
                        std::lock_guard<std::mutex> lock(rightMutex);
                        RightControllerPose = stringToPoseArray(right["pose"].get<std::string>());
                        RightTrigger = right["trigger"].get<double>();
                        RightGrip = right["grip"].get<double>();
                        RightMenuButton = right["menuButton"].get<bool>();
                        RightAxis[0] = right["axisX"].get<double>();
                        RightAxis[1] = right["axisY"].get<double>();
                        RightAxisClick = right["axisClick"].get<bool>();
                        RightPrimaryButton = right["primaryButton"].get<bool>();
                        RightSecondaryButton = right["secondaryButton"].get<bool>();
                    }
                }
                if (value.contains("Head")) {
                    auto& headset = value["Head"];
                    {
                        std::lock_guard<std::mutex> lock(headsetPoseMutex);
                        HeadsetPose = stringToPoseArray(headset["pose"].get<std::string>());
                    }
                }
                if (value["Hand"].contains("leftHand")) {
                    auto& leftHand = value["Hand"]["leftHand"];
                    {
                        std::lock_guard<std::mutex> lock(leftHandMutex);

                        LeftHandScale = leftHand["scale"].get<double>();
                        LeftHandIsActive = leftHand["isActive"].get<int>();
                        auto& joints = leftHand["HandJointLocations"];
                        int jointCount = std::min(static_cast<int>(joints.size()), 26);
                        for (int i = 0; i < jointCount; i++) {
                            LeftHandTrackingState[i] = stringToPoseArray(joints[i]["p"].get<std::string>());
                        }
                    }
                }
                if (value["Hand"].contains("rightHand")) {
                    auto& rightHand = value["Hand"]["rightHand"];
                    {
                        std::lock_guard<std::mutex> lock(rightHandMutex);
                        RightHandScale = rightHand["scale"].get<double>();
                        RightHandIsActive = rightHand["isActive"].get<int>();
                        auto& joints = rightHand["HandJointLocations"];
                        int jointCount = std::min(static_cast<int>(joints.size()), 26);
                        for (int i = 0; i < jointCount; i++) {
                            RightHandTrackingState[i] = stringToPoseArray(joints[i]["p"].get<std::string>());
                        }
                    }
                }
                // Parse Body data for whole body motion capture
                if (value.contains("Body")) {
                    auto& body = value["Body"];
                    {
                        std::lock_guard<std::mutex> lock(bodyMutex);

                        if (body.contains("timeStampNs")) {
                            BodyTimeStampNs = body["timeStampNs"].get<int64_t>();
                        } else {
                            BodyTimeStampNs = frameTimeStampNs;
                        }

                        if (body.contains("joints") && body["joints"].is_array()) {
                            auto joints = body["joints"];
                            int jointCount = std::min(static_cast<int>(joints.size()), 24);

                            for (int i = 0; i < jointCount; i++) {
                                auto& joint = joints[i];

                                // Parse pose (position and rotation)
                                if (joint.contains("p")) {
                                    BodyJointsPose[i] = stringToPoseArray(joint["p"].get<std::string>());
                                }

                                // Parse velocity and angular velocity
                                if (joint.contains("va")) {
                                    BodyJointsVelocity[i] = stringToVelocityArray(joint["va"].get<std::string>());
                                }

                                // Parse acceleration and angular acceleration
                                if (joint.contains("wva")) {
                                    BodyJointsAcceleration[i] = stringToVelocityArray(joint["wva"].get<std::string>());
                                }

                                // Parse IMU timestamp
                                if (joint.contains("t")) {
                                    BodyJointsTimestamp[i] = joint["t"].get<int64_t>();
                                }
                            }

                            BodyDataAvailable = true;
                        } else {
                            BodyDataAvailable = false;
                        }
                    }
                }
            }
        } catch (const std::exception&) {
            // Never unwind through the vendor's C callback boundary. It can
            // silently kill the receive loop while leaving both the gRPC
            // channel and the Python process looking alive.
            ++CallbackErrorCount;
        } catch (...) {
            ++CallbackErrorCount;
        }
            break;
    }
}

void clearMotionData() {
    std::lock_guard<std::mutex> lock(motionMutex);
    MotionTrackerPose.clear();
    MotionTrackerVelocity.clear();
    MotionTrackerAcceleration.clear();
    MotionTrackerSerialNumbers.clear();
    MotionTimeStampNs = 0;
    NumMotionDataAvailable = 0;
    MotionFrameSequence = 0;
    CallbackErrorCount = 0;
}

void init() {
    clearMotionData();
    if (PXREAInit(NULL, OnPXREAClientCallback, PXREAFullMask) != 0) {
        std::cout << std::endl;
        throw std::runtime_error("PXREAInit failed");
    }
    std::cout << std::endl;
}

void deinit() {
    std::cout << std::endl;
    PXREADeinit();
    std::cout << std::endl;
}

std::array<double, 7> getLeftControllerPose() {
    std::lock_guard<std::mutex> lock(leftMutex);
    return LeftControllerPose;
}

std::array<double, 7> getRightControllerPose() {
    std::lock_guard<std::mutex> lock(rightMutex);
    return RightControllerPose;
}

std::array<double, 7> getHeadsetPose() {
    std::lock_guard<std::mutex> lock(headsetPoseMutex);
    return HeadsetPose;
}

double getLeftTrigger() {
    std::lock_guard<std::mutex> lock(leftMutex);
    return LeftTrigger;
}

double getLeftGrip() {
    std::lock_guard<std::mutex> lock(leftMutex);
    return LeftGrip;
}

double getRightTrigger() {
    std::lock_guard<std::mutex> lock(rightMutex);
    return RightTrigger;
}

double getRightGrip() {
    std::lock_guard<std::mutex> lock(rightMutex);
    return RightGrip;
}

bool getLeftMenuButton() {
    std::lock_guard<std::mutex> lock(leftMutex);
    return LeftMenuButton;
}

bool getRightMenuButton() {
    std::lock_guard<std::mutex> lock(rightMutex);
    return RightMenuButton;
}

bool getLeftAxisClick() {
    std::lock_guard<std::mutex> lock(leftMutex);
    return LeftAxisClick;
}

bool getRightAxisClick() {
    std::lock_guard<std::mutex> lock(rightMutex);
    return RightAxisClick;
}

std::array<double, 2> getLeftAxis() {
    std::lock_guard<std::mutex> lock(leftMutex);
    return LeftAxis;
}


std::array<double, 2> getRightAxis() {
    std::lock_guard<std::mutex> lock(rightMutex);
    return RightAxis;
}

bool getLeftPrimaryButton() {
    std::lock_guard<std::mutex> lock(leftMutex);
    return LeftPrimaryButton;
}

bool getRightPrimaryButton() {
    std::lock_guard<std::mutex> lock(rightMutex);
    return RightPrimaryButton;
}

bool getLeftSecondaryButton() {
    std::lock_guard<std::mutex> lock(leftMutex);
    return LeftSecondaryButton;
}

bool getRightSecondaryButton() {
    std::lock_guard<std::mutex> lock(rightMutex);
    return RightSecondaryButton;
}

int64_t getTimeStampNs() {
    std::lock_guard<std::mutex> lock(timestampMutex);
    return TimeStampNs;
}

std::array<std::array<double, 7>, 26> getLeftHandTrackingState() {
    std::lock_guard<std::mutex> lock(leftHandMutex);
    return LeftHandTrackingState;
}

int getLeftHandScale() {
    std::lock_guard<std::mutex> lock(leftHandMutex);
    return LeftHandScale;
}

int getLeftHandIsActive() {
    std::lock_guard<std::mutex> lock(leftHandMutex);
    return LeftHandIsActive;
}

std::array<std::array<double, 7>, 26> getRightHandTrackingState() {
    std::lock_guard<std::mutex> lock(rightHandMutex);
    return RightHandTrackingState;
}

int getRightHandScale() {
    std::lock_guard<std::mutex> lock(rightHandMutex);
    return RightHandScale;
}

int getRightHandIsActive() {
    std::lock_guard<std::mutex> lock(rightHandMutex);
    return RightHandIsActive;
}

// Body tracking functions
bool isBodyDataAvailable() {
    std::lock_guard<std::mutex> lock(bodyMutex);
    return BodyDataAvailable;
}

std::array<std::array<double, 7>, 24> getBodyJointsPose() {
    std::lock_guard<std::mutex> lock(bodyMutex);
    return BodyJointsPose;
}

std::array<std::array<double, 6>, 24> getBodyJointsVelocity() {
    std::lock_guard<std::mutex> lock(bodyMutex);
    return BodyJointsVelocity;
}

std::array<std::array<double, 6>, 24> getBodyJointsAcceleration() {
    std::lock_guard<std::mutex> lock(bodyMutex);
    return BodyJointsAcceleration;
}

std::array<int64_t, 24> getBodyJointsTimestamp() {
    std::lock_guard<std::mutex> lock(bodyMutex);
    return BodyJointsTimestamp;
}

int64_t getBodyTimeStampNs() {
    std::lock_guard<std::mutex> lock(bodyMutex);
    return BodyTimeStampNs;
}

int numMotionDataAvailable() {
    std::lock_guard<std::mutex> lock(motionMutex);
    return NumMotionDataAvailable;
}

std::vector<std::array<double, 7>> getMotionTrackerPose() {
    std::lock_guard<std::mutex> lock(motionMutex);
    return MotionTrackerPose;
}

std::vector<std::array<double, 6>> getMotionTrackerVelocity() {
    std::lock_guard<std::mutex> lock(motionMutex);
    return MotionTrackerVelocity;
}

std::vector<std::array<double, 6>> getMotionTrackerAcceleration() {
    std::lock_guard<std::mutex> lock(motionMutex);
    return MotionTrackerAcceleration;
}

std::vector<std::string> getMotionTrackerSerialNumbers() {
    std::lock_guard<std::mutex> lock(motionMutex);
    return MotionTrackerSerialNumbers;
}

int64_t getMotionTimeStampNs() {
    std::lock_guard<std::mutex> lock(motionMutex);
    return MotionTimeStampNs;
}

uint64_t getMotionFrameSequence() {
    std::lock_guard<std::mutex> lock(motionMutex);
    return MotionFrameSequence;
}

pybind11::tuple getMotionSnapshot() {
    std::lock_guard<std::mutex> lock(motionMutex);
    return pybind11::make_tuple(
        MotionFrameSequence,
        MotionTimeStampNs,
        MotionTrackerSerialNumbers,
        MotionTrackerPose
    );
}

uint64_t getCallbackErrorCount() {
    return CallbackErrorCount.load();
}

int DeviceControlJsonWrapper(const std::string& dev_id, const std::string& json_str) {
    const int rc = PXREADeviceControlJson(dev_id.c_str(), json_str.c_str());
    if (rc != 0) {
        throw std::runtime_error("device_control_json failed");
    }
    return rc; // 0
}

int SendBytesToDeviceWrapper(const std::string& dev_id, pybind11::bytes blob) {
    std::string s = blob;  // copy Python bytes to std::string
    const int rc = PXREASendBytesToDevice(dev_id.c_str(), s.data(), static_cast<unsigned>(s.size()));
    if (rc != 0) {
        throw std::runtime_error("send_bytes_to_device failed");
    }
    return rc; // 0
}


PYBIND11_MODULE(xrobotoolkit_sdk, m) {
    m.def("init", &init, "Initialize the PXREARobot SDK.");
    m.def("close", &deinit, "Deinitialize the PXREARobot SDK.");
    m.def("get_left_controller_pose", &getLeftControllerPose, "Get the left controller pose.");
    m.def("get_right_controller_pose", &getRightControllerPose, "Get the right controller pose.");
    m.def("get_headset_pose", &getHeadsetPose, "Get the headset pose.");
    m.def("get_left_trigger", &getLeftTrigger, "Get the left trigger value.");
    m.def("get_left_grip", &getLeftGrip, "Get the left grip value.");
    m.def("get_right_trigger", &getRightTrigger, "Get the right trigger value.");
    m.def("get_right_grip", &getRightGrip, "Get the right grip value.");
    m.def("get_left_menu_button", &getLeftMenuButton, "Get the left menu button state.");
    m.def("get_right_menu_button", &getRightMenuButton, "Get the right menu button state.");
    m.def("get_left_axis_click", &getLeftAxisClick, "Get the left axis click state.");
    m.def("get_right_axis_click", &getRightAxisClick, "Get the right axis click state.");
    m.def("get_left_axis", &getLeftAxis, "Get the left axis values (x, y).");
    m.def("get_right_axis", &getRightAxis, "Get the right axis values (x, y).");
    m.def("get_X_button", &getLeftPrimaryButton, "Get the left primary button state.");
    m.def("get_A_button", &getRightPrimaryButton, "Get the right primary button state.");
    m.def("get_Y_button", &getLeftSecondaryButton, "Get the left secondary button state.");
    m.def("get_B_button", &getRightSecondaryButton, "Get the right secondary button state.");
    m.def("get_time_stamp_ns", &getTimeStampNs, "Get the timestamp in nanoseconds.");
    m.def("get_left_hand_tracking_state", &getLeftHandTrackingState, "Get the left hand state.");
    m.def("get_right_hand_tracking_state", &getRightHandTrackingState, "Get the right hand state.");
    m.def("get_left_hand_is_active", &getLeftHandIsActive, "Get the left hand tracking quality (0 = low, 1 = high).");
    m.def("get_right_hand_is_active", &getRightHandIsActive, "Get the right hand tracking quality (0 = low, 1 = high).");

    // Body tracking functions
    m.def("is_body_data_available", &isBodyDataAvailable, "Check if body tracking data is available.");
    m.def("get_body_joints_pose", &getBodyJointsPose, "Get the body joints pose data (24 joints, 7 values each: x,y,z,qx,qy,qz,qw).");
    m.def("get_body_joints_velocity", &getBodyJointsVelocity, "Get the body joints velocity data (24 joints, 6 values each: vx,vy,vz,wx,wy,wz).");
    m.def("get_body_joints_acceleration", &getBodyJointsAcceleration, "Get the body joints acceleration data (24 joints, 6 values each: ax,ay,az,wax,way,waz).");
    m.def("get_body_joints_timestamp", &getBodyJointsTimestamp, "Get the body joints IMU timestamp data (24 joints).");
    m.def("get_body_timestamp_ns", &getBodyTimeStampNs, "Get the body data timestamp in nanoseconds.");

    // Motion tracker functions
    m.def("num_motion_data_available", &numMotionDataAvailable, "Check if motion tracker data is available.");
    m.def("get_motion_tracker_pose", &getMotionTrackerPose, "Get motion tracker poses (7 values each: x,y,z,qx,qy,qz,qw).");
    m.def("get_motion_tracker_velocity", &getMotionTrackerVelocity, "Get motion tracker velocities (6 values each: vx,vy,vz,wx,wy,wz).");
    m.def("get_motion_tracker_acceleration", &getMotionTrackerAcceleration, "Get motion tracker accelerations (6 values each: ax,ay,az,wax,way,waz).");
    m.def("get_motion_tracker_serial_numbers", &getMotionTrackerSerialNumbers, "Get the serial numbers of the motion trackers.");
    m.def("get_motion_timestamp_ns", &getMotionTimeStampNs, "Get the motion data timestamp in nanoseconds.");
    m.def("get_motion_frame_sequence", &getMotionFrameSequence, "Get the local sequence number of received Motion objects.");
    m.def("get_motion_snapshot", &getMotionSnapshot, "Atomically get motion sequence, timestamp, serials, and poses.");
    m.def("get_callback_error_count", &getCallbackErrorCount, "Get the number of rejected callback fields or frames.");


    // send json bytes functions
    m.def("device_control_json", &DeviceControlJsonWrapper, "Send a JSON control command to a device");
    m.def("send_bytes_to_device", &SendBytesToDeviceWrapper, "Send raw bytes to a device");

    m.doc() = "Python bindings for PXREARobot SDK using pybind11.";
}
