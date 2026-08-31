// This software contains source code provided by Manus Technology Group B.V.
// Adapted from the supplied litchi_hardware MANUS native bridge.

#include "litchi_manus_bridge.h"

#include "ManusSDK.h"
#include "ManusSDKTypeInitializers.h"
#include "ManusSDKTypes.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace
{
constexpr int32_t kLeft = 1;
constexpr int32_t kRight = 2;
constexpr uint32_t kLeftMask = 1U;
constexpr uint32_t kRightMask = 2U;
constexpr std::array<ChainType, 5> kFingerChainOrder = {
    ChainType_FingerThumb,
    ChainType_FingerIndex,
    ChainType_FingerMiddle,
    ChainType_FingerRing,
    ChainType_FingerPinky};

struct Quaternion
{
    float w;
    float x;
    float y;
    float z;
};

struct Vector3
{
    float x;
    float y;
    float z;
};

std::mutex g_lifecycle_mutex;
std::mutex g_state_mutex;
std::condition_variable g_frame_ready;
std::array<LitchiManusFrame, 2> g_latest_frames{};
std::array<bool, 2> g_has_frame{false, false};
std::unordered_map<uint32_t, int32_t> g_glove_sides;
std::unordered_map<
    uint32_t,
    std::array<uint32_t, LITCHI_MANUS_KEYPOINT_COUNT>>
    g_node_orders;
std::unordered_set<uint32_t> g_calibrated_gloves;
std::string g_calibration_directory;
std::string g_last_error;
std::thread g_calibration_thread;
std::atomic<bool> g_running{false};
bool g_sdk_initialized = false;
bool g_connected = false;
uint32_t g_available_side_mask = 0;
uint64_t g_sequence = 0;

void SetError(const std::string& operation, SDKReturnCode result)
{
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_last_error = operation + " returned SDK code " +
                   std::to_string(static_cast<int32_t>(result));
}

void SetError(const std::string& message)
{
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_last_error = message;
}

Quaternion Normalize(Quaternion value)
{
    const float norm = std::sqrt(value.w * value.w + value.x * value.x +
                                 value.y * value.y + value.z * value.z);
    if (norm <= 1.0e-12F)
    {
        return {1.0F, 0.0F, 0.0F, 0.0F};
    }
    return {value.w / norm, value.x / norm, value.y / norm, value.z / norm};
}

Quaternion Conjugate(const Quaternion& value)
{
    return {value.w, -value.x, -value.y, -value.z};
}

Quaternion Multiply(const Quaternion& left, const Quaternion& right)
{
    return {
        left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z,
        left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
        left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
        left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w};
}

Vector3 Rotate(const Quaternion& rotation, const Vector3& vector)
{
    const Quaternion point{0.0F, vector.x, vector.y, vector.z};
    const Quaternion rotated =
        Multiply(Multiply(rotation, point), Conjugate(rotation));
    return {rotated.x, rotated.y, rotated.z};
}

Quaternion FromManus(const ManusQuaternion& value)
{
    return Normalize({value.w, value.x, value.y, value.z});
}

bool IsFingerChain(ChainType chain_type)
{
    return std::find(
               kFingerChainOrder.begin(),
               kFingerChainOrder.end(),
               chain_type) != kFingerChainOrder.end();
}

bool BuildCanonicalNodeOrder(
    uint32_t glove_id,
    std::array<uint32_t, LITCHI_MANUS_KEYPOINT_COUNT>& node_order)
{
    uint32_t node_count = 0;
    SDKReturnCode result =
        CoreSdk_GetRawSkeletonNodeCount(glove_id, node_count);
    if (result != SDKReturnCode_Success)
    {
        SetError("CoreSdk_GetRawSkeletonNodeCount", result);
        return false;
    }

    std::vector<NodeInfo> node_info(node_count);
    for (NodeInfo& info : node_info)
    {
        NodeInfo_Init(&info);
    }
    result = CoreSdk_GetRawSkeletonNodeInfoArray(
        glove_id, node_info.data(), node_count);
    if (result != SDKReturnCode_Success)
    {
        SetError("CoreSdk_GetRawSkeletonNodeInfoArray", result);
        return false;
    }

    std::size_t output_index = 1;
    std::unordered_set<uint32_t> finger_node_ids;
    for (const ChainType chain_type : kFingerChainOrder)
    {
        std::vector<const NodeInfo*> chain;
        for (const NodeInfo& info : node_info)
        {
            if (info.chainType == chain_type)
            {
                chain.push_back(&info);
            }
        }

        const std::size_t expected_count =
            chain_type == ChainType_FingerThumb ? 4U : 5U;
        if (chain.size() != expected_count)
        {
            SetError(
                "Manus raw skeleton has an unexpected number of nodes for "
                "finger chain " +
                std::to_string(static_cast<int32_t>(chain_type)) +
                ": expected " + std::to_string(expected_count) +
                ", got " + std::to_string(chain.size()));
            return false;
        }

        std::sort(
            chain.begin(),
            chain.end(),
            [](const NodeInfo* left, const NodeInfo* right) {
                return left->fingerJointType < right->fingerJointType;
            });
        FingerJointType previous_joint = FingerJointType_Invalid;
        for (const NodeInfo* info : chain)
        {
            if (info->fingerJointType == FingerJointType_Invalid ||
                info->fingerJointType == previous_joint)
            {
                SetError(
                    "Manus raw skeleton contains invalid or duplicate finger "
                    "joint metadata");
                return false;
            }
            previous_joint = info->fingerJointType;
            node_order[output_index++] = info->nodeId;
            finger_node_ids.insert(info->nodeId);
        }
    }

    if (output_index != LITCHI_MANUS_KEYPOINT_COUNT)
    {
        SetError("Manus raw skeleton did not produce 24 canonical finger nodes");
        return false;
    }

    const NodeInfo* root = nullptr;
    for (const NodeInfo& info : node_info)
    {
        if (info.chainType == ChainType_Hand &&
            finger_node_ids.count(info.nodeId) == 0)
        {
            if (root != nullptr)
            {
                SetError("Manus raw skeleton contains multiple hand root nodes");
                return false;
            }
            root = &info;
        }
    }
    if (root == nullptr)
    {
        for (const NodeInfo& info : node_info)
        {
            if (!IsFingerChain(info.chainType) &&
                finger_node_ids.count(info.nodeId) == 0)
            {
                if (root != nullptr)
                {
                    SetError(
                        "Manus raw skeleton root is ambiguous in node metadata");
                    return false;
                }
                root = &info;
            }
        }
    }
    if (root == nullptr)
    {
        SetError("Manus raw skeleton has no hand root node");
        return false;
    }

    node_order[0] = root->nodeId;
    return true;
}

void OnLandscape(const Landscape* const landscape)
{
    if (landscape == nullptr || !g_running.load())
    {
        return;
    }

    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_glove_sides.clear();
    g_available_side_mask = 0;
    for (uint32_t index = 0; index < landscape->gloveDevices.gloveCount; ++index)
    {
        const GloveLandscapeData& glove = landscape->gloveDevices.gloves[index];
        if (glove.side == Side_Left)
        {
            g_glove_sides[glove.id] = kLeft;
            g_available_side_mask |= kLeftMask;
        }
        else if (glove.side == Side_Right)
        {
            g_glove_sides[glove.id] = kRight;
            g_available_side_mask |= kRightMask;
        }
    }
}

void OnRawSkeleton(const SkeletonStreamInfo* const stream)
{
    if (stream == nullptr || !g_running.load())
    {
        return;
    }

    for (uint32_t index = 0; index < stream->skeletonsCount; ++index)
    {
        RawSkeletonInfo info{};
        SDKReturnCode result = CoreSdk_GetRawSkeletonInfo(index, &info);
        if (result != SDKReturnCode_Success ||
            info.nodesCount < LITCHI_MANUS_KEYPOINT_COUNT)
        {
            continue;
        }

        int32_t side = 0;
        {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            const auto found = g_glove_sides.find(info.gloveId);
            if (found == g_glove_sides.end())
            {
                continue;
            }
            if (!g_calibration_directory.empty() &&
                g_calibrated_gloves.count(info.gloveId) == 0)
            {
                continue;
            }
            side = found->second;
        }

        std::vector<SkeletonNode> nodes(info.nodesCount);
        result = CoreSdk_GetRawSkeletonData(index, nodes.data(), info.nodesCount);
        if (result != SDKReturnCode_Success)
        {
            continue;
        }

        std::array<uint32_t, LITCHI_MANUS_KEYPOINT_COUNT> node_order{};
        bool has_node_order = false;
        {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            const auto found = g_node_orders.find(info.gloveId);
            if (found != g_node_orders.end())
            {
                node_order = found->second;
                has_node_order = true;
            }
        }
        if (!has_node_order)
        {
            if (!BuildCanonicalNodeOrder(info.gloveId, node_order))
            {
                continue;
            }
            std::lock_guard<std::mutex> lock(g_state_mutex);
            g_node_orders[info.gloveId] = node_order;
        }

        std::unordered_map<uint32_t, const SkeletonNode*> nodes_by_id;
        for (const SkeletonNode& node : nodes)
        {
            nodes_by_id[node.id] = &node;
        }
        const auto root_node = nodes_by_id.find(node_order[0]);
        if (root_node == nodes_by_id.end())
        {
            SetError("Manus raw skeleton frame is missing its hand root node");
            continue;
        }
        const SkeletonNode& root = *root_node->second;
        const Vector3 root_position{
            root.transform.position.x,
            root.transform.position.y,
            root.transform.position.z};
        const Quaternion root_rotation = FromManus(root.transform.rotation);
        const Quaternion inverse_root = Conjugate(root_rotation);
        constexpr float kHalfAngle = -0.7853981633974483F;
        const Quaternion basis_rotation{
            std::cos(kHalfAngle), 0.0F, std::sin(kHalfAngle), 0.0F};

        LitchiManusFrame frame{};
        frame.side = side;
        frame.keypoint_count = LITCHI_MANUS_KEYPOINT_COUNT;
        frame.timestamp_ns = stream->publishTime.time;

        bool valid = true;
        for (uint32_t output_index = 0;
             output_index < LITCHI_MANUS_KEYPOINT_COUNT;
             ++output_index)
        {
            const auto node = nodes_by_id.find(node_order[output_index]);
            if (node == nodes_by_id.end())
            {
                valid = false;
                break;
            }
            const SkeletonNode& skeleton_node = *node->second;
            const Vector3 delta{
                skeleton_node.transform.position.x - root_position.x,
                skeleton_node.transform.position.y - root_position.y,
                skeleton_node.transform.position.z - root_position.z};
            const Vector3 relative_position = Rotate(inverse_root, delta);
            const Vector3 position = Rotate(basis_rotation, relative_position);
            const Quaternion relative_rotation =
                Multiply(
                    inverse_root,
                    FromManus(skeleton_node.transform.rotation));
            // A basis change that sends p -> B*p must send R -> B*R. This
            // conjugated instead, B*R*B^-1, which rotates every node's own body
            // frame by 90 deg while leaving its position alone -- so position
            // and orientation stopped describing the same frame.
            //
            // It survived because the retargeter rebuilds its palm alignment
            // per frame from POSITIONS (retargeter.py _palm_align, whose note
            // claims any fixed rotation between data sources "drops out
            // entirely" -- true for positions, false for orientations): the
            // left B cancels and the stray B^-1 on the right does not. It then
            // reaches exactly one solver input, tgt_R, the only parameter
            // derived from node quaternions, and through it the tip_ori cost
            // term. Every other target (tgt_dir, tgt_dipdir, pinch_dir,
            // gap_tgt) is positions-only and was unaffected.
            //
            // Measured over a 10180-frame hardware session: with the operator's
            // hand flat the four fingertips were driven to 69-89 deg of DIP
            // flexion; consistent, they sit at 0.1-8.4 deg. Thumb-middle pad
            // facing 121 -> 161 deg, non-converged frames 1 -> 0, pinch closure
            // and thumb tracking unchanged.
            const Quaternion orientation = Normalize(
                Multiply(basis_rotation, relative_rotation));

            LitchiManusPose& output = frame.keypoints[output_index];
            output.position_x = position.x;
            output.position_y = position.y;
            output.position_z = position.z;
            output.orientation_x = orientation.x;
            output.orientation_y = orientation.y;
            output.orientation_z = orientation.z;
            output.orientation_w = orientation.w;
        }
        if (!valid)
        {
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            frame.sequence = g_sequence++;
            const std::size_t side_index = side == kLeft ? 0U : 1U;
            g_latest_frames[side_index] = frame;
            g_has_frame[side_index] = true;
            g_available_side_mask |= side == kLeft ? kLeftMask : kRightMask;
        }
        g_frame_ready.notify_all();
    }
}

bool ReadFile(const std::filesystem::path& path, std::vector<unsigned char>& data)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
    {
        return false;
    }
    const std::streamsize size = stream.tellg();
    if (size <= 0)
    {
        return false;
    }
    data.resize(static_cast<std::size_t>(size));
    stream.seekg(0);
    return static_cast<bool>(
        stream.read(reinterpret_cast<char*>(data.data()), size));
}

void CalibrationLoop()
{
    while (g_running.load())
    {
        std::vector<std::pair<uint32_t, int32_t>> pending;
        std::string directory;
        {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            directory = g_calibration_directory;
            for (const auto& item : g_glove_sides)
            {
                if (g_calibrated_gloves.count(item.first) == 0)
                {
                    pending.push_back(item);
                }
            }
        }

        for (const auto& item : pending)
        {
            if (directory.empty())
            {
                std::lock_guard<std::mutex> lock(g_state_mutex);
                g_calibrated_gloves.insert(item.first);
                continue;
            }
            const char* filename =
                item.second == kLeft ? "Calibration_left.mcal" : "Calibration_right.mcal";
            std::vector<unsigned char> data;
            if (!ReadFile(std::filesystem::path(directory) / filename, data))
            {
                continue;
            }
            SetGloveCalibrationReturnCode calibration_result{};
            const SDKReturnCode sdk_result = CoreSdk_SetGloveCalibration(
                item.first,
                data.data(),
                static_cast<uint32_t>(data.size()),
                &calibration_result);
            if (sdk_result == SDKReturnCode_Success)
            {
                std::lock_guard<std::mutex> lock(g_state_mutex);
                g_calibrated_gloves.insert(item.first);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void ResetState()
{
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_has_frame = {false, false};
    g_glove_sides.clear();
    g_node_orders.clear();
    g_calibrated_gloves.clear();
    g_available_side_mask = 0;
    g_sequence = 0;
}

int32_t FailAndShutdown(const std::string& operation, SDKReturnCode result)
{
    SetError(operation, result);
    g_running.store(false);
    if (g_sdk_initialized)
    {
        CoreSdk_ShutDown();
        g_sdk_initialized = false;
    }
    g_connected = false;
    return -1;
}
} // namespace

extern "C"
{
uint32_t litchi_manus_bridge_abi_version(void)
{
    return LITCHI_MANUS_BRIDGE_ABI_VERSION;
}

uint32_t litchi_manus_frame_size(void)
{
    return static_cast<uint32_t>(sizeof(LitchiManusFrame));
}

int32_t litchi_manus_connect(
    uint32_t discovery_wait_seconds,
    int32_t loopback_only,
    const char* calibration_directory)
{
    std::lock_guard<std::mutex> lifecycle_lock(g_lifecycle_mutex);
    if (g_connected)
    {
        SetError(
            "the Manus native SDK already has an active tracker in this process");
        return -1;
    }

    ResetState();
    {
        std::lock_guard<std::mutex> state_lock(g_state_mutex);
        g_last_error.clear();
        g_calibration_directory =
            calibration_directory == nullptr ? "" : calibration_directory;
    }

    SDKReturnCode result = CoreSdk_InitializeIntegrated();
    if (result != SDKReturnCode_Success)
    {
        return FailAndShutdown("CoreSdk_InitializeIntegrated", result);
    }
    g_sdk_initialized = true;
    g_running.store(true);

    result = CoreSdk_RegisterCallbackForRawSkeletonStream(OnRawSkeleton);
    if (result != SDKReturnCode_Success)
    {
        return FailAndShutdown(
            "CoreSdk_RegisterCallbackForRawSkeletonStream", result);
    }
    result = CoreSdk_RegisterCallbackForLandscapeStream(OnLandscape);
    if (result != SDKReturnCode_Success)
    {
        return FailAndShutdown(
            "CoreSdk_RegisterCallbackForLandscapeStream", result);
    }

    CoordinateSystemVUH coordinate_system{};
    CoordinateSystemVUH_Init(&coordinate_system);
    coordinate_system.handedness = Side_Right;
    coordinate_system.up = AxisPolarity_PositiveZ;
    coordinate_system.view = AxisView_XFromViewer;
    coordinate_system.unitScale = 1.0F;
    result =
        CoreSdk_InitializeCoordinateSystemWithVUH(coordinate_system, true);
    if (result != SDKReturnCode_Success)
    {
        return FailAndShutdown(
            "CoreSdk_InitializeCoordinateSystemWithVUH", result);
    }

    result = CoreSdk_LookForHosts(discovery_wait_seconds, loopback_only != 0);
    if (result != SDKReturnCode_Success)
    {
        return FailAndShutdown("CoreSdk_LookForHosts", result);
    }

    uint32_t host_count = 0;
    result = CoreSdk_GetNumberOfAvailableHostsFound(&host_count);
    if (result != SDKReturnCode_Success)
    {
        return FailAndShutdown(
            "CoreSdk_GetNumberOfAvailableHostsFound", result);
    }
    if (host_count == 0)
    {
        SetError("Manus SDK did not discover a host");
        g_running.store(false);
        CoreSdk_ShutDown();
        g_sdk_initialized = false;
        return -1;
    }

    std::vector<ManusHost> hosts(host_count);
    result = CoreSdk_GetAvailableHostsFound(hosts.data(), host_count);
    if (result != SDKReturnCode_Success)
    {
        return FailAndShutdown("CoreSdk_GetAvailableHostsFound", result);
    }
    result = CoreSdk_ConnectToHost(hosts[0]);
    if (result != SDKReturnCode_Success)
    {
        return FailAndShutdown("CoreSdk_ConnectToHost", result);
    }

    result = CoreSdk_SetRawSkeletonHandMotion(HandMotion_None);
    if (result != SDKReturnCode_Success)
    {
        return FailAndShutdown(
            "CoreSdk_SetRawSkeletonHandMotion", result);
    }

    g_connected = true;
    g_calibration_thread = std::thread(CalibrationLoop);
    return 0;
}

int32_t litchi_manus_disconnect(void)
{
    std::lock_guard<std::mutex> lifecycle_lock(g_lifecycle_mutex);
    g_running.store(false);
    g_frame_ready.notify_all();
    if (g_calibration_thread.joinable())
    {
        g_calibration_thread.join();
    }

    int32_t return_code = 0;
    if (g_sdk_initialized)
    {
        const SDKReturnCode result = CoreSdk_ShutDown();
        if (result != SDKReturnCode_Success)
        {
            SetError("CoreSdk_ShutDown", result);
            return_code = -1;
        }
    }
    g_sdk_initialized = false;
    g_connected = false;
    ResetState();
    return return_code;
}

int32_t litchi_manus_is_connected(void)
{
    std::lock_guard<std::mutex> lifecycle_lock(g_lifecycle_mutex);
    return g_connected ? 1 : 0;
}

int32_t litchi_manus_read_frame(
    int32_t side,
    uint32_t timeout_milliseconds,
    LitchiManusFrame* frame)
{
    if (frame == nullptr)
    {
        SetError("frame pointer must not be null");
        return -1;
    }
    if (side != kLeft && side != kRight)
    {
        SetError("side must be 1 (left) or 2 (right)");
        return -1;
    }

    const std::size_t side_index = side == kLeft ? 0U : 1U;
    std::unique_lock<std::mutex> lock(g_state_mutex);
    const bool ready = g_frame_ready.wait_for(
        lock,
        std::chrono::milliseconds(timeout_milliseconds),
        [side_index] {
            return g_has_frame[side_index] || !g_running.load();
        });
    if (!ready || !g_has_frame[side_index])
    {
        return 0;
    }
    std::memcpy(frame, &g_latest_frames[side_index], sizeof(*frame));
    g_has_frame[side_index] = false;
    return 1;
}

uint32_t litchi_manus_available_sides(void)
{
    std::lock_guard<std::mutex> lock(g_state_mutex);
    return g_available_side_mask;
}

const char* litchi_manus_last_error(void)
{
    std::lock_guard<std::mutex> lock(g_state_mutex);
    return g_last_error.c_str();
}
}
