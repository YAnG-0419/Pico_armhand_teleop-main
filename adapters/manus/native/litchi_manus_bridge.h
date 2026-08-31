#ifndef LITCHI_MANUS_BRIDGE_H
#define LITCHI_MANUS_BRIDGE_H

#include <stdint.h>

#if defined(__GNUC__)
#define LITCHI_MANUS_API __attribute__((visibility("default")))
#else
#define LITCHI_MANUS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define LITCHI_MANUS_BRIDGE_ABI_VERSION 1U
#define LITCHI_MANUS_KEYPOINT_COUNT 25U

typedef struct LitchiManusPose
{
    float position_x;
    float position_y;
    float position_z;
    float orientation_x;
    float orientation_y;
    float orientation_z;
    float orientation_w;
} LitchiManusPose;

typedef struct LitchiManusFrame
{
    int32_t side;
    uint32_t keypoint_count;
    uint64_t timestamp_ns;
    uint64_t sequence;
    LitchiManusPose keypoints[LITCHI_MANUS_KEYPOINT_COUNT];
} LitchiManusFrame;

LITCHI_MANUS_API uint32_t litchi_manus_bridge_abi_version(void);
LITCHI_MANUS_API uint32_t litchi_manus_frame_size(void);

/*
 * Start the in-process Manus SDK and connect to the first discovered host.
 * Returns zero on success and -1 on failure. Detailed failure text is
 * available from litchi_manus_last_error().
 */
LITCHI_MANUS_API int32_t litchi_manus_connect(
    uint32_t discovery_wait_seconds,
    int32_t loopback_only,
    const char* calibration_directory);

LITCHI_MANUS_API int32_t litchi_manus_disconnect(void);
LITCHI_MANUS_API int32_t litchi_manus_is_connected(void);

/*
 * Side is 1 for left and 2 for right. Returns 1 when a frame was copied,
 * 0 after a timeout, and -1 on failure.
 */
LITCHI_MANUS_API int32_t litchi_manus_read_frame(
    int32_t side,
    uint32_t timeout_milliseconds,
    LitchiManusFrame* frame);

/* Bit 0 is left and bit 1 is right. */
LITCHI_MANUS_API uint32_t litchi_manus_available_sides(void);

/*
 * The returned pointer remains valid until another bridge call changes the
 * error. Callers must copy it before invoking another bridge function.
 */
LITCHI_MANUS_API const char* litchi_manus_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
