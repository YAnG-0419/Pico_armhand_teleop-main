cmake_minimum_required(VERSION 3.16)
project(manus_skeleton_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(MANUS_SDK_ROOT "${REPOSITORY_ROOT}/vendor/manus_sdk")

add_library(manus_skeleton_bridge SHARED native/manus_skeleton_bridge.cpp)
target_include_directories(
  manus_skeleton_bridge
  PRIVATE native "${MANUS_SDK_ROOT}/include"
)
target_link_libraries(
  manus_skeleton_bridge
  PRIVATE "${MANUS_SDK_ROOT}/lib/libManusSDK_Integrated.so" pthread
)
set_target_properties(
  manus_skeleton_bridge
  PROPERTIES
    BUILD_RPATH "${MANUS_SDK_ROOT}/lib"
    OUTPUT_NAME manus_skeleton_bridge
)
target_compile_options(
  manus_skeleton_bridge PRIVATE -Wall -Wextra -Wpedantic -Werror
)
