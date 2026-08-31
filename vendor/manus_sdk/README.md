# MANUS CoreSDK 3.1.1

This directory vendors the x86-64 Linux MANUS CoreSDK needed by
`adapters/manus`. It was copied from the locally supplied
`ManusSDK_v3.1.1/SDKMinimalClient_Linux/ManusSDK` bundle so the teleop adapter
does not depend on a sibling checkout or a path under `Downloads`.

The SDK is proprietary and remains governed by `LICENSE.vendor`. The adapter
links `libManusSDK_Integrated.so`; MANUS Core/Robotics Service and its USB
device permissions are system dependencies.
