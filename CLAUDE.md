# Repository rules

- `teleop_core.contract` is the single source of truth for robot topics and joint names.
- PICO sources publish `teleop_interfaces/ArmCommand`; only the safety gateway may publish the FR3 command bus.
- Hardware output defaults to disabled. Never weaken freshness, acquisition, limit, contact, or slew checks to make a test pass.
- Keep PICO arm input, MANUS/Wuji hand control, safety/control, and operator UI independent.
- Never initialize a second PICO SDK or MANUS SDK client beside a running operator.
