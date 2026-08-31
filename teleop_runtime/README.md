# Teleop runtime

`cli.py` composes PICO arm input, the dual-FR3 UDP/safety path, the Operator
GUI protocol, and optional MANUS-to-Wuji hand control. Hardware starts
disengaged; arm and hand engagement remain independently controlled per side.
