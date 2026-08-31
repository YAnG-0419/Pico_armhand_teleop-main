# MANUS-to-Wuji integration

This backend converts calibrated MANUS skeleton landmarks to the official Wuji
Hand 2 `hand2_beta` models, reorders optimized joints into firmware order, and
sends commands through the Wuji SDK. Every side starts disengaged, stops on
stale MANUS frames, and is disabled during cleanup.

Install optional dependencies with:

```bash
./ops/setup/setup_wuji_env.sh
```

The full PICO/FR3/MANUS/Wuji entry point is `ops/run/start_teleop.sh`. Network
Wuji Hand 2 addresses come from `config/devices.env` and remain explicit so
discovery cannot swap sides. Exporting `WUJI_LEFT_ADDRESS` or
`WUJI_RIGHT_ADDRESS` before launch overrides the tracked defaults.

Simulation never connects to Wuji hardware:

```bash
conda run --no-capture-output -n pico-armhand-teleop \
  python -m adapters.wuji.sim --side right --headless --frames 300
```

MANUS-to-Wuji hand-only operation is also available, but it connects and enables
the selected physical hand during startup:

```bash
./ops/run/start_wuji_hand_only.sh --wuji-sides right
```

The hand-only entry point uses the same addresses from `config/devices.env`;
explicit `--wuji-*-address IP:PORT` arguments override them for one run.

Wuji Hand 2 defaults are `kp=3.0`, `kd=0.1`, and `1.5 A` per-joint current
limit. Change them only after hardware validation.
