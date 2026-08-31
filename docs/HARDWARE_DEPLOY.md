# Hardware deployment

## Required confirmation

Before any real-hardware run, confirm the two FR3 IPs in
`config/workcell/current.yaml`, the device identities and Wuji addresses in
`config/devices.env`, and the tracker rigid transforms in `config/modes/pico.yaml`.
The migrated configuration preserves the source workcell values; it must not be
treated as proof that the currently connected hardware has the same identity.

## Staged validation

1. Verify that every hardware endpoint resolves through an intended robot or
   hand network interface, with a source address in the matching subnet:

   ```bash
   ip route get 172.16.0.2
   ip route get 172.16.1.2
   ip route get 192.168.1.110
   ip route get 192.168.2.111
   ```

   A default route, VPN, or proxy/TUN route is not valid. Depending on the
   physical wiring, use dedicated NICs, VLANs, or explicitly configured
   secondary addresses and routes. Never assign a host address already used by
   a robot or hand.
2. Run the unit tests and `ops/run/preflight.sh` with all hardware SDKs idle.
3. Inspect PICO tracker discovery with teleoperation stopped:

   ```bash
   conda run --no-capture-output -n pico-armhand-teleop \
     python adapters/pico/scripts/hardware/inspect_motion_trackers.py
   ```

4. If identities differ, update them with the calibration helper while
   teleoperation remains stopped:

   ```bash
   conda run --no-capture-output -n pico-armhand-teleop \
     python adapters/pico/scripts/hardware/calibrate_tracker_sides.py --write
   ```

5. Run fake FR3 with real PICO/MANUS/Wuji. Start disengaged and verify each
   side independently.
6. For real FR3, verify FCI, collision-threshold acceptance, external-torque
   topics, and safety-gateway contact gating before engaging.
7. Engage one arm only, make a small translation, disengage, then repeat for
   rotation and the other side. Validate both hands independently before a
   bimanual trial.

## Safety invariants

- Keep the physical emergency stop reachable.
- Never run two PICO clients or two MANUS clients at once.
- Never run another UDP arm bridge on ports 5560/5561.
- Preserve the 0.25 s freshness checks, 0.5 rad/s joint-speed limits, initial
  delta gate, contact gate, and controller command watchdog.
- GUI disengage is not a power cut. After a fault, save container logs before
  stopping the stack.
