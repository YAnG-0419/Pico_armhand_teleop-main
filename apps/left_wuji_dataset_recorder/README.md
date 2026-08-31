# Left FR3 + Wuji Hand 2 Dataset Recorder

该记录器集成在现有遥操进程内，不会创建第二个 Wuji SDK 连接，也不会发布任何
机械臂或手部命令。GUI 每次点击开始记录会创建一个新的 episode，停止并保存不会
停止遥操。

最终 `.npz` 包含左 FR3 实测 7 关节、左 Wuji Hand 2 实测 20 关节、相对于
`lychee_root` 的 `left_fr3v2_link8` 正运动学位姿、臂/手 engage 状态，以及壁钟、
单调时钟、手反馈时刻和新鲜度。运行中同步保留 `<episode>.raw.jsonl`，每 100 帧
flush，便于异常退出恢复；正常退出时原子生成 NPZ，且不会覆盖已有文件。

```bash
mkdir -p /home/user/franka_teleop_data/datasets

ops/run/start_teleop.sh \
  --wuji-sides left \
  --left-dataset-dir /home/user/franka_teleop_data/datasets
```

启动后，在 Operator GUI 中点击 `Start recording` 开始，点击 `Stop and save`
结束并异步生成 NPZ。保存期间遥操保持运行；保存完成后可以再次开始新 episode。

兼容旧的一次性自动记录方式：传入
`--record-left-dataset /absolute/path/episode.npz` 后，控制循环启动即记录，遥操退出
时结束。

训练前应检查 `dropped_samples == 0`、四元数范数接近 1、关节值没有 NaN，并按
任务决定是否只保留 `arm_active & hand_active & hand_valid` 的帧。
