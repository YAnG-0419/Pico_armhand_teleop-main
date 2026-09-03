# PICO 臂手遥操作

本仓库用 PICO 动捕设备驱动两台 Franka FR3 机械臂，用 MANUS 手套驱动两只无际（Wuji）手。主机上的 Operator 进程不依赖 ROS；ROS 2 与实时 Franka 控制器运行在 host 网络的 Docker 容器中。

```text
PICO trackers -> 相对笛卡尔映射 -> 双臂 IK -> UDP
  -> pico_teleop_bridge -> ArmCommand -> 安全网关 -> 双 FR3

MANUS gloves -> 规范 landmarks -> Wuji 重定向 -> Wuji SDK
  -> 双 Wuji Hand 2
```

每一侧默认处于未使能状态。PICO 丢失、机器人状态过期、网关拒绝，或 GUI 断开，都会解除控制。只有 `teleop_core` 可以向 FR3 指令总线发命令。MANUS/Wuji 运行在工作线程中，不能阻塞 100 Hz 臂环。

## 目录结构

- `teleop_runtime`：统一 Operator 后端。
- `adapters/pico`：PICO SDK 输入、位姿映射、IK、UDP 客户端及测试。
- `adapters/manus`：原生 MANUS 骨架桥接与 Python 帧约定。
- `adapters/wuji`：MANUS 到 Wuji 的重定向、模型与硬件后端。
- `apps/operator_gui`：分侧臂、手使能 GUI。
- `apps/left_wuji_dataset_recorder`：同步的左臂/左手 episode 录制。
- `ros_ws/src`：ROS 消息、安全网关、UDP 桥与 FR3 控制器。
- `config`：PICO 跟踪、安全限位与工位拓扑。
- `ops`：环境安装、预检与启动入口。

## 安装

支持的主机：x86-64 Ubuntu 22.04。PICO XRoboToolkit PC Service 与 MANUS Core 是外部系统服务；它们的 Linux SDK 库已 vendored 到本仓库。

```bash
cp docker/.env.example docker/.env
./ops/setup/setup_teleop_env.sh
./adapters/manus/scripts/build.sh
./ops/setup/build.sh
```

若 Conda 安装在非默认路径，设置 `TELEOP_CONDA_EXE=/absolute/path/to/conda`。`setup_wuji_env.sh` 仍可作为幂等的 Wuji 依赖修复/检查脚本，但主安装流程已经会装上这些依赖。

编译或启动硬件前，请核对：

- `docker/.env`：数据路径、ROS domain、CPU 集合、容器工位配置路径。
- `config/workcell/current.yaml`：两台 FR3 的 IP 与专用 CPU 集合。
- `config/devices.env`：PICO tracker 身份与 Wuji 网络地址。
- `config/modes/pico.yaml`：跟踪限位与 tracker 到控制的变换。
- `adapters/manus/config` 下的 MANUS 标定文件。
- 显式传入的 `--wuji-left-address` 与 `--wuji-right-address`。

## 离线预检

```bash
./ops/run/preflight.sh
```

预检会检查文件、导入、ROS 编译、Docker 与 Compose，不会初始化 PICO、MANUS、Wuji 或 Franka 硬件。

## 运行（Motion Tracker）

启动 XRoboToolkit PC Service、PICO 头显应用和 MANUS Core。保持 XRoboToolkit 桌面 Demo 关闭，因为它会抢同一路数据流。然后运行：

```bash
./ops/run/start_teleop.sh
```

离线联调、使用假 FR3 硬件，但保留真实 PICO、MANUS、Wuji 设备时：

```bash
./ops/run/start_teleop.sh --fake-franka
```

若要为左 FR3 + Wuji Hand 2 打开运行时录制控件，请配置输出目录。录制器复用已有 Wuji 连接，不会发布指令：

```bash
mkdir -p /home/user/franka_teleop_data/datasets
./ops/run/start_teleop.sh \
  --wuji-sides left \
  --left-dataset-dir /home/user/franka_teleop_data/datasets
```

在 Operator GUI 中使用 **Start recording** 和 **Stop and save**。每次开始都会新建一个带时间戳的 episode，遥操作本身不会中断。

模式与校验清单见 [apps/left_wuji_dataset_recorder/README.md](apps/left_wuji_dataset_recorder/README.md)。

只测 PICO 机械臂、不接 Wuji 硬件时，可手动拉起栈并覆盖手部输入源：

```bash
cd docker
docker compose up -d fake-franka-control teleop-control pico-bridge
cd ..
./ops/run/run_operator.sh --hand-source none
```

## 网络与桥接参数

默认假设 Operator 与 Docker 跑在同一台电脑上。Compose 使用 `network_mode: host`，因此指令口 `127.0.0.1:5560` 与反馈口 `127.0.0.1:5561` 是刻意的。不要在这些端口上再起另一路桥。分机部署或非 host 网络需要显式重新设计端点；只改其中一个回环地址不够。

分阶段验证见 [docs/HARDWARE_DEPLOY.md](docs/HARDWARE_DEPLOY.md)。

---

## Pico Controller 遥操

### STEP 0：硬件启动

- Franka 及手的启动方式同上，确保硬件全部启动。
- 打开 Pico：VR 眼镜侧面开机，手柄放在眼镜下方操作，防止丢失。在 VR 眼镜里打开 XRoboToolKit。
- 打开工控机 APP：XRoboToolKit-PC-Service。
- Pico 连接：
  - Pico 和工控机连接同一 WIFI（小车路由器转发的 WIFI，名称 `car2`，密码 `12345678`），确保在同一网段。
  - 在 VR 眼镜里打开 XRoboToolKit，连接选择工控机 IP，勾选发送数据（勾选 Tracking 三个选项，勾选 send）。
  - 工控机 IP：`192.168.6.139`
- 关掉工控机上的 XRoboToolKit-PC-Service 界面 Demo（保留后台服务）。
- 关闭 demo 指令：

```bash
pkill -TERM -f RobotLinuxDemo.x86_64
```

检查：

```bash
pgrep -a -f RobotLinuxDemo.x86_64 || echo "Demo 已关闭"
pgrep -a RoboticsService
```

### STEP 1：遥操启动

**步骤一：进入工作空间**

```bash
cd /home/user/llx/Pico_armhand_teleop-main/
conda activate pico-armhand-teleop
```

**步骤二：Pico 连接**

1. VR 内打开 XRoboToolkit，连接工控机，发送数据。
2. 电脑打开 XRoboToolkit-PC-Service 确认连接。
3. 然后关掉界面：可以直接叉掉，也可以执行：

```bash
pkill -TERM -f RobotLinuxDemo.x86_64
```

确认已经关掉：

```bash
pgrep -a -f RobotLinuxDemo.x86_64 || echo "Demo 已关闭"
pgrep -a RoboticsService
```

期望：第一行提示 `Demo 已关闭`，第二行还能看到 `./RoboticsServiceProcess`。

**步骤三：启动全套遥操**

```bash
./ops/run/start_teleop.sh --arm-source controllers
```

只启动单侧：

```bash
./ops/run/start_teleop.sh --arm-source controllers --wuji-sides left
```

**步骤四：关机**

关掉 GUI，Docker 会自动关闭，等待一下。然后关掉 `RoboticsServiceProcess`：

```bash
pkill -TERM -f RoboticsServiceProcess
pkill -KILL -f RoboticsServiceProcess
```

可选检查：

```bash
pgrep -a RoboticsServiceProcess
```

### STEP 2：GUI 界面与踏板

**GUI 界面**

[Image]

- 左右手可以分开上下使能和复位。
- 可以记录手和臂的轨迹。

**踏板**

同上。

### STEP 3：注意事项

- 注意：是 **Controller**，不是 **Tracker**。这是两个东西，一个是手环，一个是手柄。
- 注意：要关闭 Linux Demo，否则会抢控制权。
- 注意：Pico 遥操速度幅度不要太大，不要快速突变，可能会有 IK 解算问题，小心大回环。
- 注意：手柄放在眼镜下方操作，防止丢失。VR 眼镜可以套在脖子上。
- 注意：遥操基坐标和 Pico 中 XRoboToolKit 启动时 UI 的朝向有关，推荐和 Franka 一致。
