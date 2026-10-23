# Autonomous GNC

Guidance, navigation and control for an autonomous ground vehicle.

| Folder | Purpose |
|---|---|
| docs/ | Architecture, CAN/UDP interface maps, bring-up procedures |
| hardware/ | Schematics, PCB, wiring for VCU, Mega, sensors |
| firmware/ | STM32F407 VCU (real-time, safety) and Arduino Mega (ultrasonic) |
| ros2_ws/ | ROS 2 workspace for perception, SLAM, localization, planning |
| ml/ | Datasets, training notebooks, trained models, evaluation |
| config/ | Sensor configs, LiDAR/radar/camera extrinsics, Nav2 and EKF parameters |
| sim/ | Simulation worlds and test scenarios |
| scripts/ | Setup, flashing, data logging helpers |
| tests/ | Unit, integration and hardware-in-the-loop tests |
