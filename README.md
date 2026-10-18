# Autonomous Slalom & Home Return Robot with YOLOv8 and LiDAR

TurtleBot3 Waffle Pi (ROS 2 Humble / Gazebo) を用い、YOLOv8による物体認識とLiDARによる動的衝突回避、オドメトリを用いた自律帰還を行うタスクです。

## 動作要件・環境
- OS: Ubuntu 22.04 (Docker)
- ROS 2: Humble
- Simulator: Gazebo 11
- Robot: TurtleBot3 Waffle Pi
- Python Packages: `rclpy`, `ultralytics`, `opencv-python`, `cv_bridge`

## 実行手順・再現コマンド

### 1. シミュレータの起動 (Terminal 1)
```bash
export TURTLEBOT3_MODEL=waffle_pi
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

※ ロボットの位置を初期化したい場合は、Gazebo画面上で Shift + Ctrl + R を押す。
