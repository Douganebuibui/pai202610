#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import time
import math

class SafeSlalomNavigator(Node):
    def __init__(self):
        super().__init__('safe_slalom_navigator')

        self.sub_img = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        self.bridge = CvBridge()
        self.model = YOLO('yolov8m.pt')

        self.state = 'PATROL'  # 'PATROL' -> 'RETURN' -> 'COMPLETE'
        self.start_time = time.time()
        self.patrol_duration = 45.0  # 巡回時間（秒）

        # センサー計測値
        self.dist_front = 10.0
        self.dist_left = 10.0
        self.dist_right = 10.0

        # 自己位置
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.start_x = None
        self.start_y = None

        # スタック防止用：回避動作維持タイマー
        self.avoid_until = 0.0
        self.avoid_dir = 1.0  # +1: 左旋回, -1: 右旋回

        self.get_logger().info('=== 自律巡回＆スタック防止帰還ノードを起動 ===')

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        self.current_x = pos.x
        self.current_y = pos.y

        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny, cosy)

        if self.start_x is None:
            self.start_x = self.current_x
            self.start_y = self.current_y

    def scan_callback(self, msg):
        ranges = msg.ranges
        n = len(ranges)
        if n == 0:
            return

        def get_min(start_deg, end_deg):
            i1 = int((start_deg % 360) * n / 360)
            i2 = int((end_deg % 360) * n / 360)
            if i1 <= i2:
                valid = [r for r in ranges[i1:i2] if 0.05 < r < 3.5]
            else:
                valid = [r for r in ranges[i1:] + ranges[:i2] if 0.05 < r < 3.5]
            return min(valid) if len(valid) > 0 else 10.0

        # 前方（正面±28度）、左側（30〜85度）、右側（275〜330度）
        self.dist_front = min(get_min(332, 360), get_min(0, 28))
        self.dist_left = get_min(30, 85)
        self.dist_right = get_min(275, 330)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # YOLOv8 推論実行
        results = self.model(frame, verbose=False)
        annotated_frame = results[0].plot()
        boxes = results[0].boxes

        twist = Twist()
        now = time.time()
        elapsed = now - self.start_time

        dist_to_home = 0.0
        if self.start_x is not None:
            dist_to_home = math.hypot(self.start_x - self.current_x, self.start_y - self.current_y)

        # ----------------------------------------------------
        # 走行制御
        # ----------------------------------------------------
        if self.state == 'PATROL':
            status_text = f"PATROL & SLALOM ({int(self.patrol_duration - elapsed)}s)"
            status_color = (0, 255, 0)

            if self.dist_front < 0.45:
                twist.linear.x = 0.02
                twist.angular.z = 0.45 if self.dist_left > self.dist_right else -0.45
            else:
                side_repulsion = 0.0
                if self.dist_left < 0.38:
                    side_repulsion -= 0.22
                if self.dist_right < 0.38:
                    side_repulsion += 0.22

                twist.linear.x = 0.12
                twist.angular.z = 0.25 * math.sin(elapsed * 0.8) + side_repulsion

            if elapsed >= self.patrol_duration:
                self.state = 'RETURN'
                self.get_logger().info('【巡回終了】初期位置への自律帰還を開始します')

        elif self.state == 'RETURN':
            if dist_to_home < 0.25:
                self.state = 'COMPLETE'
                self.get_logger().info('【帰還完了】スタート位置に到着しました')
            else:
                # 【脱出判定1】至近距離（32cm未満）に詰まった場合はバック
                if self.dist_front < 0.32:
                    status_text = "RETURNING: ESCAPING (BACKING)"
                    status_color = (0, 0, 255)
                    twist.linear.x = -0.06
                    twist.angular.z = 0.35 if self.dist_left > self.dist_right else -0.35
                    self.avoid_until = now + 1.5  # バック後も少し回避をキープ

                # 【脱出判定2】回避タイマー稼働中、または前方に障害物（48cm未満）
                elif now < self.avoid_until or self.dist_front < 0.48:
                    if self.dist_front < 0.48 and now >= self.avoid_until:
                        # 新たに障害物を検知した場合、広い方向を決めて2.0秒間回避をロック
                        self.avoid_dir = 1.0 if self.dist_left >= self.dist_right else -1.0
                        self.avoid_until = now + 2.0

                    status_text = f"RETURNING: CIRCLING AROUND (Dist: {dist_to_home:.2f}m)"
                    status_color = (0, 200, 255)
                    twist.linear.x = 0.08  # 停止させずに柱の横をすり抜けさせる
                    twist.angular.z = 0.45 * self.avoid_dir

                # 【通常帰還】障害物がなければホーム方向へ直進
                else:
                    status_text = f"RETURNING HOME (Dist: {dist_to_home:.2f}m)"
                    status_color = (0, 165, 255)

                    target_yaw = math.atan2(self.start_y - self.current_y, self.start_x - self.current_x)
                    angle_err = target_yaw - self.current_yaw
                    angle_err = (angle_err + math.pi) % (2.0 * math.pi) - math.pi

                    twist.linear.x = 0.11
                    twist.angular.z = float(max(min(0.85 * angle_err, 0.45), -0.45))

        elif self.state == 'COMPLETE':
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            status_text = "MISSION COMPLETE: HOME REACHED"
            status_color = (0, 255, 255)

        self.pub_cmd.publish(twist)

        # 画面HUD描画
        det_count = len(boxes)
        detect_text = f"YOLO: {det_count} OBSTACLES" if det_count > 0 else "YOLO: SEARCHING..."
        cv2.putText(annotated_frame, status_text, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, detect_text, (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"LIDAR: {self.dist_front:.2f}m", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

        cv2.imshow("YOLOv8 Tracking View", annotated_frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = SafeSlalomNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_twist = Twist()
        node.pub_cmd.publish(stop_twist)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()