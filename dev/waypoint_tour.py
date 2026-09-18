import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
import time
import math

# 돌고 싶은 지점들 (x, y)
WAYPOINTS = [
    (1.12, -6.27),
    (2.52, -6.78),
    (3.84, -8.15),
    (1.83, -9.52),
    (0.64, -8.61),
]

# 자동차 중심 좌표 (각 지점에서 이쪽을 바라봄)
CAR_CENTER_X = 2.0
CAR_CENTER_Y = -7.87

# 각 지점 도착 후 정지 시간 (초)
PAUSE_SEC = 3.0

class WaypointTourNode(Node):
    def __init__(self):
        super().__init__('waypoint_tour_node')
        self.client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        print("🤖 Nav2 액션 서버 연결 대기 중...")
        self.client.wait_for_server()
        print("✅ Nav2 연결 완료!")

    def make_pose(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose

    def go_to(self, x, y, yaw):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.make_pose(x, y, yaw)

        send_goal_future = self.client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            print(f"   ❌ 목표 거부됨: ({x:.2f}, {y:.2f})")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return True

    def run_tour(self):
        print(f"\n🚗 총 {len(WAYPOINTS)}개 지점 순회 시작!")
        print(f"🎯 모든 지점에서 자동차 중심({CAR_CENTER_X:.2f}, {CAR_CENTER_Y:.2f})을 바라봄\n")
        for i, (x, y) in enumerate(WAYPOINTS, start=1):
            # 이 지점에서 자동차 중심을 바라보는 각도 계산
            yaw = math.atan2(CAR_CENTER_Y - y, CAR_CENTER_X - x)
            print(f"➡️  [{i}/{len(WAYPOINTS)}] ({x:.2f}, {y:.2f}) 이동 | 바라볼 각도: {math.degrees(yaw):.0f}°")
            ok = self.go_to(x, y, yaw)
            if ok:
                print(f"   ✅ 도착! 자동차 바라보고 {PAUSE_SEC}초 정지 (📸 촬영 자리)")
                time.sleep(PAUSE_SEC)
            else:
                print(f"   ⚠️ 이동 실패, 다음 지점으로")
        print("\n🏁 모든 지점 순회 완료!")

def main(args=None):
    rclpy.init(args=args)
    node = WaypointTourNode()
    try:
        node.run_tour()
    except KeyboardInterrupt:
        print("\n중단됨")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
