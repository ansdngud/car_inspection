import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener
import math
import time

DETECT_MAX_RANGE = 3.0
DETECT_MIN_RANGE = 0.20
CLUSTER_RADIUS = 2.0
ORBIT_MARGIN = 1.0
NUM_WAYPOINTS = 8
PAUSE_SEC = 3.0
CLOCKWISE = True
RETURN_HOME = True   # ? ?? ? ??? ?? ??

def get_yaw_from_quaternion(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class AutoOrbitNode(Node):
    def __init__(self):
        super().__init__('auto_orbit_node')
        self.scan = None
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

    def scan_cb(self, msg):
        self.scan = msg

    def get_robot_pose(self):
        for _ in range(30):
            rclpy.spin_once(self, timeout_sec=0.1)
        for _ in range(100):
            try:
                t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
                return (t.transform.translation.x,
                        t.transform.translation.y,
                        get_yaw_from_quaternion(t.transform.rotation))
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.1)
        return None

    def detect_car(self):
        print("???? ??? ?? ?...")
        for _ in range(50):
            if self.scan is not None:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.scan is None:
            print("??? ???? ? ??")
            return None

        pose = self.get_robot_pose()
        if pose is None:
            print("?? ??(map TF)? ? ??. ??? ?? ??? ? ?? ?????.")
            return None
        rx, ry, ryaw = pose
        print("?? ?? ??: (%.2f, %.2f)" % (rx, ry))
        self.robot_x, self.robot_y, self.robot_yaw = rx, ry, ryaw

        msg = self.scan
        candidates = []
        for i, r in enumerate(msg.ranges):
            if DETECT_MIN_RANGE < r < DETECT_MAX_RANGE:
                ang = msg.angle_min + i * msg.angle_increment
                a = ryaw + ang + math.pi
                candidates.append((r, rx + r*math.cos(a), ry + r*math.sin(a)))

        if len(candidates) < 5:
            print("?? ?? ?? (%d?)" % len(candidates))
            return None

        candidates.sort(key=lambda c: c[0])
        seed = candidates[0]
        print("?? ??? ?: ?? %.2fm, ?? (%.2f, %.2f)" % (seed[0], seed[1], seed[2]))

        cluster = [(p[1], p[2]) for p in candidates
                   if math.hypot(p[1]-seed[1], p[2]-seed[2]) < CLUSTER_RADIUS]
        if len(cluster) < 3:
            print("???? ?? ?? (%d?)" % len(cluster))
            return None

        sx = sum(p[0] for p in cluster) / len(cluster)
        sy = sum(p[1] for p in cluster) / len(cluster)

        max_span = 0.0
        for a in range(len(cluster)):
            for b in range(a+1, len(cluster)):
                d = math.hypot(cluster[a][0]-cluster[b][0], cluster[a][1]-cluster[b][1])
                if d > max_span:
                    max_span = d
        box_radius = max_span / 2.0
        if box_radius < 0.15:
            box_radius = 0.25
        print("?? ? ? ~ %.2fm -> ?? ?? ~ %.2fm" % (max_span, box_radius))

        dx, dy = sx - rx, sy - ry
        dist = math.hypot(dx, dy)
        if dist > 0.01:
            cx = sx + (dx/dist) * box_radius
            cy = sy + (dy/dist) * box_radius
        else:
            cx, cy = sx, sy
        print("?? ?? ??: (%.2f, %.2f) | ? %d?" % (cx, cy, len(cluster)))

        return cx, cy, box_radius

    def generate_waypoints(self, cx, cy, box_radius):
        orbit_r = box_radius + ORBIT_MARGIN
        print("?? ??: %.2fm (???? %.2f + ?? %.2f)" % (orbit_r, box_radius, ORBIT_MARGIN))

        raw = []
        for k in range(NUM_WAYPOINTS):
            theta = 2 * math.pi * k / NUM_WAYPOINTS
            raw.append((cx + orbit_r*math.cos(theta), cy + orbit_r*math.sin(theta)))

        rx, ry = self.robot_x, self.robot_y
        start_idx = min(range(NUM_WAYPOINTS),
                        key=lambda i: math.hypot(raw[i][0]-rx, raw[i][1]-ry))

        order = []
        for j in range(NUM_WAYPOINTS):
            if CLOCKWISE:
                idx = (start_idx - j) % NUM_WAYPOINTS
            else:
                idx = (start_idx + j) % NUM_WAYPOINTS
            order.append(idx)
        return [raw[i] for i in order]

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
        goal = NavigateToPose.Goal()
        goal.pose = self.make_pose(x, y, yaw)
        fut = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        gh = fut.result()
        if not gh.accepted:
            return False
        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        return True

    def run(self):
        print("Nav2 ?? ??...")
        self.client.wait_for_server()
        print("Nav2 ?? ??!")

        result = self.detect_car()
        if result is None:
            return
        cx, cy, box_radius = result

        # ?? ?? ?? (???)
        home_x, home_y, home_yaw = self.robot_x, self.robot_y, self.robot_yaw
        print("?? ?? ??: (%.2f, %.2f)" % (home_x, home_y))

        wps = self.generate_waypoints(cx, cy, box_radius)
        direction = "????" if CLOCKWISE else "?????"
        print("\n?? %d? ??! ?? ??? ??? %s ??\n" % (len(wps), direction))

        for i, (x, y) in enumerate(wps, start=1):
            yaw = math.atan2(cy - y, cx - x)
            print("[%d/%d] (%.2f, %.2f) ??" % (i, len(wps), x, y))
            ok = self.go_to(x, y, yaw)
            if ok:
                print("   ??! ??? ?? %.1f? ??" % PAUSE_SEC)
                time.sleep(PAUSE_SEC)
            else:
                print("   ?? ??, ????")
        print("\n??? ? ?? ?? ??!")

        # ??? ??
        if RETURN_HOME:
            print("\n?? ??? ?? ?... (%.2f, %.2f)" % (home_x, home_y))
            ok = self.go_to(home_x, home_y, home_yaw)
            if ok:
                print("?? ?? ?? ??! ?? ??.")
            else:
                print("?? ??.")

def main(args=None):
    rclpy.init(args=args)
    node = AutoOrbitNode()
    try:
        node.run()
    except KeyboardInterrupt:
        print("\n???")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
