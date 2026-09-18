import json
import time
import os
import datetime
import math
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
from fairino import Robot as FairinoRobot
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

# ===== AGV 설정 =====
DETECT_MAX_RANGE = 3.0
DETECT_MIN_RANGE = 0.20
CLUSTER_RADIUS = 2.0
ORBIT_MARGIN = 0.6
NUM_WAYPOINTS = 8
CLOCKWISE = True
RETURN_HOME = True

# ===== 팔/카메라 설정 =====
CAM2EE_JSON = '/home/woohyung/cam2ee.json'
MODEL_PATH = '/home/woohyung/dent_best.pt'
ROBOT_IP = '192.168.58.2'
STANDOFF = 0.25          # 최종 촬영 거리 (m)
STANDOFF_FAR = 0.40      # 1차 접근 거리 (여기서 중앙 보정)
HOLE_SAMPLES = 20        # 탐지 프레임 수
DETECT_TIMEOUT = 25.0    # 각 지점에서 구멍 찾는 최대 시간 (초)
ARM_VEL = 20             # 팔 속도 (%)
CONF = 0.25              # YOLO 확신도 문턱 (탐지)
IMGSZ = 640              # 추론 해상도
SAVE_DIR = '/home/woohyung/hole_captures'
SAME_HOLE_DIST = 0.10    # 프레임 간 같은 구멍 매칭 허용 거리 (m)
NORMAL_TILT_LIMIT = 70.0 # 법선이 광축에서 이 각도(도) 이상 벗어나면 그 평면 거부
PLANE_MATCH_TOL = 0.06   # 구멍 주변 depth와 평면 교점 거리 허용 오차 (m)
MAX_PLANES = 2           # 한 시야에서 인식할 최대 평면 수
# =====================

def get_yaw_from_quaternion(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def ransac_plane(points, iters=60, thresh=0.005):
    """RANSAC 평면 피팅. 반환: (법선, 평면 위 점, 인라이어) 또는 (None,None,None)"""
    pts = np.array(points)
    n_pts = len(pts)
    if n_pts < 30:
        return None, None, None
    best_mask, best_count = None, 0
    for _ in range(iters):
        idx = np.random.choice(n_pts, 3, replace=False)
        p1, p2, p3 = pts[idx]
        n = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = np.abs((pts - p1) @ n)
        inliers = d < thresh
        c = int(inliers.sum())
        if c > best_count:
            best_count, best_mask = c, inliers
    if best_mask is None or best_count < 30:
        return None, None, None
    inl = pts[best_mask]
    centroid = inl.mean(axis=0)
    _, _, vh = np.linalg.svd(inl - centroid)
    return vh[2], centroid, best_mask

def find_planes(points, tilt_cos_limit, max_planes=MAX_PLANES):
    """순차 RANSAC으로 여러 평면 찾기. 각 평면: (법선(카메라쪽), 평면 위 점)"""
    planes = []
    pts = np.array(points)
    for _ in range(max_planes):
        if len(pts) < 30:
            break
        n, c, inl = ransac_plane(pts)
        if n is None:
            break
        if n[2] > 0:
            n = -n
        if -n[2] >= tilt_cos_limit:
            planes.append((n, c))
        pts = pts[~inl]
    return planes

# ==================== 팔 검사 모듈 ====================
class ArmInspector:
    def __init__(self):
        with open(CAM2EE_JSON) as f:
            calib = json.load(f)
        self.T_cam2ee = np.array(calib['T_cam2ee'])

        self.robot = FairinoRobot.RPC(ROBOT_IP)
        err, joints = self.robot.GetActualJointPosDegree()
        if err != 0:
            raise RuntimeError("FR5 연결 실패")
        self.home_joints = list(joints)   # 검사 시작 자세 (실행 순간의 자세)
        print("[팔] 연결 OK. 검사 시작 자세 저장:", ["%.1f" % j for j in self.home_joints])

        self.model = YOLO(MODEL_PATH, task='segment')
        os.makedirs(SAVE_DIR, exist_ok=True)
        self.tilt_cos_limit = float(np.cos(np.radians(NORMAL_TILT_LIMIT)))

    def get_T_ee2base(self):
        err, pose = self.robot.GetActualTCPPose()
        if err != 0:
            return None
        x, y, z, rx, ry, rz = pose
        t = np.array([x, y, z]) / 1000.0
        R = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def center_refine(self, target_base, max_iters=3, tol_m=0.02):
        """도착 후 보정: 목표 구멍 좌표를 화면 픽셀로 역투영해서 해당 마스크를 찾고,
        화면 중앙에 오도록 팔을 실제로 이동. (평면 불필요, 깜빡임 완화 포함)"""
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        try:
            for it in range(max_iters):
                off_list = []
                n_nomask, n_far = 0, 0
                for _ in range(12):
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    color_image = np.asanyarray(color_frame.get_data())
                    results = self.model(color_image, conf=0.15, device='cpu', imgsz=IMGSZ, verbose=False)
                    r = results[0]
                    annotated_dbg = results[0].plot()
                    if r.masks is None or len(r.masks) == 0:
                        n_nomask += 1
                        cv2.putText(annotated_dbg, "REFINE: no mask", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.imshow('Refine view', annotated_dbg)
                        cv2.waitKey(1)
                        continue

                    T_ee2base = self.get_T_ee2base()
                    if T_ee2base is None:
                        continue
                    T_cam2base = T_ee2base @ self.T_cam2ee

                    # 목표 구멍을 카메라 화면 픽셀로 역투영
                    p_t = np.array([target_base[0], target_base[1], target_base[2], 1.0])
                    p_t_cam = (np.linalg.inv(T_cam2base) @ p_t)[:3]
                    if p_t_cam[2] < 0.05:
                        continue
                    intr = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                    u_t = intr.fx * p_t_cam[0] / p_t_cam[2] + intr.ppx
                    v_t = intr.fy * p_t_cam[1] / p_t_cam[2] + intr.ppy

                    # 예상 픽셀과 가장 가까운 마스크 = 목표 구멍
                    best_pd, best_uv = 1e9, None
                    for i in range(len(r.masks)):
                        m = cv2.resize(r.masks.data[i].cpu().numpy(), (640, 480))
                        ys, xs = np.where(m > 0.5)
                        if len(xs) < 50:
                            continue
                        cu, cv_ = int(xs.mean()), int(ys.mean())
                        pd = ((cu - u_t) ** 2 + (cv_ - v_t) ** 2) ** 0.5
                        if pd < best_pd:
                            best_pd, best_uv = pd, (cu, cv_)

                    cv2.drawMarker(annotated_dbg, (int(u_t), int(v_t)), (255, 0, 0),
                                   cv2.MARKER_CROSS, 30, 3)
                    if best_uv is not None:
                        cv2.circle(annotated_dbg, best_uv, 8, (0, 255, 0), 2)
                        cv2.putText(annotated_dbg, "pd=%.0fpx" % best_pd, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imshow('Refine view', annotated_dbg)
                    cv2.waitKey(1)

                    if best_uv is None or best_pd > 200.0:
                        n_far += 1
                        continue

                    cu, cv_ = best_uv
                    z = float(p_t_cam[2])
                    off_x = (cu - intr.ppx) / intr.fx * z
                    off_y = (cv_ - intr.ppy) / intr.fy * z
                    off_list.append((off_x, off_y))

                if len(off_list) < 2:
                    print("   [보정] 목표 구멍이 안 보임, 보정 생략 (마스크없음 %d / 픽셀불일치 %d / 유효 %d)"
                          % (n_nomask, n_far, len(off_list)))
                    try:
                        cv2.destroyWindow('Refine view')
                    except Exception:
                        pass
                    return
                off_x = float(np.median([o[0] for o in off_list]))
                off_y = float(np.median([o[1] for o in off_list]))
                err_norm = (off_x ** 2 + off_y ** 2) ** 0.5
                print("   [보정 %d] 중앙 오차: X=%.3f Y=%.3f (%.3fm)" % (it + 1, off_x, off_y, err_norm))
                if err_norm < tol_m:
                    print("   [보정] 중앙 정렬 완료!")
                    try:
                        cv2.destroyWindow('Refine view')
                    except Exception:
                        pass
                    return
                T_ee2base = self.get_T_ee2base()
                if T_ee2base is None:
                    return
                R_cam2base = (T_ee2base @ self.T_cam2ee)[:3, :3]
                moved_ok = False
                scale = 1.0
                for _try in range(2):   # IK 실패 시 절반 이동 재시도
                    delta_base = R_cam2base @ np.array([off_x * scale, off_y * scale, 0.0])
                    err_p, cur_pose = self.robot.GetActualTCPPose()
                    if err_p != 0:
                        return
                    new_pose = list(cur_pose)
                    new_pose[0] += delta_base[0] * 1000.0
                    new_pose[1] += delta_base[1] * 1000.0
                    new_pose[2] += delta_base[2] * 1000.0
                    ik = self.robot.GetInverseKin(0, new_pose, -1)
                    if isinstance(ik, tuple) and ik[0] == 0:
                        jt = list(ik[1])
                        err_j, cur_j = self.robot.GetActualJointPosDegree()
                        if err_j == 0:
                            jt[5] = cur_j[5]   # J6 고정 유지
                        self.robot.MoveJ(jt, 0, 0, vel=ARM_VEL)
                        moved_ok = True
                        break
                    else:
                        scale = 0.5
                        print("   [보정] IK 실패, 절반 이동 재시도...")
                if not moved_ok:
                    print("   [보정] 이동 불가, 보정 중단")
                    return
        finally:
            pipeline.stop()
            try:
                cv2.destroyWindow('Refine view')
            except Exception:
                pass

    def inspect_and_capture(self, waypoint_idx):
        """현재 위치에서 모든 구멍 탐지(다중 평면) -> 구멍마다
        40cm 접근 -> 중앙 보정 -> 15cm 전진 -> 촬영 -> 복귀. 촬영 개수 반환."""
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        hole_clusters = []
        t_start = time.time()
        frames_collected = 0
        n_captured = 0

        try:
            # ===== 탐지: 다중 평면 RANSAC + 광선-평면 교점 =====
            while frames_collected < HOLE_SAMPLES and (time.time() - t_start) < DETECT_TIMEOUT:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())

                results = self.model(color_image, conf=CONF, device='cpu', imgsz=IMGSZ, verbose=False)
                annotated = results[0].plot()
                r = results[0]

                status = ""

                if r.masks is None or len(r.masks) == 0:
                    status = "no mask"
                else:
                    all_masks = np.zeros((480, 640), dtype=np.uint8)
                    mask_list = []
                    for i in range(len(r.masks)):
                        m = cv2.resize(r.masks.data[i].cpu().numpy(), (640, 480))
                        mk = (m > 0.5).astype(np.uint8)
                        mask_list.append(mk)
                        all_masks = np.maximum(all_masks, mk)

                    band = cv2.subtract(cv2.dilate(all_masks, np.ones((45, 45), np.uint8)),
                                        cv2.dilate(all_masks, np.ones((13, 13), np.uint8)))
                    bys, bxs = np.where(band > 0)

                    surf_pts = []
                    if len(bxs) >= 100:
                        step = max(1, len(bxs) // 400)
                        for k in range(0, len(bxs), step):
                            d = depth_frame.get_distance(int(bxs[k]), int(bys[k]))
                            if d > 0.1:
                                p = rs.rs2_deproject_pixel_to_point(intr, [int(bxs[k]), int(bys[k])], d)
                                surf_pts.append(p)

                    planes = []
                    if len(surf_pts) >= 30:
                        planes = find_planes(surf_pts, self.tilt_cos_limit)

                    if len(bxs) < 100:
                        status = "band too small (%d)" % len(bxs)
                    elif len(surf_pts) < 30:
                        status = "surf pts too few (%d)" % len(surf_pts)
                    elif len(planes) == 0:
                        status = "no valid plane"
                    else:
                        got_any = False
                        used_clusters = set()
                        for i in range(len(r.masks)):
                            mask = mask_list[i]
                            ys, xs = np.where(mask > 0)
                            if len(xs) < 50:
                                continue
                            u, v = int(xs.mean()), int(ys.mean())

                            # 구멍 주변(작은 링)의 실제 depth 중앙값
                            ring_small = cv2.subtract(
                                cv2.dilate(mask, np.ones((21, 21), np.uint8)),
                                cv2.dilate(all_masks, np.ones((9, 9), np.uint8)))
                            srys, srxs = np.where(ring_small > 0)
                            if len(srxs) < 20:
                                continue
                            ds = []
                            stepr = max(1, len(srxs) // 80)
                            for k in range(0, len(srxs), stepr):
                                dd = depth_frame.get_distance(int(srxs[k]), int(srys[k]))
                                if dd > 0.1:
                                    ds.append(dd)
                            if len(ds) < 10:
                                continue
                            ring_med = float(np.median(ds))

                            # 이 구멍이 속한 평면: 교점 거리가 실제 depth와 맞는 평면
                            ray = np.array(rs.rs2_deproject_pixel_to_point(intr, [u, v], 1.0))
                            best_plane, best_diff, best_t = None, 1e9, None
                            for (pn, pc) in planes:
                                denom = float(np.dot(pn, ray))
                                if abs(denom) < 1e-6:
                                    continue
                                t = float(np.dot(pn, pc)) / denom
                                if t < 0.1 or t > 2.0:
                                    continue
                                diff = abs(t - ring_med)
                                if diff < best_diff:
                                    best_diff, best_plane, best_t = diff, (pn, pc), t
                            if best_plane is None or best_diff > PLANE_MATCH_TOL:
                                continue

                            normal_cam, _ = best_plane
                            p_hole_cam = best_t * ray

                            T_ee2base = self.get_T_ee2base()
                            if T_ee2base is None:
                                continue
                            T_cam2base = T_ee2base @ self.T_cam2ee
                            p_base = (T_cam2base @ np.array([p_hole_cam[0], p_hole_cam[1], p_hole_cam[2], 1.0]))[:3]
                            n_base = T_cam2base[:3, :3] @ normal_cam
                            n_base = n_base / np.linalg.norm(n_base)

                            best_c, best_d = -1, 1e9
                            for ci, cluster in enumerate(hole_clusters):
                                if ci in used_clusters:
                                    continue
                                c_center = np.median(np.array(cluster['pts']), axis=0)
                                d = np.linalg.norm(p_base - c_center)
                                if d < best_d:
                                    best_d, best_c = d, ci

                            if best_c >= 0 and best_d < SAME_HOLE_DIST:
                                hole_clusters[best_c]['pts'].append(p_base)
                                hole_clusters[best_c]['normals'].append(n_base)
                                used_clusters.add(best_c)
                            else:
                                hole_clusters.append({'pts': [p_base], 'normals': [n_base]})
                                used_clusters.add(len(hole_clusters) - 1)

                            cv2.circle(annotated, (u, v), 6, (0, 0, 255), -1)
                            got_any = True

                        if got_any:
                            frames_collected += 1
                            status = "OK planes=%d" % len(planes)
                        else:
                            status = "masks rejected (planes=%d)" % len(planes)

                print("[WP%d 프레임] %s | 수집 %d/%d | 클러스터 %d"
                      % (waypoint_idx, status, frames_collected, HOLE_SAMPLES, len(hole_clusters)))

                cv2.putText(annotated, "WP%d | Frame %d/%d | Holes: %d"
                            % (waypoint_idx, frames_collected, HOLE_SAMPLES, len(hole_clusters)),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(annotated, status,
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow('Inspecting...', annotated)
                cv2.waitKey(1)

            pipeline.stop()
            cv2.destroyAllWindows()

            valid_holes = [c for c in hole_clusters if len(c['pts']) >= 5]
            if len(valid_holes) == 0:
                print("   [팔] 구멍 미발견. 스킵.")
                return 0

            print("   [팔] 구멍 %d개 발견! 순서대로 촬영합니다." % len(valid_holes))

            # ===== 각 구멍 순서대로 =====
            for h_idx, cluster in enumerate(valid_holes, start=1):
                hole = np.median(np.array(cluster['pts']), axis=0)
                normal = np.median(np.array(cluster['normals']), axis=0)
                normal = normal / np.linalg.norm(normal)
                target_pos = hole + normal * STANDOFF_FAR
                z_axis = -normal
                print("   [팔] 구멍 %d/%d: base=(%.3f, %.3f, %.3f) 법선=(%.2f, %.2f, %.2f)"
                      % (h_idx, len(valid_holes), hole[0], hole[1], hole[2],
                         normal[0], normal[1], normal[2]))

                err, cur_pose = self.robot.GetActualTCPPose()
                R_cur = Rotation.from_euler('xyz', cur_pose[3:], degrees=True).as_matrix()
                cur_x = R_cur[:, 0]
                candidates = [cur_x, np.array([0, 0, 1.0]), np.array([0, 1.0, 0]),
                              np.array([1.0, 0, 0]), -cur_x]

                target_pose_list = []
                for up in candidates:
                    up = up / np.linalg.norm(up)
                    if abs(np.dot(up, z_axis)) > 0.95:
                        continue
                    x_axis = np.cross(up, z_axis); x_axis /= np.linalg.norm(x_axis)
                    y_axis = np.cross(z_axis, x_axis)
                    R_t = np.column_stack([x_axis, y_axis, z_axis])
                    euler = Rotation.from_matrix(R_t).as_euler('xyz', degrees=True)
                    target_pose_list.append([target_pos[0]*1000, target_pos[1]*1000, target_pos[2]*1000,
                                             euler[0], euler[1], euler[2]])

                # 이동 (IK + J6 고정 + 손목 뒤집힘 해 거부)
                moved = False
                for tp in target_pose_list:
                    ik = self.robot.GetInverseKin(0, tp, -1)
                    if isinstance(ik, tuple) and ik[0] == 0:
                        joints_target = list(ik[1])
                        if abs(joints_target[3] - self.home_joints[3]) > 90.0 or \
                           abs(joints_target[4] - self.home_joints[4]) > 90.0:
                            continue
                        joints_target[5] = self.home_joints[5]   # J6 고정
                        ret = self.robot.MoveJ(joints_target, 0, 0, vel=ARM_VEL)
                        if ret == 0:
                            moved = True
                            break

                if moved:
                    print("   [팔] 1차 도착(%.0fcm). 중앙 보정..." % (STANDOFF_FAR * 100))
                    time.sleep(1.0)
                    self.center_refine(hole)
                    time.sleep(0.5)

                    # 카메라 광축 방향으로 전진 (STANDOFF_FAR -> STANDOFF)
                    forward = STANDOFF_FAR - STANDOFF
                    print("   [팔] %.0fcm 전진 -> 최종 %.0fcm" % (forward * 100, STANDOFF * 100))
                    T_ee2base = self.get_T_ee2base()
                    if T_ee2base is not None:
                        R_cam2base = (T_ee2base @ self.T_cam2ee)[:3, :3]
                        delta = R_cam2base @ np.array([0.0, 0.0, forward])
                        err_p, cur_pose2 = self.robot.GetActualTCPPose()
                        if err_p == 0:
                            new_pose = list(cur_pose2)
                            new_pose[0] += delta[0] * 1000.0
                            new_pose[1] += delta[1] * 1000.0
                            new_pose[2] += delta[2] * 1000.0
                            ik = self.robot.GetInverseKin(0, new_pose, -1)
                            if isinstance(ik, tuple) and ik[0] == 0:
                                jt = list(ik[1])
                                err_j, cur_j = self.robot.GetActualJointPosDegree()
                                if err_j == 0:
                                    jt[5] = cur_j[5]
                                self.robot.MoveJ(jt, 0, 0, vel=ARM_VEL)
                            else:
                                print("   [팔] (전진 IK 실패, 현재 거리에서 촬영)")

                    time.sleep(0.5)
                    self._capture("wp%d_hole%d" % (waypoint_idx, h_idx))
                    n_captured += 1
                else:
                    print("   [팔] 구멍 %d 자세 실패, 스킵" % h_idx)

                self.robot.MoveJ(self.home_joints, 0, 0, vel=ARM_VEL)

            return n_captured

        except Exception as e:
            print("   [팔] 오류:", e)
            try:
                pipeline.stop()
                cv2.destroyAllWindows()
            except Exception:
                pass
            self.robot.MoveJ(self.home_joints, 0, 0, vel=ARM_VEL)
            return n_captured

    def _capture(self, tag):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        try:
            for _ in range(15):
                frames = pipeline.wait_for_frames()
            img = np.asanyarray(frames.get_color_frame().get_data())
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SAVE_DIR, "%s_%s.jpg" % (tag, ts))
            cv2.imwrite(path, img)
            print("   [팔] 촬영 저장: %s" % path)
        finally:
            pipeline.stop()

# ==================== AGV 순회 노드 ====================
class FullInspectionNode(Node):
    def __init__(self, arm):
        super().__init__('full_inspection_node')
        self.arm = arm
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
        print("[AGV] 라이다로 휴지곽 탐지 중...")
        for _ in range(50):
            if self.scan is not None:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.scan is None:
            print("[AGV] 라이다 못 받음")
            return None

        pose = self.get_robot_pose()
        if pose is None:
            print("[AGV] map TF 못 받음. 로봇 살짝 움직인 뒤 다시.")
            return None
        rx, ry, ryaw = pose
        print("[AGV] 현재 위치: (%.2f, %.2f)" % (rx, ry))
        self.robot_x, self.robot_y, self.robot_yaw = rx, ry, ryaw

        msg = self.scan
        candidates = []
        for i, r in enumerate(msg.ranges):
            if DETECT_MIN_RANGE < r < DETECT_MAX_RANGE:
                ang = msg.angle_min + i * msg.angle_increment
                a = ryaw + ang + math.pi
                candidates.append((r, rx + r*math.cos(a), ry + r*math.sin(a)))
        if len(candidates) < 5:
            print("[AGV] 점 부족")
            return None

        candidates.sort(key=lambda c: c[0])
        seed = candidates[0]
        cluster = [(p[1], p[2]) for p in candidates
                   if math.hypot(p[1]-seed[1], p[2]-seed[2]) < CLUSTER_RADIUS]
        if len(cluster) < 3:
            print("[AGV] 덩어리 너무 작음")
            return None

        sx = sum(p[0] for p in cluster) / len(cluster)
        sy = sum(p[1] for p in cluster) / len(cluster)
        max_span = 0.0
        for a in range(len(cluster)):
            for b in range(a+1, len(cluster)):
                d = math.hypot(cluster[a][0]-cluster[b][0], cluster[a][1]-cluster[b][1])
                if d > max_span:
                    max_span = d
        box_radius = max(max_span / 2.0, 0.15)

        dx, dy = sx - rx, sy - ry
        dist = math.hypot(dx, dy)
        if dist > 0.01:
            cx = sx + (dx/dist) * box_radius
            cy = sy + (dy/dist) * box_radius
        else:
            cx, cy = sx, sy
        print("[AGV] 휴지곽 중심: (%.2f, %.2f), 반경 %.2fm" % (cx, cy, box_radius))
        return cx, cy, box_radius

    def generate_waypoints(self, cx, cy, box_radius):
        orbit_r = box_radius + ORBIT_MARGIN
        print("[AGV] 주행 반경: %.2fm" % orbit_r)
        raw = []
        for k in range(NUM_WAYPOINTS):
            th = 2 * math.pi * k / NUM_WAYPOINTS
            raw.append((cx + orbit_r*math.cos(th), cy + orbit_r*math.sin(th)))
        start_idx = min(range(NUM_WAYPOINTS),
                        key=lambda i: math.hypot(raw[i][0]-self.robot_x, raw[i][1]-self.robot_y))
        order = []
        for j in range(NUM_WAYPOINTS):
            idx = (start_idx - j) % NUM_WAYPOINTS if CLOCKWISE else (start_idx + j) % NUM_WAYPOINTS
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
        print("[AGV] Nav2 연결 대기...")
        self.client.wait_for_server()
        print("[AGV] Nav2 연결 완료!")

        result = self.detect_car()
        if result is None:
            return
        cx, cy, box_radius = result

        home_x, home_y, home_yaw = self.robot_x, self.robot_y, self.robot_yaw
        print("[AGV] 시작 위치 저장: (%.2f, %.2f)" % (home_x, home_y))

        wps = self.generate_waypoints(cx, cy, box_radius)
        total_captured = 0

        print("\n===== 전체 검사 시작: %d개 지점 =====\n" % len(wps))
        for i, (x, y) in enumerate(wps, start=1):
            yaw = math.atan2(cy - y, cx - x)   # 휴지곽 바라보기
            print("[%d/%d] 지점 (%.2f, %.2f) 이동..." % (i, len(wps), x, y))
            ok = self.go_to(x, y, yaw)
            if not ok:
                print("   [AGV] 이동 실패, 다음 지점으로")
                continue
            print("   [AGV] 도착. 팔 검사 시작...")
            n = self.arm.inspect_and_capture(i)
            total_captured += n

        print("\n===== 한 바퀴 완료! 총 %d장 촬영 =====" % total_captured)

        if RETURN_HOME:
            print("[AGV] 시작 위치 복귀 중...")
            self.go_to(home_x, home_y, home_yaw)
            print("[AGV] 복귀 완료. 전체 작업 종료!")

def main(args=None):
    # 팔 먼저 초기화 (실행 순간의 자세 = 검사 시작 자세)
    arm = ArmInspector()

    rclpy.init(args=args)
    node = FullInspectionNode(arm)
    try:
        node.run()
    except KeyboardInterrupt:
        print("\n중단됨. 팔 복귀 시도...")
        try:
            arm.robot.MoveJ(arm.home_joints, 0, 0, vel=ARM_VEL)
        except Exception:
            pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
