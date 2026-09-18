import json
import time
import os
import datetime
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
from fairino import Robot
from scipy.spatial.transform import Rotation

# ===== 설정 =====
CAM2EE_JSON = '/home/woohyung/cam2ee.json'
MODEL_PATH = '/home/woohyung/hole__best.pt'
ROBOT_IP = '192.168.58.2'
STANDOFF = 0.25        # 구멍 표면에서 떨어질 거리 (m)
N_SAMPLES = 10
MOVE_VEL = 30
SAVE_DIR = '/home/jaemin/hole_captures'
# ================

def get_T_ee2base(robot):
    err, pose = robot.GetActualTCPPose()
    if err != 0:
        return None
    x, y, z, rx, ry, rz = pose
    t = np.array([x, y, z]) / 1000.0
    R = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def fit_plane_normal(points):
    pts = np.array(points)
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    normal = vh[2]
    return normal, centroid

def capture_photo(save_dir, tag):
    os.makedirs(save_dir, exist_ok=True)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    try:
        for _ in range(15):
            frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        img = np.asanyarray(color_frame.get_data())
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(save_dir, "%s_%s.jpg" % (tag, ts))
        cv2.imwrite(path, img)
        print("촬영 저장: %s" % path)
        return path
    finally:
        pipeline.stop()

def main():
    with open(CAM2EE_JSON) as f:
        calib = json.load(f)
    T_cam2ee = np.array(calib['T_cam2ee'])

    robot = Robot.RPC(ROBOT_IP)
    err, start_joints = robot.GetActualJointPosDegree()
    if err != 0:
        print("로봇 연결 실패")
        return
    start_joints = list(start_joints)
    print("로봇 연결 OK")
    print("시작 자세 저장:", ["%.1f" % j for j in start_joints])
    print("J6 시작 각도: %.1f (이동 후에도 이 값 유지)" % start_joints[5])

    model = YOLO(MODEL_PATH, task='segment')
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    print("구멍을 비추세요. %d프레임 수집 후 팔이 이동합니다..." % N_SAMPLES)

    hole_pts_base = []
    normal_base_list = []

    try:
        while len(hole_pts_base) < N_SAMPLES:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())

            results = model(color_image, conf=0.4, device='cpu', imgsz=480, verbose=False)
            annotated = results[0].plot()
            r = results[0]

            if r.masks is not None and len(r.masks) > 0:
                best_i, best_area, masks_np = 0, 0, []
                for i in range(len(r.masks)):
                    m = cv2.resize(r.masks.data[i].cpu().numpy(), (640, 480))
                    masks_np.append(m)
                    a = (m > 0.5).sum()
                    if a > best_area:
                        best_area, best_i = a, i
                mask = (masks_np[best_i] > 0.5).astype(np.uint8)

                ys, xs = np.where(mask > 0)
                if len(xs) == 0:
                    continue
                u, v = int(xs.mean()), int(ys.mean())

                kernel = np.ones((25, 25), np.uint8)
                dilated = cv2.dilate(mask, kernel)
                ring = dilated - mask
                rys, rxs = np.where(ring > 0)
                if len(rxs) < 30:
                    continue

                ring_pts = []
                step = max(1, len(rxs) // 150)
                for k in range(0, len(rxs), step):
                    d = depth_frame.get_distance(int(rxs[k]), int(rys[k]))
                    if d > 0.1:
                        p = rs.rs2_deproject_pixel_to_point(intr, [int(rxs[k]), int(rys[k])], d)
                        ring_pts.append(p)
                if len(ring_pts) < 20:
                    continue

                normal_cam, _ = fit_plane_normal(ring_pts)
                if normal_cam[2] > 0:
                    normal_cam = -normal_cam

                ring_depth = float(np.median([p[2] for p in ring_pts]))
                p_hole_cam = rs.rs2_deproject_pixel_to_point(intr, [u, v], ring_depth)

                T_ee2base = get_T_ee2base(robot)
                if T_ee2base is None:
                    continue
                T_cam2base = T_ee2base @ T_cam2ee
                p_hole_base = (T_cam2base @ np.array([p_hole_cam[0], p_hole_cam[1], p_hole_cam[2], 1.0]))[:3]
                n_base = T_cam2base[:3, :3] @ normal_cam
                n_base = n_base / np.linalg.norm(n_base)

                hole_pts_base.append(p_hole_base)
                normal_base_list.append(n_base)
                print("샘플 %d/%d: 구멍(base)=(%.3f,%.3f,%.3f) 법선=(%.2f,%.2f,%.2f)"
                      % (len(hole_pts_base), N_SAMPLES,
                         p_hole_base[0], p_hole_base[1], p_hole_base[2],
                         n_base[0], n_base[1], n_base[2]))

                cv2.circle(annotated, (u, v), 6, (0, 0, 255), -1)

            cv2.imshow('Collecting...', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pipeline.stop()
                cv2.destroyAllWindows()
                return

        pipeline.stop()
        cv2.destroyAllWindows()

        hole = np.median(np.array(hole_pts_base), axis=0)
        normal = np.median(np.array(normal_base_list), axis=0)
        normal = normal / np.linalg.norm(normal)
        target_pos = hole + normal * STANDOFF
        z_axis = -normal

        err, cur_pose = robot.GetActualTCPPose()
        R_cur = Rotation.from_euler('xyz', cur_pose[3:], degrees=True).as_matrix()
        cur_x = R_cur[:, 0]
        candidates = [cur_x, np.array([0,0,1.0]), np.array([0,1.0,0]),
                      np.array([1.0,0,0]), -cur_x]

        target_pose_list = []
        for up in candidates:
            up = up / np.linalg.norm(up)
            if abs(np.dot(up, z_axis)) > 0.95:
                continue
            x_axis = np.cross(up, z_axis); x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            R_target = np.column_stack([x_axis, y_axis, z_axis])
            euler = Rotation.from_matrix(R_target).as_euler('xyz', degrees=True)
            target_pose_list.append([target_pos[0]*1000, target_pos[1]*1000, target_pos[2]*1000,
                                     euler[0], euler[1], euler[2]])

        print("\n===== 계산 결과 =====")
        print("구멍(base): (%.3f, %.3f, %.3f) m" % tuple(hole))
        print("법선: (%.2f, %.2f, %.2f)" % tuple(normal))
        print("자세 후보 %d개" % len(target_pose_list))

        input("\n>>> 엔터: 이동(J6 고정) -> 촬영 -> 원위치 복귀 (취소: Ctrl+C) <<<")

        # 1) 이동: IK로 관절해 구하고, J6는 시작값으로 고정
        moved = False
        for i, tp in enumerate(target_pose_list):
            print("후보%d 시도..." % i)
            ik = robot.GetInverseKin(0, tp, -1)
            if isinstance(ik, tuple) and ik[0] == 0:
                joints_target = list(ik[1])
                joints_target[5] = start_joints[5]   # J6 고정! 카메라 회전 방지
                print("  IK 해 찾음. J6=%.1f 유지, MoveJ 이동..." % start_joints[5])
                ret = robot.MoveJ(joints_target, 0, 0, vel=MOVE_VEL)
                if ret == 0:
                    print("도착! (후보%d, J6 안 돌아감)" % i)
                    moved = True
                    break
                else:
                    print("  MoveJ 실패 (에러 %s), 다음 후보..." % str(ret))
            else:
                print("  IK 해 없음 (%s), 다음 후보..." % str(ik if not isinstance(ik, tuple) else ik[0]))

        if moved:
            # 2) 촬영
            time.sleep(1.0)
            capture_photo(SAVE_DIR, "hole")
        else:
            print("이동 실패. 촬영 건너뜀.")

        # 3) 원위치 복귀
        print("원위치 복귀 중...")
        ret = robot.MoveJ(start_joints, 0, 0, vel=MOVE_VEL)
        if ret == 0:
            print("복귀 완료! 작업 끝.")
        else:
            print("복귀 실패 (에러 %s)." % str(ret))

    except KeyboardInterrupt:
        print("\n중단됨")
        try:
            print("원위치 복귀 시도...")
            robot.MoveJ(start_joints, 0, 0, vel=MOVE_VEL)
        except Exception:
            pass

if __name__ == '__main__':
    main()
