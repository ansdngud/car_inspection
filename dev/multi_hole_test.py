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
STANDOFF = 0.25          # 최종 촬영 거리 (m)
STANDOFF_FAR = 0.40      # 1차 접근 거리 (탐지 잘 되는 거리, 여기서 중앙 보정)
HOLE_SAMPLES = 20        # 탐지 프레임 수
DETECT_TIMEOUT = 25.0    # 구멍 찾는 최대 시간 (초)
ARM_VEL = 20             # 팔 속도 (%)
CONF = 0.25              # YOLO 확신도 문턱
IMGSZ = 640              # 추론 해상도 (다중 구멍 잘 잡히게 640)
SAVE_DIR = '/home/jaemin/hole_captures'
SAME_HOLE_DIST = 0.10    # 프레임 간 같은 구멍 매칭 허용 거리 (m)
NORMAL_TILT_LIMIT = 70.0 # 법선이 광축에서 이 각도(도) 이상 벗어나면 그 평면 거부
PLANE_MATCH_TOL = 0.06   # 구멍 주변 depth와 평면 교점 거리 허용 오차 (m)
MAX_PLANES = 2           # 한 시야에서 인식할 최대 평면 수
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

def ransac_plane(points, iters=60, thresh=0.005):
    """RANSAC 평면 피팅. 반환: (법선, 평면 위 점, 인라이어 불리언배열) 또는 (None,None,None)"""
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
        # 기울기 필터 통과한 평면만 채택 (책상/키보드 같은 수평면 배제)
        if -n[2] >= tilt_cos_limit:
            planes.append((n, c))
        # 인라이어 제거 후 남은 점으로 다음 평면 탐색
        pts = pts[~inl]
    return planes

def center_refine(robot, model, target_base, max_iters=3, tol_m=0.02):
    """도착 후 보정: 목표 구멍의 좌표를 화면 픽셀로 역투영해서 해당 마스크를 찾고,
    화면 중앙에 오도록 팔을 이동. (평면 피팅 불필요 -> 작은 판에서도 작동)"""
    with open(CAM2EE_JSON) as f:
        T_cam2ee = np.array(json.load(f)['T_cam2ee'])
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
                results = model(color_image, conf=0.15, device='cpu', imgsz=IMGSZ, verbose=False)
                r = results[0]
                annotated_dbg = results[0].plot()
                if r.masks is None or len(r.masks) == 0:
                    n_nomask += 1
                    cv2.putText(annotated_dbg, "REFINE: no mask", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow('Refine view', annotated_dbg)
                    cv2.waitKey(1)
                    continue

                T_ee2base = get_T_ee2base(robot)
                if T_ee2base is None:
                    continue
                T_cam2base = T_ee2base @ T_cam2ee

                # 목표 구멍을 카메라 화면 픽셀로 역투영: "목표가 화면 어디에 보여야 하나"
                p_t = np.array([target_base[0], target_base[1], target_base[2], 1.0])
                p_t_cam = (np.linalg.inv(T_cam2base) @ p_t)[:3]
                if p_t_cam[2] < 0.05:
                    continue
                # 핀홀 투영 (RealSense 내부 파라미터)
                intr = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                u_t = intr.fx * p_t_cam[0] / p_t_cam[2] + intr.ppx
                v_t = intr.fy * p_t_cam[1] / p_t_cam[2] + intr.ppy

                # 예상 픽셀과 가장 가까운 마스크 = 목표 구멍 (depth 불필요)
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

                # 디버그: 예상 위치(파란 십자)와 마스크들 표시
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
                    continue   # 목표 구멍이 이 프레임에 안 보임 (픽셀 150 초과)

                # 오프셋: 목표까지의 예상 거리로 역투영 (반복 보정이라 근사면 충분)
                cu, cv_ = best_uv
                z = float(p_t_cam[2])
                off_x = (cu - intr.ppx) / intr.fx * z
                off_y = (cv_ - intr.ppy) / intr.fy * z
                off_list.append((off_x, off_y))

            if len(off_list) < 2:
                print("   [보정] 목표 구멍이 안 보임, 보정 생략 (마스크없음 %d / 픽셀불일치 %d / 유효 %d)"
                      % (n_nomask, n_far, len(off_list)))
                cv2.destroyWindow('Refine view') if True else None
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
            T_ee2base = get_T_ee2base(robot)
            if T_ee2base is None:
                return
            R_cam2base = (T_ee2base @ T_cam2ee)[:3, :3]
            moved_ok = False
            scale = 1.0
            for _try in range(2):   # IK 실패 시 절반 이동 재시도
                delta_base = R_cam2base @ np.array([off_x * scale, off_y * scale, 0.0])
                err_p, cur_pose = robot.GetActualTCPPose()
                if err_p != 0:
                    return
                new_pose = list(cur_pose)
                new_pose[0] += delta_base[0] * 1000.0
                new_pose[1] += delta_base[1] * 1000.0
                new_pose[2] += delta_base[2] * 1000.0
                ik = robot.GetInverseKin(0, new_pose, -1)
                if isinstance(ik, tuple) and ik[0] == 0:
                    jt = list(ik[1])
                    err_j, cur_j = robot.GetActualJointPosDegree()
                    if err_j == 0:
                        jt[5] = cur_j[5]   # J6 고정 유지
                    robot.MoveJ(jt, 0, 0, vel=ARM_VEL)
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

def capture_photo(tag):
    os.makedirs(SAVE_DIR, exist_ok=True)
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
        print("   촬영 저장: %s" % path)
    finally:
        pipeline.stop()

def main():
    with open(CAM2EE_JSON) as f:
        calib = json.load(f)
    T_cam2ee = np.array(calib['T_cam2ee'])

    robot = Robot.RPC(ROBOT_IP)
    err, joints = robot.GetActualJointPosDegree()
    if err != 0:
        print("FR5 연결 실패")
        return
    home_joints = list(joints)
    print("로봇 연결 OK. 검사 시작 자세 저장:", ["%.1f" % j for j in home_joints])

    model = YOLO(MODEL_PATH, task='segment')

    # ===== 탐지 =====
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    print("구멍들을 비추세요. %d프레임 수집... (q로 취소)" % HOLE_SAMPLES)

    hole_clusters = []
    t_start = time.time()
    frames_collected = 0
    tilt_cos_limit = float(np.cos(np.radians(NORMAL_TILT_LIMIT)))

    try:
        while frames_collected < HOLE_SAMPLES and (time.time() - t_start) < DETECT_TIMEOUT:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())

            results = model(color_image, conf=CONF, device='cpu', imgsz=IMGSZ, verbose=False)
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

                # 면 띠: 모든 구멍 주변, 구멍 자체는 안전거리 확보하고 제외
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
                    # 다중 평면 탐색 (휴지곽 면 + 잘라낸 판 등)
                    planes = find_planes(surf_pts, tilt_cos_limit)

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

                        # 이 구멍이 어느 평면 위에 있는지: 교점 거리가 실제 depth와 맞는 평면 선택
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
                            continue   # 어느 평면과도 안 맞음 -> 제외

                        normal_cam, _ = best_plane
                        p_hole_cam = best_t * ray

                        T_ee2base = get_T_ee2base(robot)
                        if T_ee2base is None:
                            continue
                        T_cam2base = T_ee2base @ T_cam2ee
                        p_base = (T_cam2base @ np.array([p_hole_cam[0], p_hole_cam[1], p_hole_cam[2], 1.0]))[:3]
                        n_base = T_cam2base[:3, :3] @ normal_cam
                        n_base = n_base / np.linalg.norm(n_base)

                        # 클러스터 매칭 (같은 프레임 마스크 = 서로 다른 구멍)
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

            # 상태를 터미널에도 출력
            print("[프레임] %s | 수집 %d/%d | 클러스터 %d" % (status, frames_collected, HOLE_SAMPLES, len(hole_clusters)))

            cv2.putText(annotated, "Frame %d/%d | Holes: %d" % (frames_collected, HOLE_SAMPLES, len(hole_clusters)),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated, status,
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow('Detecting holes...', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pipeline.stop()
                cv2.destroyAllWindows()
                return

        pipeline.stop()
        cv2.destroyAllWindows()

        valid_holes = [c for c in hole_clusters if len(c['pts']) >= 5]
        if len(valid_holes) == 0:
            print("구멍 미발견 (클러스터 %d개, 샘플 부족)" % len(hole_clusters))
            for ci, c in enumerate(hole_clusters):
                print("  클러스터 %d: 샘플 %d개" % (ci, len(c['pts'])))
            return

        print("\n===== 구멍 %d개 발견! =====" % len(valid_holes))
        for h_idx, cluster in enumerate(valid_holes, start=1):
            hole = np.median(np.array(cluster['pts']), axis=0)
            nrm = np.median(np.array(cluster['normals']), axis=0)
            nrm = nrm / np.linalg.norm(nrm)
            print("  구멍 %d: base=(%.3f, %.3f, %.3f) 법선=(%.2f, %.2f, %.2f) 샘플 %d개"
                  % (h_idx, hole[0], hole[1], hole[2],
                     nrm[0], nrm[1], nrm[2], len(cluster['pts'])))

        input("\n>>> 엔터: 순서대로 이동->촬영->복귀 시작 (취소: Ctrl+C) <<<")

        # ===== 각 구멍 순서대로: 40cm 접근 -> 보정 -> 전진 -> 촬영 -> 복귀 =====
        n_captured = 0
        for h_idx, cluster in enumerate(valid_holes, start=1):
            hole = np.median(np.array(cluster['pts']), axis=0)
            normal = np.median(np.array(cluster['normals']), axis=0)
            normal = normal / np.linalg.norm(normal)
            target_pos = hole + normal * STANDOFF_FAR
            z_axis = -normal
            print("\n[구멍 %d/%d] base=(%.3f, %.3f, %.3f) 법선=(%.2f, %.2f, %.2f)"
                  % (h_idx, len(valid_holes), hole[0], hole[1], hole[2],
                     normal[0], normal[1], normal[2]))

            err, cur_pose = robot.GetActualTCPPose()
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
                ik = robot.GetInverseKin(0, tp, -1)
                if isinstance(ik, tuple) and ik[0] == 0:
                    joints_target = list(ik[1])
                    if abs(joints_target[3] - home_joints[3]) > 90.0 or \
                       abs(joints_target[4] - home_joints[4]) > 90.0:
                        print("   (손목 뒤집힘 해 거부, 다음 후보...)")
                        continue
                    joints_target[5] = home_joints[5]   # J6 고정
                    ret = robot.MoveJ(joints_target, 0, 0, vel=ARM_VEL)
                    if ret == 0:
                        moved = True
                        break

            if moved:
                print("   1차 도착(%.0fcm)! 중앙 보정 시작..." % (STANDOFF_FAR * 100))
                time.sleep(1.0)
                center_refine(robot, model, hole)
                time.sleep(0.5)

                # 카메라 광축 방향으로 전진 (STANDOFF_FAR -> STANDOFF)
                forward = STANDOFF_FAR - STANDOFF
                print("   %.0fcm 전진 -> 최종 %.0fcm..." % (forward * 100, STANDOFF * 100))
                T_ee2base = get_T_ee2base(robot)
                if T_ee2base is not None:
                    R_cam2base = (T_ee2base @ T_cam2ee)[:3, :3]
                    delta = R_cam2base @ np.array([0.0, 0.0, forward])
                    err_p, cur_pose2 = robot.GetActualTCPPose()
                    if err_p == 0:
                        new_pose = list(cur_pose2)
                        new_pose[0] += delta[0] * 1000.0
                        new_pose[1] += delta[1] * 1000.0
                        new_pose[2] += delta[2] * 1000.0
                        ik = robot.GetInverseKin(0, new_pose, -1)
                        if isinstance(ik, tuple) and ik[0] == 0:
                            jt = list(ik[1])
                            err_j, cur_j = robot.GetActualJointPosDegree()
                            if err_j == 0:
                                jt[5] = cur_j[5]
                            robot.MoveJ(jt, 0, 0, vel=ARM_VEL)
                        else:
                            print("   (전진 IK 실패, 현재 거리에서 촬영)")

                time.sleep(0.5)
                capture_photo("hole%d" % h_idx)
                n_captured += 1
            else:
                print("   자세 실패, 이 구멍 스킵")

            print("   검사 자세 복귀...")
            robot.MoveJ(home_joints, 0, 0, vel=ARM_VEL)

        print("\n===== 완료! 총 %d/%d개 촬영 =====" % (n_captured, len(valid_holes)))

    except KeyboardInterrupt:
        print("\n중단됨. 복귀 시도...")
        try:
            pipeline.stop()
            cv2.destroyAllWindows()
        except Exception:
            pass
        robot.MoveJ(home_joints, 0, 0, vel=ARM_VEL)

if __name__ == '__main__':
    main()
