'''
#!/usr/bin/env python3
"""
Eye-in-Hand Calibration for FR5 Robot
======================================

카메라가 로봇 End-Effector에 부착되어 있고, ChArUco 보드는 월드(작업공간)에
고정되어 있는 Eye-in-Hand 구성에서의 핸드-아이 캘리브레이션.

목표: T_cam2ee (카메라 -> 엔드이펙터) 변환 행렬을 구한다.

이 버전의 주요 개선점:
- scipy를 사용해 12가지 오일러각 컨벤션을 모두 시도
- 각 컨벤션마다 5가지 calibrateHandEye 방법 시도
- 검증 표준편차가 가장 작은 컨벤션+방법 조합 자동 선택
- 결과 테이블을 오름차순으로 출력하여 비교 가능

사전 요구사항:
    pip install scipy
"""

import cv2
import json
import time
import numpy as np
import pyrealsense2 as rs
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

from scipy.spatial.transform import Rotation as ScipyRotation
from fairino.Robot import RPC


class EyeInHandCalibrator:
    """Eye-in-Hand calibration을 수행하는 클래스."""

    def __init__(self,
                 robot_ip: str = "192.168.58.2",
                 save_dir: str = "./calib_data_eye_in_hand",
                 charuco_squares_x: int = 12,
                 charuco_squares_y: int = 8,
                 charuco_square_length: float = 46.0,   # mm
                 charuco_marker_length: float = 36.0,   # mm
                 min_charuco_corners: int = 15,
                 min_detection_confidence: float = 0.75,
                 max_distance_threshold: float = 0.3,
                 far_pose_filter_mm: float = 400.0):
        self.robot_ip = robot_ip
        self.save_dir = Path(save_dir)
        self.poses_dir = self.save_dir / "poses"
        self.angles_dir = self.save_dir / "angles"

        self.charuco_squares_x = charuco_squares_x
        self.charuco_squares_y = charuco_squares_y
        self.charuco_square_length = charuco_square_length
        self.charuco_marker_length = charuco_marker_length

        self.min_charuco_corners = min_charuco_corners
        self.min_detection_confidence = min_detection_confidence
        self.max_distance_threshold = max_distance_threshold
        self.far_pose_filter_mm = far_pose_filter_mm

        # ChArUco 보드
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.charuco_board = cv2.aruco.CharucoBoard(
            size=(charuco_squares_x, charuco_squares_y),
            squareLength=self.charuco_square_length / 1000.0,
            markerLength=self.charuco_marker_length / 1000.0,
            dictionary=self.aruco_dict
        )

        # 로봇 연결
        self.robot = RPC(robot_ip)
        self.robot.DragTeachSwitch(0)

        # 디렉토리
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.poses_dir.mkdir(parents=True, exist_ok=True)
        self.angles_dir.mkdir(parents=True, exist_ok=True)

        # SDK 호출 방식 캐싱
        self._tcp_flag_mode = None
        self._joint_flag_mode = None

        print("=" * 60)
        print("Eye-in-Hand Calibrator 초기화 완료")
        print("=" * 60)
        print(f"ChArUco 크기: {charuco_squares_x}x{charuco_squares_y}")
        print(f"사각형 길이: {charuco_square_length:.1f}mm")
        print(f"마커 길이:   {charuco_marker_length:.1f}mm")
        print(f"모드:        Eye-in-Hand (카메라가 EE에 부착)")
        print(f"목표:        T_cam2ee (카메라 -> 엔드이펙터) 계산")
        print(f"품질 기준:   최소 코너 {min_charuco_corners}개, "
              f"신뢰도 {min_detection_confidence}, 최대 거리 {max_distance_threshold}m")
        print(f"원거리 필터: > {far_pose_filter_mm:.0f}mm 포즈는 자동 제외")
        print(f"컨벤션 테스트: scipy로 12가지 오일러각 컨벤션 모두 시도")
        print("=" * 60)

    # ------------------------------------------------------------------
    # FR5 SDK 안전 래퍼
    # ------------------------------------------------------------------
    def _safe_get_tcp_pose(self) -> Optional[List[float]]:
        """FR5의 GetActualTCPPose를 여러 호출 방식으로 시도."""
        attempts = []
        if self._tcp_flag_mode is not None:
            attempts.append(self._tcp_flag_mode)
        for mode in ["flag1", "flag0", "noarg"]:
            if mode not in attempts:
                attempts.append(mode)

        for mode in attempts:
            try:
                if mode == "flag1":
                    ret = self.robot.GetActualTCPPose(1)
                elif mode == "flag0":
                    ret = self.robot.GetActualTCPPose(0)
                else:
                    ret = self.robot.GetActualTCPPose()
            except TypeError:
                continue
            except Exception as e:
                print(f"\n⚠️ TCP 포즈 호출 예외 ({mode}): {e}")
                continue

            if isinstance(ret, tuple) and len(ret) >= 2:
                error_code, coords = ret[0], ret[1]
                if error_code == 0 and coords is not None and len(coords) >= 6:
                    self._tcp_flag_mode = mode
                    return list(coords)

            if isinstance(ret, int):
                continue

        print(f"\n⚠️ TCP 포즈 모든 호출 방식 실패")
        return None

    def _safe_get_joint_angles(self) -> Optional[List[float]]:
        """FR5의 GetActualJointPosDegree를 여러 호출 방식으로 시도."""
        attempts = []
        if self._joint_flag_mode is not None:
            attempts.append(self._joint_flag_mode)
        for mode in ["flag1", "flag0", "noarg"]:
            if mode not in attempts:
                attempts.append(mode)

        for mode in attempts:
            try:
                if mode == "flag1":
                    ret = self.robot.GetActualJointPosDegree(1)
                elif mode == "flag0":
                    ret = self.robot.GetActualJointPosDegree(0)
                else:
                    ret = self.robot.GetActualJointPosDegree()
            except TypeError:
                continue
            except Exception as e:
                print(f"\n⚠️ 관절 각도 호출 예외 ({mode}): {e}")
                continue

            if isinstance(ret, tuple) and len(ret) >= 2:
                error_code, angles = ret[0], ret[1]
                if error_code == 0 and angles is not None and len(angles) >= 6:
                    self._joint_flag_mode = mode
                    return list(angles)

            if isinstance(ret, int):
                continue

        print(f"\n⚠️ 관절 각도 모든 호출 방식 실패")
        return None

    # ------------------------------------------------------------------
    # ChArUco 검출
    # ------------------------------------------------------------------
    def detect_charuco_pose(self, image: np.ndarray,
                            camera_matrix: np.ndarray,
                            dist_coeffs: np.ndarray
                            ) -> Optional[Tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]]:
        """이미지에서 ChArUco 보드의 카메라 기준 포즈(target2cam)를 검출."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)

        charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            charuco_detector.detectBoard(gray_enhanced)

        if (charuco_corners is not None and charuco_ids is not None
                and len(charuco_corners) >= 6
                and len(charuco_corners) == len(charuco_ids)):

            obj_points, img_points = self.charuco_board.matchImagePoints(
                charuco_corners, charuco_ids)

            if obj_points is not None and img_points is not None and len(obj_points) >= 6:
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, img_points,
                    camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    return rvec, tvec, charuco_corners, charuco_ids

        # ArUco fallback
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.adaptiveThreshWinSizeMin = 3
        aruco_params.adaptiveThreshWinSizeMax = 23
        aruco_params.adaptiveThreshWinSizeStep = 10
        aruco_params.minMarkerPerimeterRate = 0.01
        aruco_params.maxMarkerPerimeterRate = 4.0
        aruco_params.errorCorrectionRate = 0.6

        aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, aruco_params)
        corners, ids, _ = aruco_detector.detectMarkers(gray_enhanced)

        if ids is not None and len(ids) >= 10:
            marker_points_3d = []
            marker_points_2d = []

            for i, marker_id in enumerate(ids):
                marker_id = int(marker_id[0])
                marker_row = marker_id // (self.charuco_squares_x - 1)
                marker_col = marker_id % (self.charuco_squares_x - 1)

                cx = marker_col * (self.charuco_square_length / 1000.0)
                cy = marker_row * (self.charuco_square_length / 1000.0)
                half = (self.charuco_marker_length / 1000.0) / 2

                marker_points_3d.extend([
                    [cx - half, cy - half, 0],
                    [cx + half, cy - half, 0],
                    [cx + half, cy + half, 0],
                    [cx - half, cy + half, 0],
                ])
                marker_points_2d.extend(corners[i][0])

            if len(marker_points_3d) >= 12:
                marker_points_3d = np.array(marker_points_3d, dtype=np.float32)
                marker_points_2d = np.array(marker_points_2d, dtype=np.float32)

                ret, rvec, tvec = cv2.solvePnP(
                    marker_points_3d, marker_points_2d,
                    camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if ret:
                    charuco_corners_from_markers = np.array(
                        [np.mean(corners[i][0], axis=0) for i in range(len(ids))],
                        dtype=np.float32
                    )
                    charuco_ids_from_markers = np.array(
                        [int(mid[0]) for mid in ids], dtype=np.int32
                    )
                    return rvec, tvec, charuco_corners_from_markers, charuco_ids_from_markers

        return None

    # ------------------------------------------------------------------
    # 로봇 포즈 / 관절
    # ------------------------------------------------------------------
    def get_robot_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        수집 단계에서 사용. 기본 zyx(소문자=고정축)로 저장하되,
        실제 계산 단계에서는 coords 원본을 다시 읽어서 12개 컨벤션 모두 시도함.
        """
        coords = self._safe_get_tcp_pose()
        if coords is None:
            coords = [0.0] * 6

        position = np.array(coords[0:3]) / 1000.0
        R_gripper2base = ScipyRotation.from_euler(
            "zyx", coords[3:6], degrees=True
        ).as_matrix()
        return R_gripper2base, position.reshape(3, 1)

    def get_robot_angles(self) -> List[float]:
        angles = self._safe_get_joint_angles()
        return angles if angles is not None else [0.0] * 6

    # ------------------------------------------------------------------
    # 품질 평가
    # ------------------------------------------------------------------
    def evaluate_charuco_quality(self, charuco_corners, charuco_ids,
                                 rvec, tvec) -> Tuple[bool, str, float]:
        quality_score = 0.0
        issues = []

        corner_count = len(charuco_corners) if charuco_corners is not None else 0
        if charuco_corners is None or corner_count < self.min_charuco_corners:
            issues.append(f"코너 부족 ({corner_count}/{self.min_charuco_corners})")
            quality_score -= 0.1
        else:
            corner_ratio = corner_count / (
                (self.charuco_squares_x - 1) * (self.charuco_squares_y - 1)
            )
            quality_score += corner_ratio * 0.6

        distance = float(np.linalg.norm(tvec))
        if distance > self.max_distance_threshold:
            issues.append(f"거리 과대 ({distance*1000:.0f}mm)")
            quality_score -= 0.2
        elif distance < 0.15:
            issues.append(f"거리 과소 ({distance*1000:.0f}mm)")
            quality_score -= 0.1
        else:
            distance_score = 1.0 - (distance - 0.15) / (0.5 - 0.15)
            quality_score += max(0, distance_score) * 0.3

        R_mat, _ = cv2.Rodrigues(rvec)
        z_axis = np.array([0, 0, 1])
        board_z_in_cam = R_mat @ z_axis
        angle_deg = np.degrees(
            np.arccos(np.clip(np.abs(np.dot(board_z_in_cam, z_axis)), 0, 1))
        )
        if angle_deg > 75:
            issues.append(f"각도 극단 ({angle_deg:.0f}도)")
            quality_score -= 0.1
        else:
            quality_score += (1.0 - angle_deg / 75.0) * 0.2

        if charuco_corners is not None and corner_count > 0:
            try:
                arr = np.array(charuco_corners).reshape(-1, 2)
                if np.std(arr[:, 0]) < 30 or np.std(arr[:, 1]) < 30:
                    issues.append("코너 분포 집중")
                    quality_score -= 0.05
            except Exception:
                pass

        quality_score += 0.3

        is_good = quality_score >= self.min_detection_confidence
        msg = (f"품질 양호 ({quality_score:.2f})" if is_good
               else f"품질 불량 ({quality_score:.2f}) - {'; '.join(issues)}")
        return is_good, msg, float(quality_score)

    # ------------------------------------------------------------------
    # 데이터 수집
    # ------------------------------------------------------------------
    def collect_poses_with_camera_feed(self, num_poses: int = 20) -> bool:
        print(f"\n=== 실시간 포즈 수집 (Eye-in-Hand) ===")
        print(f"수집할 포즈 수: {num_poses}")
        print("카메라는 EE에 부착되어 있습니다. 고정된 ChArUco 보드를")
        print("여러 각도/거리/회전으로 바라보도록 로봇을 움직여 주세요.")
        print("⚠️  J4, J5, J6 모두 크게 변화시켜야 합니다!\n")

        config = rs.config()
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

        pipeline = rs.pipeline()
        profile = pipeline.start(config)
        print("✅ RealSense 카메라 연결 성공 (848x480)")

        color_profile = profile.get_stream(rs.stream.color)
        color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        camera_matrix = np.array([
            [color_intrinsics.fx, 0.0, color_intrinsics.ppx],
            [0.0, color_intrinsics.fy, color_intrinsics.ppy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        dist_coeffs = np.zeros(5, dtype=np.float64)

        print(f"  - 해상도: {color_intrinsics.width}x{color_intrinsics.height}")
        print(f"  - fx={color_intrinsics.fx:.2f}, fy={color_intrinsics.fy:.2f}")
        print(f"  - cx={color_intrinsics.ppx:.2f}, cy={color_intrinsics.ppy:.2f}\n")

        collected = 0
        pose_index = 0

        self.robot.DragTeachSwitch(1)
        print("🤚 직접 교시 모드 ON: 로봇을 손으로 움직여 포즈를 잡으세요.\n")

        try:
            while collected < num_poses:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = (np.asanyarray(depth_frame.get_data())
                               if depth_frame else None)

                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray_enh = clahe.apply(gray)
                vis_image = cv2.cvtColor(gray_enh, cv2.COLOR_GRAY2BGR)

                aruco_params = cv2.aruco.DetectorParameters()
                aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, aruco_params)
                corners, ids, _ = aruco_detector.detectMarkers(gray_enh)
                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedMarkers(vis_image, corners, ids)

                charuco_detected = False
                quality_score = 0.0
                distance = 0.0
                charuco_result = self.detect_charuco_pose(
                    color_image, camera_matrix, dist_coeffs)

                if charuco_result is not None:
                    rvec, tvec, charuco_corners, charuco_ids = charuco_result
                    charuco_detected = True
                    distance = float(np.linalg.norm(tvec))

                    if charuco_corners is not None and charuco_ids is not None:
                        is_good, _, quality_score = self.evaluate_charuco_quality(
                            charuco_corners, charuco_ids, rvec, tvec)
                        if is_good:
                            cv2.drawFrameAxes(vis_image, camera_matrix,
                                              dist_coeffs, rvec, tvec, 0.05)

                try:
                    angles = self.get_robot_angles()
                    coords = self._safe_get_tcp_pose() or [0.0] * 6
                except Exception:
                    angles = [0.0] * 6
                    coords = [0.0] * 6

                # HUD
                cv2.putText(vis_image, f"Pose: {collected}/{num_poses}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if charuco_detected:
                    cv2.putText(vis_image,
                                f"ChArUco: DETECTED ({distance*1000:.0f}mm)",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(vis_image, "ChArUco: NOT DETECTED", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                q_color = ((0, 255, 0) if quality_score >= self.min_detection_confidence
                           else (0, 0, 255))
                cv2.putText(vis_image, f"Quality: {quality_score:.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, q_color, 2)

                # J4/J5/J6 강조 표시 (이게 중요!)
                cv2.putText(vis_image,
                            f"J1-J3: [{angles[0]:.1f},{angles[1]:.1f},{angles[2]:.1f}]",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                cv2.putText(vis_image,
                            f"J4={angles[3]:+6.1f}  J5={angles[4]:+6.1f}  J6={angles[5]:+6.1f}",
                            (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(vis_image,
                            f"TCP: [{coords[0]:.1f},{coords[1]:.1f},{coords[2]:.1f}]",
                            (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                if charuco_detected:
                    cv2.putText(vis_image, "Press 's' to save, 'q' to quit",
                                (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(vis_image, "No board - cannot save",
                                (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.imshow('Eye-in-Hand Calibration', vis_image)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print(f"\n사용자 중단. 수집된 포즈: {collected}")
                    break

                elif key == ord('s'):
                    print("\n🔒 로봇 고정 + 1초 안정화...")
                    self.robot.DragTeachSwitch(0)
                    time.sleep(1.0)

                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if not color_frame:
                        self.robot.DragTeachSwitch(1)
                        continue

                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = (np.asanyarray(depth_frame.get_data())
                                   if depth_frame else None)

                    charuco_result = self.detect_charuco_pose(
                        color_image, camera_matrix, dist_coeffs)
                    if charuco_result is None:
                        print("❌ 저장 실패: ChArUco 검출 안됨")
                        self.robot.DragTeachSwitch(1)
                        continue

                    rvec, tvec, charuco_corners, charuco_ids = charuco_result
                    corner_count = (len(charuco_corners)
                                    if charuco_corners is not None else 0)
                    if corner_count < 6:
                        print(f"❌ 저장 실패: 코너 {corner_count}개 (< 6)")
                        self.robot.DragTeachSwitch(1)
                        continue

                    distance = float(np.linalg.norm(tvec))

                    if distance * 1000 > self.far_pose_filter_mm:
                        print(f"❌ 저장 실패: 거리 {distance*1000:.0f}mm > "
                              f"{self.far_pose_filter_mm:.0f}mm")
                        self.robot.DragTeachSwitch(1)
                        continue

                    if charuco_corners is not None and charuco_ids is not None:
                        is_good, _, quality_score = self.evaluate_charuco_quality(
                            charuco_corners, charuco_ids, rvec, tvec)
                    else:
                        is_good, quality_score = True, 0.5

                    if not is_good:
                        print(f"❌ 품질 불충분 ({quality_score:.2f}), 저장 안 함")
                        self.robot.DragTeachSwitch(1)
                        continue

                    # --- 저장 ---
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    angles = self.get_robot_angles()
                    coords = self._safe_get_tcp_pose() or [0.0] * 6

                    angle_path = self.angles_dir / f"{pose_index:02d}_{timestamp}_angles.json"
                    with open(angle_path, 'w') as f:
                        json.dump({
                            "angles": list(angles),
                            "coords": list(coords),
                            "pose_index": pose_index,
                            "timestamp": timestamp
                        }, f, indent=2)

                    R_g2b, t_g2b = self.get_robot_pose()
                    pose_path = self.poses_dir / f"{pose_index:02d}_{timestamp}_pose.json"
                    with open(pose_path, 'w') as f:
                        json.dump({
                            "R_gripper2base": R_g2b.tolist(),
                            "t_gripper2base": t_g2b.tolist(),
                            "angles": list(angles),
                            "coords": list(coords),
                            "pose_index": pose_index,
                            "timestamp": timestamp
                        }, f, indent=2)

                    color_path = self.save_dir / "color" / f"{pose_index:02d}_{timestamp}.jpg"
                    color_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(color_path), color_image)

                    depth_path = None
                    if depth_image is not None:
                        depth_path = self.save_dir / "depth" / f"{pose_index:02d}_{timestamp}.npy"
                        depth_path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(str(depth_path), depth_image)

                        depth_png = (self.save_dir / "depth" / "converted_png"
                                     / f"{pose_index:02d}_{timestamp}.png")
                        depth_png.parent.mkdir(parents=True, exist_ok=True)
                        depth_norm = cv2.normalize(depth_image, None, 0, 255,
                                                   cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        cv2.imwrite(str(depth_png), depth_norm)

                    intrinsics_path = (self.save_dir / "intrinsics"
                                       / f"{pose_index:02d}_{timestamp}.json")
                    intrinsics_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(intrinsics_path, 'w') as f:
                        json.dump({
                            "color_intrinsics": {
                                "width": color_intrinsics.width,
                                "height": color_intrinsics.height,
                                "fx": float(color_intrinsics.fx),
                                "fy": float(color_intrinsics.fy),
                                "ppx": float(color_intrinsics.ppx),
                                "ppy": float(color_intrinsics.ppy),
                                "distortion_model": str(color_intrinsics.model),
                                "distortion_coeffs":
                                    [float(c) for c in color_intrinsics.coeffs]
                            },
                            "camera_matrix": camera_matrix.tolist(),
                            "dist_coeffs": dist_coeffs.tolist(),
                            "timestamp": datetime.now().isoformat()
                        }, f, indent=2)

                    charuco_path = self.save_dir / f"{pose_index:02d}_{timestamp}_charuco.json"
                    with open(charuco_path, 'w') as f:
                        json.dump({
                            "rvec_target2cam": rvec.tolist(),
                            "tvec_target2cam": tvec.tolist(),
                            "pose_index": pose_index,
                            "timestamp": timestamp,
                            "charuco_corners_count": int(corner_count),
                            "distance_mm": float(distance * 1000),
                            "quality_score": float(quality_score),
                            "detection_method":
                                "charuco" if charuco_corners is not None else "aruco_fallback",
                            "color_image_path": str(color_path),
                            "depth_image_path": str(depth_path) if depth_path else None,
                            "intrinsics_path": str(intrinsics_path)
                        }, f, indent=2)

                    print(f"✅ 포즈 {pose_index} 저장 완료 "
                          f"(거리 {distance*1000:.0f}mm, 품질 {quality_score:.2f}, "
                          f"J4={angles[3]:.1f} J5={angles[4]:.1f} J6={angles[5]:.1f})")

                    collected += 1
                    pose_index += 1

                    self.robot.DragTeachSwitch(1)
                    print("🤚 직접 교시 모드 ON: 다음 포즈로 이동하세요.\n")

        except KeyboardInterrupt:
            print(f"\nKeyboardInterrupt. 수집된 포즈: {collected}")
        except Exception as e:
            print(f"\n❌ 오류: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.robot.DragTeachSwitch(0)
            print("\n🔒 직접 교시 모드 OFF")
            try:
                pipeline.stop()
            except Exception:
                pass
            cv2.destroyAllWindows()

        print(f"\n{'='*60}")
        print(f" 수집 완료: 총 {collected}개 포즈")
        print(f"{'='*60}")
        return collected >= 10

    # ------------------------------------------------------------------
    # Eye-in-Hand 변환 행렬 계산 (12개 컨벤션 brute force)
    # ------------------------------------------------------------------
    def calculate_transformation_matrix(self) -> bool:
        """
        scipy가 지원하는 12가지 오일러각 컨벤션 × 5가지 calibrateHandEye 방법을
        모두 시도해서 검증 표준편차가 가장 작은 조합을 자동 선택.
        """
        print("\n=== Eye-in-Hand 변환 행렬 계산 (12 컨벤션 × 5 방법) ===")

        charuco_files = sorted(self.save_dir.glob("*_charuco.json"),
                               key=lambda x: int(x.stem.split('_')[0]))

        if len(charuco_files) < 3:
            print(f"❌ 데이터 부족 (필요: 3+, 현재: {len(charuco_files)})")
            return False

        # 데이터 로드
        pose_data = []
        for charuco_file in charuco_files:
            idx = int(charuco_file.stem.split('_')[0])
            matching = list(self.poses_dir.glob(f"{idx:02d}_*_pose.json"))
            if not matching:
                print(f"⚠️ 포즈 {idx}: pose 파일 없음, 건너뜀")
                continue
            pose_file = matching[0]

            with open(charuco_file, 'r') as f:
                cdata = json.load(f)
            with open(pose_file, 'r') as f:
                pdata = json.load(f)

            dist_mm = cdata.get("distance_mm", 0)
            if dist_mm > self.far_pose_filter_mm:
                print(f"  포즈 {idx:02d}: 거리 {dist_mm:.0f}mm > "
                      f"{self.far_pose_filter_mm:.0f}mm, 제외")
                continue

            rvec = np.array(cdata["rvec_target2cam"], dtype=np.float64).reshape(3, 1)
            tvec = np.array(cdata["tvec_target2cam"], dtype=np.float64).reshape(3, 1)
            R_t2c, _ = cv2.Rodrigues(rvec)

            pose_data.append({
                "idx": idx,
                "R_t2c": R_t2c,
                "t_t2c": tvec,
                "coords": pdata["coords"],
                "quality": cdata.get("quality_score", 0.5),
                "dist_mm": dist_mm,
            })

        n = len(pose_data)
        if n < 3:
            print(f"❌ 유효 포즈 부족: {n}")
            return False

        print(f"\n총 {n}개 포즈 사용")

        # scipy가 지원하는 12가지 오일러 컨벤션
        # 소문자 = extrinsic (고정축, 월드축 기준 순차 회전)
        # 대문자 = intrinsic (움직이는 축, 자기 축 기준 순차 회전)
        conventions = [
            "xyz", "xzy", "yxz", "yzx", "zxy", "zyx",
            "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX",
        ]

        methods = {
            "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
            "PARK":       cv2.CALIB_HAND_EYE_PARK,
            "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
            "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
            "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
        }

        best_overall = None
        best_conv_name = None
        best_std_norm = float('inf')
        all_summaries = []

        for conv_name in conventions:
            try:
                R_gripper2base_list = []
                t_gripper2base_list = []
                R_target2cam_list = []
                t_target2cam_list = []

                for d in pose_data:
                    coords = d["coords"]
                    position = np.array(coords[0:3]) / 1000.0
                    R_g2b = ScipyRotation.from_euler(
                        conv_name, coords[3:6], degrees=True
                    ).as_matrix()
                    t_g2b = position.reshape(3, 1)

                    R_gripper2base_list.append(R_g2b)
                    t_gripper2base_list.append(t_g2b)
                    R_target2cam_list.append(d["R_t2c"])
                    t_target2cam_list.append(d["t_t2c"])

                results = {}
                for name, method in methods.items():
                    try:
                        R_c2e, t_c2e = cv2.calibrateHandEye(
                            R_gripper2base_list, t_gripper2base_list,
                            R_target2cam_list, t_target2cam_list,
                            method=method
                        )
                        results[name] = (R_c2e, t_c2e)
                    except Exception:
                        pass

                if not results:
                    continue

                # 중간값에 가장 가까운 방법 선택
                names = list(results.keys())
                t_list = np.array([results[nm][1].flatten() for nm in names])
                median_t = np.median(t_list, axis=0)
                dists = np.linalg.norm(t_list - median_t, axis=1)
                chosen = names[int(np.argmin(dists))]

                R_cam2ee, t_cam2ee = results[chosen]
                T_cam2ee = np.eye(4)
                T_cam2ee[:3, :3] = R_cam2ee
                T_cam2ee[:3,  3] = t_cam2ee.flatten()

                # 검증: 고정된 보드의 베이스 좌표계 위치 일관성
                target_pts = []
                for R_g, t_g, R_t, t_t in zip(R_gripper2base_list, t_gripper2base_list,
                                              R_target2cam_list, t_target2cam_list):
                    T_g = np.eye(4); T_g[:3, :3] = R_g; T_g[:3, 3] = t_g.flatten()
                    T_t = np.eye(4); T_t[:3, :3] = R_t; T_t[:3, 3] = t_t.flatten()
                    target_pts.append((T_g @ T_cam2ee @ T_t)[:3, 3])

                target_pts = np.array(target_pts)
                std_pt = target_pts.std(axis=0)
                std_norm = float(np.linalg.norm(std_pt))
                max_dev = float(np.linalg.norm(target_pts - target_pts.mean(axis=0), axis=1).max())
                t_norm = float(np.linalg.norm(t_cam2ee))

                all_summaries.append({
                    "conv": conv_name,
                    "method": chosen,
                    "std_norm_mm": std_norm * 1000,
                    "max_dev_mm": max_dev * 1000,
                    "t_norm_mm": t_norm * 1000,
                })

                if std_norm < best_std_norm:
                    best_std_norm = std_norm
                    best_overall = {
                        "R_cam2ee": R_cam2ee,
                        "t_cam2ee": t_cam2ee,
                        "method": chosen,
                        "std_pt_mm": std_pt * 1000,
                        "max_dev_mm": max_dev * 1000,
                        "mean_pt_m": target_pts.mean(axis=0),
                    }
                    best_conv_name = conv_name

            except Exception as e:
                print(f"  컨벤션 {conv_name} 계산 중 오류: {e}")
                continue

        # 모든 컨벤션 결과 테이블 출력 (std_norm 순 정렬)
        all_summaries.sort(key=lambda x: x["std_norm_mm"])
        print(f"\n{'='*72}")
        print(f"🏆 모든 컨벤션 결과 (std_norm 오름차순)")
        print(f"{'='*72}")
        print(f"  {'컨벤션':10s} {'방법':12s} {'std_norm':>12s} "
              f"{'max_dev':>12s} {'||t||':>12s}")
        print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for i, s in enumerate(all_summaries):
            marker = " ← 최선" if i == 0 else ""
            print(f"  {s['conv']:10s} {s['method']:12s} "
                  f"{s['std_norm_mm']:10.1f}mm "
                  f"{s['max_dev_mm']:10.1f}mm "
                  f"{s['t_norm_mm']:10.1f}mm{marker}")

        if best_overall is None:
            print("\n❌ 모든 컨벤션/방법 실패")
            return False

        R_cam2ee = best_overall["R_cam2ee"]
        t_cam2ee = best_overall["t_cam2ee"]
        T_cam2ee = np.eye(4)
        T_cam2ee[:3, :3] = R_cam2ee
        T_cam2ee[:3,  3] = t_cam2ee.flatten()

        result = {
            "description": "Eye-in-Hand calibration: 카메라 -> 엔드이펙터 변환",
            "unit": "meters, radians",
            "euler_convention": best_conv_name,
            "method_used": best_overall["method"],
            "R_cam2ee": R_cam2ee.tolist(),
            "t_cam2ee": t_cam2ee.flatten().tolist(),
            "T_cam2ee": T_cam2ee.tolist(),
            "num_poses": n,
            "timestamp": datetime.now().isoformat(),
            "validation": {
                "target_in_base_mean_m":  best_overall["mean_pt_m"].tolist(),
                "target_in_base_std_mm":  best_overall["std_pt_mm"].tolist(),
                "target_in_base_max_dev_mm": float(best_overall["max_dev_mm"])
            },
            "all_conventions_sorted": all_summaries,
        }

        out_path = self.save_dir / "cam2ee.json"
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✅ Eye-in-Hand 캘리브레이션 완료")
        print(f"{'='*60}")
        print(f"결과 저장: {out_path}")
        print(f"최적 컨벤션: {best_conv_name}")
        print(f"최적 방법:   {best_overall['method']}")
        print(f"\nR_cam2ee =\n{R_cam2ee}")
        t_flat = t_cam2ee.flatten()
        print(f"\nt_cam2ee (m)  = [{t_flat[0]:+.4f}, {t_flat[1]:+.4f}, {t_flat[2]:+.4f}]")
        print(f"t_cam2ee (mm) = [{t_flat[0]*1000:+.1f}, {t_flat[1]*1000:+.1f}, "
              f"{t_flat[2]*1000:+.1f}]")
        print(f"\n[검증] 표준편차 (mm): "
              f"[{best_overall['std_pt_mm'][0]:.1f}, "
              f"{best_overall['std_pt_mm'][1]:.1f}, "
              f"{best_overall['std_pt_mm'][2]:.1f}]")
        print(f"       최대 편차 (mm): {best_overall['max_dev_mm']:.1f}")
        print(f"\n목표: std_norm < 10mm, ||t|| < 150mm 이면 성공")
        return True


# ======================================================================
# main
# ======================================================================
def main():
    print("Eye-in-Hand Calibration for FR5 Robot")
    print("=" * 60)
    print("카메라가 EE에 부착되어 있고, ChArUco 보드는 월드에 고정된 구성")
    print("결과: T_cam2ee (카메라 -> 엔드이펙터)")
    print("=" * 60)

    calibrator = EyeInHandCalibrator(
        robot_ip="192.168.58.2",
        save_dir="./calib_data_eye_in_hand",
        charuco_squares_x=12,
        charuco_squares_y=8,
        charuco_square_length=46.0,
        charuco_marker_length=36.0,
        min_charuco_corners=15,
        min_detection_confidence=0.75,
        max_distance_threshold=0.3,
        far_pose_filter_mm=400.0,
    )

    try:
        print("\n=== 메뉴 ===")
        print("1. 실시간 포즈 수집 + 자동 계산")
        print("2. 이미 수집된 데이터로 계산만 수행")
        print("3. 종료")
        choice = input("선택 (1/2/3): ").strip()

        if choice == "1":
            n = int(input("수집할 포즈 수 (권장 25~35): ") or "30")
            print(f"\n{n}개 포즈 수집을 시작합니다...")
            print("💡 팁:")
            print("   - J4, J5, J6 세 관절을 **모두** 크게 움직여야 합니다")
            print("   - 수집 중 HUD의 노란색 J4/J5/J6 값을 매번 확인하세요")
            print("   - 이전 포즈와 최소 20도 이상 다른 값을 잡으세요")
            print("   - 거리는 150~250mm가 이상적\n")

            if calibrator.collect_poses_with_camera_feed(n):
                print("\n📊 수집 데이터로 T_cam2ee 계산 중...")
                calibrator.calculate_transformation_matrix()
            else:
                print("\n❌ 수집 실패 (최소 10개 필요)")

        elif choice == "2":
            calibrator.calculate_transformation_matrix()

        else:
            print("종료합니다.")

    except KeyboardInterrupt:
        print("\n사용자 중단")
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            calibrator.robot.DragTeachSwitch(0)
            print("🔒 안전 종료: 직접 교시 모드 OFF")
        except Exception:
            pass


if __name__ == "__main__":
    main()

'''
#!/usr/bin/env python3
"""
Eye-in-Hand Calibration for FR5 Robot (AGV-mounted version)
============================================================

카메라가 로봇 End-Effector에 부착되어 있고, ChArUco 보드는 월드(작업공간)에
고정되어 있는 Eye-in-Hand 구성에서의 핸드-아이 캘리브레이션.

이 버전의 AGV 세팅용 조정 사항:
- 카메라-보드 거리 파라미터를 멀리 있는 보드에 맞게 조정
- RealSense의 실제 왜곡 계수를 PnP에 사용 (기존: 0으로 설정)
- 품질 평가의 "이상적 거리" 범위를 파라미터화
- 기본값은 "보드를 테이블/상자 위에 올려둔 경우" (400~700mm)에 맞춤
- 바닥에 두는 경우 FLOOR_MODE 플래그로 쉽게 전환 가능

목표: T_cam2ee (카메라 -> 엔드이펙터) 변환 행렬을 구한다.

사전 요구사항:
    pip install scipy opencv-python pyrealsense2 numpy
"""

#!/usr/bin/env python3
"""
Eye-in-Hand Calibration for FR5 Robot (AGV-mounted version)
============================================================

카메라가 로봇 End-Effector에 부착되어 있고, ChArUco 보드는 월드(작업공간)에
고정되어 있는 Eye-in-Hand 구성에서의 핸드-아이 캘리브레이션.

이 버전의 AGV 세팅용 조정 사항:
- 카메라-보드 거리 파라미터를 멀리 있는 보드에 맞게 조정
- RealSense의 실제 왜곡 계수를 PnP에 사용 (기존: 0으로 설정)
- 품질 평가의 "이상적 거리" 범위를 파라미터화
- 기본값은 "보드를 테이블/상자 위에 올려둔 경우" (400~700mm)에 맞춤
- 바닥에 두는 경우 FLOOR_MODE 플래그로 쉽게 전환 가능

목표: T_cam2ee (카메라 -> 엔드이펙터) 변환 행렬을 구한다.

사전 요구사항:
    pip install scipy opencv-python pyrealsense2 numpy
"""

import cv2
import json
import time
import numpy as np
import pyrealsense2 as rs
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

from scipy.spatial.transform import Rotation as ScipyRotation
from fairino.Robot import RPC


# ======================================================================
# 캘리브레이션 거리 프로파일
# ======================================================================
# AGV 세팅에서 실제 사용 거리 범위에 맞춰 프로파일을 선택하세요.
# MODE 값으로 아래 프로파일 중 하나를 고릅니다.
# ======================================================================
#   "agv_floor":   AGV 위 FR5가 바닥 보드를 볼 때 (500~900mm) ← 기본값
#   "close_floor": 바닥에 보드를 두고 팔을 뻗어 가까이서 보는 경우 (200~500mm)
#   "table":       보드를 테이블/상자 위에 올려둔 경우 (300~700mm)
#   "far_floor":   바닥 보드를 아주 멀리서 보는 경우 (700~1500mm)
# ======================================================================

MODE = "agv_floor"   # ← 여기만 바꾸면 프로파일 전환

_PROFILES = {
    "agv_floor": {
        # AGV 위 FR5의 자연스러운 거리 (500~900mm)
        "ideal_distance_min":     0.5,    # 500mm부터 이상 범위
        "ideal_distance_max":     0.9,    # 900mm까지 이상 범위
        "max_distance_threshold": 1.0,    # 1000mm까지 저장 가능
        "far_pose_filter_mm":     1100.0, # 1100mm 초과 시 제외
        "min_charuco_corners":    20,     # 먼 거리라 코너 요구 상향
        "ideal_distance_text":    "500~900mm",
    },
    "close_floor": {
        "ideal_distance_min":     0.2,
        "ideal_distance_max":     0.5,
        "max_distance_threshold": 0.55,
        "far_pose_filter_mm":     700.0,
        "min_charuco_corners":    15,
        "ideal_distance_text":    "200~450mm",
    },
    "table": {
        "ideal_distance_min":     0.3,
        "ideal_distance_max":     0.7,
        "max_distance_threshold": 0.7,
        "far_pose_filter_mm":     800.0,
        "min_charuco_corners":    15,
        "ideal_distance_text":    "400~600mm",
    },
    "far_floor": {
        "ideal_distance_min":     0.7,
        "ideal_distance_max":     1.5,
        "max_distance_threshold": 1.5,
        "far_pose_filter_mm":     2000.0,
        "min_charuco_corners":    20,
        "ideal_distance_text":    "700~1200mm",
    },
}

if MODE not in _PROFILES:
    raise ValueError(f"MODE must be one of {list(_PROFILES.keys())}, got '{MODE}'")

DIST_PROFILE = _PROFILES[MODE]


class EyeInHandCalibrator:
    """Eye-in-Hand calibration을 수행하는 클래스."""

    def __init__(self,
                 robot_ip: str = "192.168.58.2",
                 save_dir: str = "./calib_data_eye_in_hand",
                 charuco_squares_x: int = 8,
                 charuco_squares_y: int = 12,
                 charuco_square_length: float = 45.0,   # mm
                 charuco_marker_length: float = 35.0,   # mm
                 min_charuco_corners: int = 15,
                 min_detection_confidence: float = 0.75,
                 # --- AGV 세팅에 맞춘 거리 파라미터 ---
                 ideal_distance_min: float = 0.5,
                 ideal_distance_max: float = 0.9,
                 max_distance_threshold: float = 1.0,
                 far_pose_filter_mm: float = 1100.0):
        self.robot_ip = robot_ip
        self.save_dir = Path(save_dir)
        self.poses_dir = self.save_dir / "poses"
        self.angles_dir = self.save_dir / "angles"

        self.charuco_squares_x = charuco_squares_x
        self.charuco_squares_y = charuco_squares_y
        self.charuco_square_length = charuco_square_length
        self.charuco_marker_length = charuco_marker_length

        self.min_charuco_corners = min_charuco_corners
        self.min_detection_confidence = min_detection_confidence

        self.ideal_distance_min = ideal_distance_min
        self.ideal_distance_max = ideal_distance_max
        self.max_distance_threshold = max_distance_threshold
        self.far_pose_filter_mm = far_pose_filter_mm

        # ChArUco 보드 (보드 생성 코드와 일치: 8x12, 45mm/35mm, DICT_5X5_100)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.charuco_board = cv2.aruco.CharucoBoard(
            size=(charuco_squares_x, charuco_squares_y),
            squareLength=self.charuco_square_length / 1000.0,
            markerLength=self.charuco_marker_length / 1000.0,
            dictionary=self.aruco_dict
        )

        # 로봇 연결
        self.robot = RPC(robot_ip)
        self.robot.DragTeachSwitch(0)

        # 디렉토리
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.poses_dir.mkdir(parents=True, exist_ok=True)
        self.angles_dir.mkdir(parents=True, exist_ok=True)

        # SDK 호출 방식 캐싱
        self._tcp_flag_mode = None
        self._joint_flag_mode = None

        print("=" * 60)
        print("Eye-in-Hand Calibrator 초기화 완료 (AGV 버전)")
        print("=" * 60)
        print(f"ChArUco 크기: {charuco_squares_x}x{charuco_squares_y}")
        print(f"사각형 길이: {charuco_square_length:.1f}mm")
        print(f"마커 길이:   {charuco_marker_length:.1f}mm")
        print(f"모드:        Eye-in-Hand (카메라가 EE에 부착)")
        print(f"목표:        T_cam2ee (카메라 -> 엔드이펙터) 계산")
        print(f"품질 기준:   최소 코너 {min_charuco_corners}개, "
              f"신뢰도 {min_detection_confidence}")
        print(f"이상 거리:   {ideal_distance_min*1000:.0f}~"
              f"{ideal_distance_max*1000:.0f}mm")
        print(f"최대 거리:   {max_distance_threshold*1000:.0f}mm (감점 임계)")
        print(f"원거리 필터: > {far_pose_filter_mm:.0f}mm 포즈는 자동 제외")
        print(f"컨벤션 테스트: scipy로 12가지 오일러각 컨벤션 모두 시도")
        print("=" * 60)

    # ------------------------------------------------------------------
    # FR5 SDK 안전 래퍼
    # ------------------------------------------------------------------
    def _safe_get_tcp_pose(self) -> Optional[List[float]]:
        """FR5의 GetActualTCPPose를 여러 호출 방식으로 시도."""
        attempts = []
        if self._tcp_flag_mode is not None:
            attempts.append(self._tcp_flag_mode)
        for mode in ["flag1", "flag0", "noarg"]:
            if mode not in attempts:
                attempts.append(mode)

        for mode in attempts:
            try:
                if mode == "flag1":
                    ret = self.robot.GetActualTCPPose(1)
                elif mode == "flag0":
                    ret = self.robot.GetActualTCPPose(0)
                else:
                    ret = self.robot.GetActualTCPPose()
            except TypeError:
                continue
            except Exception as e:
                print(f"\n⚠️ TCP 포즈 호출 예외 ({mode}): {e}")
                continue

            if isinstance(ret, tuple) and len(ret) >= 2:
                error_code, coords = ret[0], ret[1]
                if error_code == 0 and coords is not None and len(coords) >= 6:
                    self._tcp_flag_mode = mode
                    return list(coords)

            if isinstance(ret, int):
                continue

        print(f"\n⚠️ TCP 포즈 모든 호출 방식 실패")
        return None

    def _safe_get_joint_angles(self) -> Optional[List[float]]:
        """FR5의 GetActualJointPosDegree를 여러 호출 방식으로 시도."""
        attempts = []
        if self._joint_flag_mode is not None:
            attempts.append(self._joint_flag_mode)
        for mode in ["flag1", "flag0", "noarg"]:
            if mode not in attempts:
                attempts.append(mode)

        for mode in attempts:
            try:
                if mode == "flag1":
                    ret = self.robot.GetActualJointPosDegree(1)
                elif mode == "flag0":
                    ret = self.robot.GetActualJointPosDegree(0)
                else:
                    ret = self.robot.GetActualJointPosDegree()
            except TypeError:
                continue
            except Exception as e:
                print(f"\n⚠️ 관절 각도 호출 예외 ({mode}): {e}")
                continue

            if isinstance(ret, tuple) and len(ret) >= 2:
                error_code, angles = ret[0], ret[1]
                if error_code == 0 and angles is not None and len(angles) >= 6:
                    self._joint_flag_mode = mode
                    return list(angles)

            if isinstance(ret, int):
                continue

        print(f"\n⚠️ 관절 각도 모든 호출 방식 실패")
        return None

    # ------------------------------------------------------------------
    # ChArUco 검출
    # ------------------------------------------------------------------
    def detect_charuco_pose(self, image: np.ndarray,
                            camera_matrix: np.ndarray,
                            dist_coeffs: np.ndarray
                            ) -> Optional[Tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]]:
        """이미지에서 ChArUco 보드의 카메라 기준 포즈(target2cam)를 검출."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)

        charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            charuco_detector.detectBoard(gray_enhanced)

        if (charuco_corners is not None and charuco_ids is not None
                and len(charuco_corners) >= 6
                and len(charuco_corners) == len(charuco_ids)):

            obj_points, img_points = self.charuco_board.matchImagePoints(
                charuco_corners, charuco_ids)

            if (obj_points is not None and img_points is not None
                    and len(obj_points) >= 6):
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, img_points,
                    camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    return rvec, tvec, charuco_corners, charuco_ids

        return None

    # ------------------------------------------------------------------
    # 로봇 포즈 / 관절
    # ------------------------------------------------------------------
    def get_robot_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        coords = self._safe_get_tcp_pose()
        if coords is None:
            coords = [0.0] * 6

        position = np.array(coords[0:3]) / 1000.0
        R_gripper2base = ScipyRotation.from_euler(
            "zyx", coords[3:6], degrees=True
        ).as_matrix()
        return R_gripper2base, position.reshape(3, 1)

    def get_robot_angles(self) -> List[float]:
        angles = self._safe_get_joint_angles()
        return angles if angles is not None else [0.0] * 6

    # ------------------------------------------------------------------
    # 품질 평가
    # ------------------------------------------------------------------
    def evaluate_charuco_quality(self, charuco_corners, charuco_ids,
                                 rvec, tvec) -> Tuple[bool, str, float]:
        quality_score = 0.0
        issues = []

        corner_count = len(charuco_corners) if charuco_corners is not None else 0
        if charuco_corners is None or corner_count < self.min_charuco_corners:
            issues.append(f"코너 부족 ({corner_count}/{self.min_charuco_corners})")
            quality_score -= 0.1
        else:
            corner_ratio = corner_count / (
                (self.charuco_squares_x - 1) * (self.charuco_squares_y - 1)
            )
            quality_score += corner_ratio * 0.6

        distance = float(np.linalg.norm(tvec))
        if distance > self.max_distance_threshold:
            issues.append(f"거리 과대 ({distance*1000:.0f}mm)")
            quality_score -= 0.2
        elif distance < self.ideal_distance_min:
            issues.append(f"거리 과소 ({distance*1000:.0f}mm)")
            quality_score -= 0.1
        else:
            distance_score = 1.0 - (distance - self.ideal_distance_min) / \
                             (self.ideal_distance_max - self.ideal_distance_min)
            quality_score += max(0, distance_score) * 0.3

        R_mat, _ = cv2.Rodrigues(rvec)
        z_axis = np.array([0, 0, 1])
        board_z_in_cam = R_mat @ z_axis
        angle_deg = np.degrees(
            np.arccos(np.clip(np.abs(np.dot(board_z_in_cam, z_axis)), 0, 1))
        )
        if angle_deg > 75:
            issues.append(f"각도 극단 ({angle_deg:.0f}도)")
            quality_score -= 0.1
        else:
            quality_score += (1.0 - angle_deg / 75.0) * 0.2

        if charuco_corners is not None and corner_count > 0:
            try:
                arr = np.array(charuco_corners).reshape(-1, 2)
                if np.std(arr[:, 0]) < 30 or np.std(arr[:, 1]) < 30:
                    issues.append("코너 분포 집중")
                    quality_score -= 0.05
            except Exception:
                pass

        quality_score += 0.3

        is_good = quality_score >= self.min_detection_confidence
        msg = (f"품질 양호 ({quality_score:.2f})" if is_good
               else f"품질 불량 ({quality_score:.2f}) - {'; '.join(issues)}")
        return is_good, msg, float(quality_score)

    # ------------------------------------------------------------------
    # 데이터 수집
    # ------------------------------------------------------------------
    def collect_poses_with_camera_feed(self, num_poses: int = 25) -> bool:
        print(f"\n=== 실시간 포즈 수집 (Eye-in-Hand, AGV 버전) ===")
        print(f"수집할 포즈 수: {num_poses}")
        print(f"이상 거리 범위: {self.ideal_distance_min*1000:.0f}~"
              f"{self.ideal_distance_max*1000:.0f}mm")
        print("카메라는 EE에 부착되어 있습니다. 고정된 ChArUco 보드를")
        print("여러 각도/거리/회전으로 바라보도록 로봇을 움직여 주세요.")
        print("⚠️  J4, J5, J6 모두 크게 변화시켜야 합니다!\n")

        config = rs.config()
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

        pipeline = rs.pipeline()
        profile = pipeline.start(config)
        print("✅ RealSense 카메라 연결 성공 (848x480)")

        color_profile = profile.get_stream(rs.stream.color)
        color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        camera_matrix = np.array([
            [color_intrinsics.fx, 0.0, color_intrinsics.ppx],
            [0.0, color_intrinsics.fy, color_intrinsics.ppy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        raw_coeffs = list(color_intrinsics.coeffs)
        while len(raw_coeffs) < 5:
            raw_coeffs.append(0.0)
        dist_coeffs = np.array(raw_coeffs[:5], dtype=np.float64)

        print(f"  - 해상도: {color_intrinsics.width}x{color_intrinsics.height}")
        print(f"  - fx={color_intrinsics.fx:.2f}, fy={color_intrinsics.fy:.2f}")
        print(f"  - cx={color_intrinsics.ppx:.2f}, cy={color_intrinsics.ppy:.2f}")
        print(f"  - 왜곡 모델: {color_intrinsics.model}")
        print(f"  - 왜곡 계수: {[f'{c:.4f}' for c in dist_coeffs]}\n")

        collected = 0
        pose_index = 0

        self.robot.DragTeachSwitch(1)
        print("🤚 직접 교시 모드 ON: 로봇을 손으로 움직여 포즈를 잡으세요.\n")

        try:
            while collected < num_poses:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = (np.asanyarray(depth_frame.get_data())
                               if depth_frame else None)

                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray_enh = clahe.apply(gray)
                vis_image = cv2.cvtColor(gray_enh, cv2.COLOR_GRAY2BGR)

                charuco_detected = False
                quality_score = 0.0
                distance = 0.0
                corner_count = 0
                charuco_result = self.detect_charuco_pose(
                    color_image, camera_matrix, dist_coeffs)

                if charuco_result is not None:
                    rvec, tvec, charuco_corners, charuco_ids = charuco_result
                    charuco_detected = True
                    distance = float(np.linalg.norm(tvec))
                    corner_count = len(charuco_corners)

                    # 검출된 코너 시각화
                    cv2.aruco.drawDetectedCornersCharuco(
                        vis_image, charuco_corners, charuco_ids)

                    is_good, _, quality_score = self.evaluate_charuco_quality(
                        charuco_corners, charuco_ids, rvec, tvec)
                    if is_good:
                        cv2.drawFrameAxes(vis_image, camera_matrix,
                                          dist_coeffs, rvec, tvec, 0.05)

                try:
                    angles = self.get_robot_angles()
                    coords = self._safe_get_tcp_pose() or [0.0] * 6
                except Exception:
                    angles = [0.0] * 6
                    coords = [0.0] * 6

                # HUD
                cv2.putText(vis_image, f"Pose: {collected}/{num_poses}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if charuco_detected:
                    if self.ideal_distance_min <= distance <= self.ideal_distance_max:
                        dist_color = (0, 255, 0)
                    elif distance <= self.max_distance_threshold:
                        dist_color = (0, 255, 255)
                    else:
                        dist_color = (0, 0, 255)
                    cv2.putText(vis_image,
                                f"ChArUco: {corner_count} corners, "
                                f"{distance*1000:.0f}mm",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                dist_color, 2)
                else:
                    cv2.putText(vis_image, "ChArUco: NOT DETECTED", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                q_color = ((0, 255, 0)
                           if quality_score >= self.min_detection_confidence
                           else (0, 0, 255))
                cv2.putText(vis_image, f"Quality: {quality_score:.2f}",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, q_color, 2)

                cv2.putText(vis_image,
                            f"J1-J3: [{angles[0]:.1f},{angles[1]:.1f},{angles[2]:.1f}]",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (200, 200, 200), 1)
                cv2.putText(vis_image,
                            f"J4={angles[3]:+6.1f}  J5={angles[4]:+6.1f}  "
                            f"J6={angles[5]:+6.1f}",
                            (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 255), 2)
                cv2.putText(vis_image,
                            f"TCP: [{coords[0]:.1f},{coords[1]:.1f},"
                            f"{coords[2]:.1f}]",
                            (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1)

                if charuco_detected:
                    cv2.putText(vis_image, "Press 's' to save, 'q' to quit",
                                (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)
                else:
                    cv2.putText(vis_image, "No board - cannot save",
                                (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)

                cv2.imshow('Eye-in-Hand Calibration (AGV)', vis_image)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print(f"\n사용자 중단. 수집된 포즈: {collected}")
                    break

                elif key == ord('s'):
                    print("\n🔒 로봇 고정 + 2초 안정화...")
                    self.robot.DragTeachSwitch(0)
                    time.sleep(2.0)

                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if not color_frame:
                        self.robot.DragTeachSwitch(1)
                        continue

                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = (np.asanyarray(depth_frame.get_data())
                                   if depth_frame else None)

                    charuco_result = self.detect_charuco_pose(
                        color_image, camera_matrix, dist_coeffs)
                    if charuco_result is None:
                        print("❌ 저장 실패: ChArUco 검출 안됨")
                        self.robot.DragTeachSwitch(1)
                        continue

                    rvec, tvec, charuco_corners, charuco_ids = charuco_result
                    corner_count = len(charuco_corners)
                    if corner_count < 6:
                        print(f"❌ 저장 실패: 코너 {corner_count}개 (< 6)")
                        self.robot.DragTeachSwitch(1)
                        continue

                    distance = float(np.linalg.norm(tvec))

                    if distance * 1000 > self.far_pose_filter_mm:
                        print(f"❌ 저장 실패: 거리 {distance*1000:.0f}mm > "
                              f"{self.far_pose_filter_mm:.0f}mm")
                        self.robot.DragTeachSwitch(1)
                        continue

                    is_good, _, quality_score = self.evaluate_charuco_quality(
                        charuco_corners, charuco_ids, rvec, tvec)

                    if not is_good:
                        print(f"❌ 품질 불충분 ({quality_score:.2f}), 저장 안 함")
                        self.robot.DragTeachSwitch(1)
                        continue

                    # --- 저장 ---
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    angles = self.get_robot_angles()
                    coords = self._safe_get_tcp_pose() or [0.0] * 6

                    angle_path = (self.angles_dir /
                                  f"{pose_index:02d}_{timestamp}_angles.json")
                    with open(angle_path, 'w') as f:
                        json.dump({
                            "angles": list(angles),
                            "coords": list(coords),
                            "pose_index": pose_index,
                            "timestamp": timestamp
                        }, f, indent=2)

                    R_g2b, t_g2b = self.get_robot_pose()
                    pose_path = (self.poses_dir /
                                 f"{pose_index:02d}_{timestamp}_pose.json")
                    with open(pose_path, 'w') as f:
                        json.dump({
                            "R_gripper2base": R_g2b.tolist(),
                            "t_gripper2base": t_g2b.tolist(),
                            "angles": list(angles),
                            "coords": list(coords),
                            "pose_index": pose_index,
                            "timestamp": timestamp
                        }, f, indent=2)

                    color_path = (self.save_dir / "color" /
                                  f"{pose_index:02d}_{timestamp}.jpg")
                    color_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(color_path), color_image)

                    depth_path = None
                    if depth_image is not None:
                        depth_path = (self.save_dir / "depth" /
                                      f"{pose_index:02d}_{timestamp}.npy")
                        depth_path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(str(depth_path), depth_image)

                        depth_png = (self.save_dir / "depth" / "converted_png"
                                     / f"{pose_index:02d}_{timestamp}.png")
                        depth_png.parent.mkdir(parents=True, exist_ok=True)
                        depth_norm = cv2.normalize(
                            depth_image, None, 0, 255,
                            cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        cv2.imwrite(str(depth_png), depth_norm)

                    intrinsics_path = (self.save_dir / "intrinsics" /
                                       f"{pose_index:02d}_{timestamp}.json")
                    intrinsics_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(intrinsics_path, 'w') as f:
                        json.dump({
                            "color_intrinsics": {
                                "width": color_intrinsics.width,
                                "height": color_intrinsics.height,
                                "fx": float(color_intrinsics.fx),
                                "fy": float(color_intrinsics.fy),
                                "ppx": float(color_intrinsics.ppx),
                                "ppy": float(color_intrinsics.ppy),
                                "distortion_model": str(color_intrinsics.model),
                                "distortion_coeffs":
                                    [float(c) for c in color_intrinsics.coeffs]
                            },
                            "camera_matrix": camera_matrix.tolist(),
                            "dist_coeffs": dist_coeffs.tolist(),
                            "timestamp": datetime.now().isoformat()
                        }, f, indent=2)

                    charuco_path = (self.save_dir /
                                    f"{pose_index:02d}_{timestamp}_charuco.json")
                    with open(charuco_path, 'w') as f:
                        json.dump({
                            "rvec_target2cam": rvec.tolist(),
                            "tvec_target2cam": tvec.tolist(),
                            "pose_index": pose_index,
                            "timestamp": timestamp,
                            "charuco_corners_count": int(corner_count),
                            "distance_mm": float(distance * 1000),
                            "quality_score": float(quality_score),
                            "detection_method": "charuco",
                            "color_image_path": str(color_path),
                            "depth_image_path":
                                str(depth_path) if depth_path else None,
                            "intrinsics_path": str(intrinsics_path)
                        }, f, indent=2)

                    print(f"✅ 포즈 {pose_index} 저장 완료 "
                          f"(거리 {distance*1000:.0f}mm, 코너 {corner_count}개, "
                          f"품질 {quality_score:.2f}, "
                          f"J4={angles[3]:.1f} J5={angles[4]:.1f} "
                          f"J6={angles[5]:.1f})")

                    collected += 1
                    pose_index += 1

                    self.robot.DragTeachSwitch(1)
                    print("🤚 직접 교시 모드 ON: 다음 포즈로 이동하세요.\n")

        except KeyboardInterrupt:
            print(f"\nKeyboardInterrupt. 수집된 포즈: {collected}")
        except Exception as e:
            print(f"\n❌ 오류: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.robot.DragTeachSwitch(0)
            print("\n🔒 직접 교시 모드 OFF")
            try:
                pipeline.stop()
            except Exception:
                pass
            cv2.destroyAllWindows()

        print(f"\n{'='*60}")
        print(f" 수집 완료: 총 {collected}개 포즈")
        print(f"{'='*60}")
        return collected >= 10

    # ------------------------------------------------------------------
    # 회전 다양성 사전 검증
    # ------------------------------------------------------------------
    def _check_rotation_diversity(self, R_list: List[np.ndarray]) -> None:
        if len(R_list) < 2:
            return

        angles = []
        for i in range(len(R_list)):
            for j in range(i + 1, len(R_list)):
                R_rel = R_list[j] @ R_list[i].T
                cos_theta = np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0)
                angles.append(np.degrees(np.arccos(cos_theta)))

        angles = np.array(angles)
        print(f"\n[회전 다양성 검증]")
        print(f"  상대 회전 각도: min={angles.min():.1f}°, "
              f"max={angles.max():.1f}°, mean={angles.mean():.1f}°")
        if angles.max() < 30:
            print(f"  ⚠️ 경고: 모든 포즈의 상대 회전이 30° 미만입니다.")
        elif angles.mean() < 20:
            print(f"  ⚠️ 주의: 평균 상대 회전이 20° 미만입니다.")
        else:
            print(f"  ✅ 회전 다양성 양호")

    # ------------------------------------------------------------------
    # Eye-in-Hand 변환 행렬 계산
    # pose 파일 매칭 버그 수정: 타임스탬프 정확 매칭으로 변경
    # ------------------------------------------------------------------
    def calculate_transformation_matrix(self) -> bool:
        print("\n=== Eye-in-Hand 변환 행렬 계산 (12 컨벤션 × 5 방법) ===")

        charuco_files = sorted(self.save_dir.glob("*_charuco.json"),
                               key=lambda x: int(x.stem.split('_')[0]))

        if len(charuco_files) < 3:
            print(f"❌ 데이터 부족 (필요: 3+, 현재: {len(charuco_files)})")
            return False

        # 데이터 로드 (타임스탬프 정확 매칭)
        pose_data = []
        for charuco_file in charuco_files:
            stem = charuco_file.stem  # 예: "00_2026-04-22_14-42-50_charuco"
            parts = stem.split('_')
            if len(parts) < 4:
                print(f"⚠️ 파일명 형식 이상: {charuco_file.name}")
                continue
            idx_str = parts[0]
            ts_str = f"{parts[1]}_{parts[2]}"
            idx = int(idx_str)

            # 같은 타임스탬프의 pose 파일
            pose_file = self.poses_dir / f"{idx_str}_{ts_str}_pose.json"
            if not pose_file.exists():
                print(f"⚠️ 포즈 {idx}: 매칭 pose 파일 없음, 건너뜀")
                continue

            with open(charuco_file, 'r') as f:
                cdata = json.load(f)
            with open(pose_file, 'r') as f:
                pdata = json.load(f)

            dist_mm = cdata.get("distance_mm", 0)
            if dist_mm > self.far_pose_filter_mm:
                print(f"  포즈 {idx:02d}: 거리 {dist_mm:.0f}mm > "
                      f"{self.far_pose_filter_mm:.0f}mm, 제외")
                continue

            rvec = np.array(cdata["rvec_target2cam"],
                            dtype=np.float64).reshape(3, 1)
            tvec = np.array(cdata["tvec_target2cam"],
                            dtype=np.float64).reshape(3, 1)
            R_t2c, _ = cv2.Rodrigues(rvec)

            pose_data.append({
                "idx": idx,
                "R_t2c": R_t2c,
                "t_t2c": tvec,
                "coords": pdata["coords"],
                "quality": cdata.get("quality_score", 0.5),
                "dist_mm": dist_mm,
            })

        n = len(pose_data)
        if n < 3:
            print(f"❌ 유효 포즈 부족: {n}")
            return False

        print(f"\n총 {n}개 포즈 사용")

        self._check_rotation_diversity([d["R_t2c"] for d in pose_data])

        conventions = [
            "xyz", "xzy", "yxz", "yzx", "zxy", "zyx",
            "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX",
        ]

        methods = {
            "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
            "PARK":       cv2.CALIB_HAND_EYE_PARK,
            "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
            "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
            "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
        }

        best_overall = None
        best_conv_name = None
        best_std_norm = float('inf')
        all_summaries = []

        for conv_name in conventions:
            try:
                R_gripper2base_list = []
                t_gripper2base_list = []
                R_target2cam_list = []
                t_target2cam_list = []

                for d in pose_data:
                    coords = d["coords"]
                    position = np.array(coords[0:3]) / 1000.0
                    R_g2b = ScipyRotation.from_euler(
                        conv_name, coords[3:6], degrees=True
                    ).as_matrix()
                    t_g2b = position.reshape(3, 1)

                    R_gripper2base_list.append(R_g2b)
                    t_gripper2base_list.append(t_g2b)
                    R_target2cam_list.append(d["R_t2c"])
                    t_target2cam_list.append(d["t_t2c"])

                results = {}
                for name, method in methods.items():
                    try:
                        R_c2e, t_c2e = cv2.calibrateHandEye(
                            R_gripper2base_list, t_gripper2base_list,
                            R_target2cam_list, t_target2cam_list,
                            method=method
                        )
                        results[name] = (R_c2e, t_c2e)
                    except Exception:
                        pass

                if not results:
                    continue

                names = list(results.keys())
                t_list = np.array([results[nm][1].flatten() for nm in names])
                median_t = np.median(t_list, axis=0)
                dists = np.linalg.norm(t_list - median_t, axis=1)
                chosen = names[int(np.argmin(dists))]

                R_cam2ee, t_cam2ee = results[chosen]
                T_cam2ee = np.eye(4)
                T_cam2ee[:3, :3] = R_cam2ee
                T_cam2ee[:3, 3] = t_cam2ee.flatten()

                target_pts = []
                for R_g, t_g, R_t, t_t in zip(R_gripper2base_list,
                                              t_gripper2base_list,
                                              R_target2cam_list,
                                              t_target2cam_list):
                    T_g = np.eye(4)
                    T_g[:3, :3] = R_g
                    T_g[:3, 3] = t_g.flatten()
                    T_t = np.eye(4)
                    T_t[:3, :3] = R_t
                    T_t[:3, 3] = t_t.flatten()
                    target_pts.append((T_g @ T_cam2ee @ T_t)[:3, 3])

                target_pts = np.array(target_pts)
                std_pt = target_pts.std(axis=0)
                std_norm = float(np.linalg.norm(std_pt))
                max_dev = float(np.linalg.norm(
                    target_pts - target_pts.mean(axis=0), axis=1).max())
                t_norm = float(np.linalg.norm(t_cam2ee))

                all_summaries.append({
                    "conv": conv_name,
                    "method": chosen,
                    "std_norm_mm": std_norm * 1000,
                    "max_dev_mm": max_dev * 1000,
                    "t_norm_mm": t_norm * 1000,
                })

                if std_norm < best_std_norm:
                    best_std_norm = std_norm
                    best_overall = {
                        "R_cam2ee": R_cam2ee,
                        "t_cam2ee": t_cam2ee,
                        "method": chosen,
                        "std_pt_mm": std_pt * 1000,
                        "max_dev_mm": max_dev * 1000,
                        "mean_pt_m": target_pts.mean(axis=0),
                    }
                    best_conv_name = conv_name

            except Exception as e:
                print(f"  컨벤션 {conv_name} 계산 중 오류: {e}")
                continue

        all_summaries.sort(key=lambda x: x["std_norm_mm"])
        print(f"\n{'='*72}")
        print(f"🏆 모든 컨벤션 결과 (std_norm 오름차순)")
        print(f"{'='*72}")
        print(f"  {'컨벤션':10s} {'방법':12s} {'std_norm':>12s} "
              f"{'max_dev':>12s} {'||t||':>12s}")
        print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for i, s in enumerate(all_summaries):
            marker = " ← 최선" if i == 0 else ""
            print(f"  {s['conv']:10s} {s['method']:12s} "
                  f"{s['std_norm_mm']:10.1f}mm "
                  f"{s['max_dev_mm']:10.1f}mm "
                  f"{s['t_norm_mm']:10.1f}mm{marker}")

        if len(all_summaries) >= 3:
            top_stds = [s["std_norm_mm"] for s in all_summaries[:3]]
            if top_stds[0] < top_stds[1] * 0.5:
                print(f"\n⚠️ 주의: 1등이 2등보다 2배 이상 좋습니다.")
            else:
                print(f"\n✅ 상위 결과들이 일관적입니다.")

        if best_overall is None:
            print("\n❌ 모든 컨벤션/방법 실패")
            return False

        R_cam2ee = best_overall["R_cam2ee"]
        t_cam2ee = best_overall["t_cam2ee"]
        T_cam2ee = np.eye(4)
        T_cam2ee[:3, :3] = R_cam2ee
        T_cam2ee[:3, 3] = t_cam2ee.flatten()

        result = {
            "description": "Eye-in-Hand calibration: 카메라 -> 엔드이펙터 변환",
            "unit": "meters, radians",
            "euler_convention": best_conv_name,
            "method_used": best_overall["method"],
            "R_cam2ee": R_cam2ee.tolist(),
            "t_cam2ee": t_cam2ee.flatten().tolist(),
            "T_cam2ee": T_cam2ee.tolist(),
            "num_poses": n,
            "board_config": {
                "squares_x": self.charuco_squares_x,
                "squares_y": self.charuco_squares_y,
                "square_length_mm": self.charuco_square_length,
                "marker_length_mm": self.charuco_marker_length,
            },
            "timestamp": datetime.now().isoformat(),
            "validation": {
                "target_in_base_mean_m": best_overall["mean_pt_m"].tolist(),
                "target_in_base_std_mm": best_overall["std_pt_mm"].tolist(),
                "target_in_base_max_dev_mm": float(best_overall["max_dev_mm"])
            },
            "all_conventions_sorted": all_summaries,
        }

        out_path = self.save_dir / "cam2ee.json"
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✅ Eye-in-Hand 캘리브레이션 완료")
        print(f"{'='*60}")
        print(f"결과 저장: {out_path}")
        print(f"최적 컨벤션: {best_conv_name}")
        print(f"최적 방법:   {best_overall['method']}")
        print(f"\nR_cam2ee =\n{R_cam2ee}")
        t_flat = t_cam2ee.flatten()
        print(f"\nt_cam2ee (m)  = [{t_flat[0]:+.4f}, {t_flat[1]:+.4f}, "
              f"{t_flat[2]:+.4f}]")
        print(f"t_cam2ee (mm) = [{t_flat[0]*1000:+.1f}, "
              f"{t_flat[1]*1000:+.1f}, {t_flat[2]*1000:+.1f}]")
        print(f"\n[검증] 표준편차 (mm): "
              f"[{best_overall['std_pt_mm'][0]:.1f}, "
              f"{best_overall['std_pt_mm'][1]:.1f}, "
              f"{best_overall['std_pt_mm'][2]:.1f}]")
        print(f"       최대 편차 (mm): {best_overall['max_dev_mm']:.1f}")
        print(f"\n목표: std_norm < 15mm, ||t|| < 200mm 이면 성공")
        return True


# ======================================================================
# main
# ======================================================================
def main():
    print("Eye-in-Hand Calibration for FR5 Robot (AGV-mounted)")
    print("=" * 60)
    print("카메라가 EE에 부착되어 있고, ChArUco 보드는 월드에 고정된 구성")
    print(f"현재 프로파일: MODE = '{MODE}'")
    print(f"이상 거리:    {DIST_PROFILE['ideal_distance_text']}")
    print("결과: T_cam2ee (카메라 -> 엔드이펙터)")
    print("=" * 60)

    calibrator = EyeInHandCalibrator(
        robot_ip="192.168.58.2",
        save_dir="./calib_data_eye_in_hand",
        charuco_squares_x=8,
        charuco_squares_y=12,
        charuco_square_length=45.0,
        charuco_marker_length=35.0,
        min_charuco_corners=DIST_PROFILE["min_charuco_corners"],
        min_detection_confidence=0.75,
        ideal_distance_min=DIST_PROFILE["ideal_distance_min"],
        ideal_distance_max=DIST_PROFILE["ideal_distance_max"],
        max_distance_threshold=DIST_PROFILE["max_distance_threshold"],
        far_pose_filter_mm=DIST_PROFILE["far_pose_filter_mm"],
    )

    try:
        print("\n=== 메뉴 ===")
        print("1. 실시간 포즈 수집 + 자동 계산")
        print("2. 이미 수집된 데이터로 계산만 수행")
        print("3. 종료")
        choice = input("선택 (1/2/3): ").strip()

        if choice == "1":
            n = int(input("수집할 포즈 수 (권장 25~30): ") or "25")
            print(f"\n{n}개 포즈 수집을 시작합니다...")
            print("💡 팁:")
            print("   - J4, J5, J6 세 관절을 **모두** 크게 움직여야 합니다")
            print("   - 수집 중 HUD의 노란색 J4/J5/J6 값을 매번 확인하세요")
            print("   - 이전 포즈와 최소 20도 이상 다른 값을 잡으세요")
            print(f"   - 이상적 거리: {DIST_PROFILE['ideal_distance_text']}")
            print("   - HUD의 거리 숫자가 초록색이면 이상 범위 안")
            print("   - 보드가 흔들리지 않도록 단단히 고정 확인\n")

            if calibrator.collect_poses_with_camera_feed(n):
                print("\n📊 수집 데이터로 T_cam2ee 계산 중...")
                calibrator.calculate_transformation_matrix()
            else:
                print("\n❌ 수집 실패 (최소 10개 필요)")

        elif choice == "2":
            calibrator.calculate_transformation_matrix()

        else:
            print("종료합니다.")

    except KeyboardInterrupt:
        print("\n사용자 중단")
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            calibrator.robot.DragTeachSwitch(0)
            print("🔒 안전 종료: 직접 교시 모드 OFF")
        except Exception:
            pass


if __name__ == "__main__":
    main()