# 이동 매니퓰레이터 기반 차량 외관 결함 자동 검사 시스템

AGV가 차량 주위를 자율 순회하고, 로봇팔 말단의 RGB-D 카메라로 **덴트(dent)와 스크래치(scratch)** 를 탐지·분류한 뒤, 결함 정면으로 접근·보정하여 근접 촬영하는 완전 자동 검사 시스템입니다.

<!-- 전체 동작 GIF: 순회 → 탐지 → 팔 접근 → 촬영 시퀀스 -->

---

## 시스템 흐름

```
[AGV] 라이다로 검사 대상 인식 → 대상 둘레 웨이포인트 자동 생성 → Nav2로 순회
   ↓ 각 웨이포인트 도착
[탐지] RealSense + YOLO 세그멘테이션 → dent / scratch 마스크
   ↓
[위치 추정] hand-eye 변환 + 순차적 다중 평면 RANSAC → 결함 3D 위치·표면 법선
   ↓
[접근] 법선 방향 정면 접근 (J6 고정 IK)  ※ 작업 반경 밖이면 AGV 추가 접근 (2단계 접근)
   ↓
[보정·촬영] 화면 역투영 기반 시각 보정으로 결함을 화면 중앙 정렬 → 근접 촬영 → 팔 복귀
   ↓
다음 웨이포인트 → 한 바퀴 완료 후 시작점 복귀
```

## 주요 특징

- **탐지부터 촬영까지 전 과정 자동화** — 사람 개입 없이 대상 인식, 순회, 탐지, 접근, 촬영 수행
- **순차적 다중 평면 RANSAC** — 서로 다른 각도의 표면 위 결함도 위치와 법선을 각각 강인하게 추정
- **시각 보정 루프** — 좌표 변환 누적 오차를 화면 역투영 기반 반복 보정으로 제거
- **2단계 접근 전략** — 로봇팔 리치 밖의 결함은 AGV가 추가 접근 후 재시도

## 실험 결과

| 항목 | 결과 |
|---|---|
| 결함 탐지 (Mask mAP50) | **0.979** (dent 0.972 / scratch 0.986) |
| 시각 보정 정렬 오차 | **9.2 cm → 0.6 cm** 수렴 |
| 통합 파이프라인 | 대상 인식 → 순회 → 탐지 → 정면 접근 → 촬영 전 과정 실증 |


---

## 하드웨어

| 구성 | 모델 |
|---|---|
| 이동 로봇 | AgileX Ranger Mini |
| 라이다 | RPLIDAR [A2M12] |
| 로봇팔 | Fairino FR5 (펌웨어 v3.8.4.1) |
| 카메라 | Intel RealSense [D455] (Eye-in-Hand) |

## 소프트웨어 환경

- Ubuntu 20.04 / ROS2 Foxy
- slam_toolbox, Nav2
- Python 3.8
- ultralytics (YOLO), pyrealsense2, OpenCV, NumPy, SciPy

---

## 설치

### 1. Fairino Python SDK

로봇 펌웨어 버전과 SDK 버전이 맞아야 합니다. (v3.8.4.1 → SDK v2.1.4)

```bash
cd ~
git clone https://github.com/FAIR-INNOVATION/fairino-python-sdk.git
cd fairino-python-sdk
git checkout dc55385
```

### 2. Python 패키지

```bash
pip3 install ultralytics pyrealsense2 opencv-python numpy scipy
```

### 3. 이 레포

```bash
git clone https://github.com/ansdngud/car_inspection.git
cd car_inspection
export PYTHONPATH=~/fairino-python-sdk/linux:$PYTHONPATH
```

### 4. 모델 가중치

[Releases](../../releases)에서 `dent_best.pt`를 받아 경로에 둡니다.

---

## 실행

### 사전 준비

```bash
# FR5 통신용 IP 설정 (재부팅 시 초기화됨)
sudo ip addr add 192.168.58.22/24 dev enp2s0
```

### 실행 순서

```bash
# 1. AGV 구동 (CAN)
sudo modprobe gs_usb
sudo ip link set can0 up type can bitrate 500000

# 2. 로봇 본체
cd ~/ranger_ws
source install/setup.bash
ros2 run ranger_base ranger_base_node --ros-args -p publish_odom_tf:=true

# 3. odom 토픽 발행 확인
source /opt/ros/foxy/setup.bash
ros2 topic hz /odom

# 4. 라이다
cd ~/ros2_ws
source install/setup.bash
ros2 launch sllidar_ros2 view_sllidar_a2m12_launch.py

# 5. TF 연결 (base_link <-> laser)
source /opt/ros/foxy/setup.bash
ros2 run tf2_ros static_transform_publisher 0.15 0 0.30 3.14159 0 0 base_link laser

# 6. NAV2 실행
source /opt/ros/foxy/setup.bash
ros2 launch nav2_bringup bringup_launch.py map:=$HOME/my_map.yaml params_file:=$HOME/nav2_params.yaml use_sim_time:=false autostart:=true

# 7. NAV2용 RViz 실행
source /opt/ros/foxy/setup.bash
ros2 launch nav2_bringup rviz_launch.py

# 8. 통합 코드 실행
cd ~/fairino-python-sdk/linux
source /opt/ros/foxy/setup.bash
python3 full_inspection.py
```

> ⚠️ 로봇팔이 자동으로 움직이므로 실행 중에는 항상 비상정지 버튼을 손에 두세요.

### 주요 설정값 (`full_inspection.py` 상단)

| 변수 | 의미 |
|---|---|
| `MODEL_PATH` | YOLO 가중치 경로 |
| `CAM2EE_JSON` | hand-eye 캘리브레이션 결과 경로 |
| `ROBOT_IP` | FR5 IP (기본 `192.168.58.2`) |
| `STANDOFF` | 촬영 시 결함 표면과의 거리 (m) |
| `NUM_WAYPOINTS` | 순회 웨이포인트 수 |
| `SAVE_DIR` | 촬영 사진 저장 폴더 |

### 출력

촬영 사진은 `SAVE_DIR`에 웨이포인트 번호와 결함 번호가 붙은 파일명으로 저장됩니다.

---

## 모델 학습

```bash
# labelme 라벨(dent, scratch) → YOLO 세그멘테이션 형식 변환
python3 train/json2yolo.py

# 학습
yolo segment train data=training/data.yaml model=[yolo26n-seg.pt] epochs=[150] imgsz=640

# 학습 (yolo26n-seg, epochs=150, batch=8, imgsz=640, patience=50)
python3 training/train.py

사전학습 모델 `yolo26n-seg.pt`를 기반으로 학습했으며, 조기 종료(patience=50)로 117 에폭에서 학습이 종료되었고 67 에폭의 모델을 최종 모델로 사용했습니다. (RTX 3090)
```

실험 대상 시편(차량 문짝)을 로봇 시점에서 촬영한 약 100장으로 학습했습니다. 범용 결함 탐지 모델이 아닌, 시스템 실증을 위한 시편 특화 모델입니다.

---

## 폴더 구조

```
├── full_inspection.py      # 최종 통합 시스템
├── config/cam2ee.json      # hand-eye 캘리브레이션 결과
├── maps/                   # SLAM 지도
├── training/               # 데이터 변환, 학습 설정
├── calibration/            # hand-eye 캘리브레이션 코드
└── dev/                    # 단계별 개발·테스트 코드
    ├── auto_orbit.py       #   AGV 순회 단독
    ├── scan_rect.py        #   라이다 대상 인식 단독
    ├── goto_hole.py        #   팔 단일 결함 접근·촬영
    ├── multi_hole_test.py  #   팔 다중 결함 접근·촬영
    └── test_connect.py     #   FR5 연결 확인
```

## 한계 및 향후 연구

- 라이다가 유리면을 잘 인식하지 못해 대상 인식에 영향을 줄 수 있음
- 로봇팔 작업 반경 제약 (2단계 접근으로 일부 보완)
- 시편 특화 모델로, 실차 전체 검사를 위해서는 데이터 확장 필요
