import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO

def main():
    model = YOLO('/home/woohyung/dent_best.pt', task='segment') 

    pipeline = rs.pipeline()
    config = rs.config()
    
    # 카메라는 640x480 해상도로 부드럽게 가져옵니다.
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    print("📸 리얼센스 카메라 가동... (노트북 CPU 최적화 모드)")
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # 🌟 CPU 전용 최적화 세팅
            results = model(
                color_image, 
                conf=0.4,         # 신뢰도 컷을 살짝 낮춰서 렉 보완
                device='cpu',     # 무조건 CPU만 사용하도록 강제
                imgsz=480,        # ★핵심★ 모델 인식 해상도를 대폭 낮춤 (여전히 느리면 320으로 변경!)
                verbose=False     # 터미널 로그 끄기
            )

            annotated_frame = results[0].plot()

            cv2.imshow('RealSense + YOLO Dent Detection (CPU Mode)', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("🛑 테스트를 종료합니다.")

if __name__ == '__main__':
    main()
