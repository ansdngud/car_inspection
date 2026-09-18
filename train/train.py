"""
덴트/스크래치 세그멘테이션 모델 학습 스크립트

- 사전학습 모델: yolo26n-seg.pt (YOLO26 nano segmentation)
- 클래스: dent(0), scratch(1)

사용법:
    1. json2yolo.py로 labelme 라벨을 YOLO 세그멘테이션 형식으로 변환
    2. data.yaml의 path를 데이터셋 실제 경로로 수정
    3. python train.py
"""
import os
from ultralytics import YOLO

# data.yaml은 이 스크립트와 같은 폴더(train/)에 있음
DATA_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.yaml')


def main():
    # 세그멘테이션 사전학습 모델 로드 (처음 실행 시 자동 다운로드)
    model = YOLO('yolo26n-seg.pt')

    model.train(
        data=DATA_YAML,
        epochs=150,        # 최대 에폭 (조기 종료로 더 일찍 끝날 수 있음)
        imgsz=640,         # 학습 해상도
        batch=8,           # GPU 메모리 부족하면 4로 줄이기
        device=0,          # GPU 0번 사용 (GPU 없으면 'cpu')
        patience=50,       # 50 에폭 동안 개선 없으면 조기 종료
        project='dent_train',
        name='exp1',
    )

    # 검증 성능 출력 (클래스별 mAP 포함)
    model.val()
    print("=== 학습 완료 ===")
    print("결과 위치: dent_train/exp1/weights/best.pt")


if __name__ == '__main__':
    main()