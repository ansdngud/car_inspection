import os
import json
import shutil
import random
from glob import glob

def convert_labelme_to_yolo():
    print("🚀 LabelMe JSON -> YOLO TXT 변환을 시작합니다...")
    
    # 경로 설정
    raw_dir = "./dataset/raw"
    yolo_dir = "./dataset/yolo"
    
    # YOLO 형식의 폴더 구조 만들기
    for split in ['train', 'val']:
        os.makedirs(os.path.join(yolo_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, split, 'labels'), exist_ok=True)

    # JSON 파일 모두 찾기
    json_files = glob(os.path.join(raw_dir, "*.json"))
    if not json_files:
        print("❌ 에러: raw 폴더에 JSON 파일이 없습니다!")
        return

    # 학습용 80%, 검증용 20%로 랜덤하게 나누기
    random.shuffle(json_files)
    split_idx = int(len(json_files) * 0.8)
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]

    # 클래스 이름 (고기)
    class_name = "meat"  # LabelMe에서 라벨링하실 때 쓴 이름과 달라도 0번으로 통일됩니다.
    
    def process_files(files, split_name):
        count = 0
        for json_path in files:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 이미지 파일 찾기 (json 파일과 이름이 같은 이미지)
            img_name = data['imagePath']
            img_path = os.path.join(raw_dir, img_name)
            
            # 이미지가 없으면 패스 (확장자가 다를 수 있으니 이름으로 다시 검색)
            if not os.path.exists(img_path):
                base_name = os.path.splitext(os.path.basename(json_path))[0]
                img_candidates = glob(os.path.join(raw_dir, f"{base_name}.*"))
                img_candidates = [c for c in img_candidates if not c.endswith('.json')]
                if img_candidates:
                    img_path = img_candidates[0]
                    img_name = os.path.basename(img_path)
                else:
                    print(f"⚠️ 경고: {base_name}의 이미지 파일을 찾을 수 없습니다.")
                    continue

            # 1. 이미지 복사
            dest_img_path = os.path.join(yolo_dir, split_name, 'images', img_name)
            shutil.copy(img_path, dest_img_path)

            # 2. JSON에서 좌표 뽑아서 YOLO TXT로 변환
            img_w = data['imageWidth']
            img_h = data['imageHeight']
            
            txt_name = os.path.splitext(img_name)[0] + '.txt'
            dest_txt_path = os.path.join(yolo_dir, split_name, 'labels', txt_name)
            
            with open(dest_txt_path, 'w', encoding='utf-8') as f:
                for shape in data['shapes']:
                    # 네모 박스의 최소/최대 좌표 찾기
                    points = shape['points']
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    xmin, xmax = min(x_coords), max(x_coords)
                    ymin, ymax = min(y_coords), max(y_coords)
                    
                    # YOLO 포맷 (중심x, 중심y, 너비, 높이) - 0~1 사이로 정규화
                    center_x = ((xmin + xmax) / 2) / img_w
                    center_y = ((ymin + ymax) / 2) / img_h
                    width = (xmax - xmin) / img_w
                    height = (ymax - ymin) / img_h
                    
                    # 클래스는 무조건 0번 (고기 하나만 찾을 것이므로)
                    f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            count += 1
        return count

    # 실행
    train_count = process_files(train_files, 'train')
    val_count = process_files(val_files, 'val')
    
    print("=" * 50)
    print("✅ 변환 및 데이터셋 분할 완료!")
    print(f"📁 총 파일: {len(json_files)}개")
    print(f"💪 학습용(Train): {train_count}개")
    print(f"🧪 검증용(Val): {val_count}개")
    print("=" * 50)

if __name__ == "__main__":
    convert_labelme_to_yolo()