import cv2

# 1. 보드 파라미터 설정 (A2 용지 & 1.4m 거리 최적화)
squares_x = 6       # 가로 칸 수
squares_y = 8       # 세로 칸 수
square_length = 60  # 체스판 사각형 한 변의 길이 (mm 단위로 나중에 인쇄 시 매칭)
marker_length = 45  # 아루코 마커 한 변의 길이 (mm)

# 아루코 딕셔너리 설정 (가장 많이 쓰이는 5x5 사용)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

# 2. 차루코 보드 객체 생성 (OpenCV 버전에 따른 호환성 처리)
try:
    # 최신 OpenCV (4.7 이상)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
except AttributeError:
    # 구버전 OpenCV
    board = cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, dictionary)

# 3. 고해상도 이미지 생성 설정
# 1mm를 약 10픽셀로 매핑하여 고해상도(선명한 테두리) 유지
# 사각형 60mm -> 600픽셀. 가로(6칸)=3600px, 세로(8칸)=4800px
image_width = squares_x * 600
image_height = squares_y * 600

# 여백 설정 (인쇄 시 잘림 방지용, 픽셀 단위)
margin_px = 300 

# 보드 이미지 그리기
try:
    # 최신 OpenCV
    board_image = board.generateImage((image_width + margin_px*2, image_height + margin_px*2), marginSize=margin_px)
except AttributeError:
    # 구버전 OpenCV
    board_image = board.draw((image_width + margin_px*2, image_height + margin_px*2), marginSize=margin_px)

# 4. 이미지 파일로 저장
file_name = 'charuco_board_a2_6x8_60mm.png'
cv2.imwrite(file_name, board_image)
print(f"✅ {file_name} 파일이 성공적으로 생성되었습니다!")