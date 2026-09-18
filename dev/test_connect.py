from fairino import Robot

# FR5 연결
robot = Robot.RPC('192.168.58.2')

# 현재 관절 각도 읽기 (팔은 안 움직임, 읽기만)
ret = robot.GetActualJointPosDegree()
print("관절 각도:", ret)

# 현재 TCP(팔 끝) 위치 읽기
ret2 = robot.GetActualTCPPose()
print("TCP 위치:", ret2)
