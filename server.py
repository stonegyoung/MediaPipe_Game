from fastapi import FastAPI, UploadFile, Request
import uvicorn
import os 

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import pose as mp_pose
import make_character

import logging

# 로거 설정
logging.basicConfig(
    filename='access_log.txt',  # 로그 파일 이름
    level=logging.INFO,         # 로그 수준 설정
    format='%(asctime)s - %(message)s',  # 로그 출력 형식
    datefmt='%Y-%m-%d %H:%M:%S' # 시간 형식
)


# 손
hands = mp_hands.Hands()
pose = mp_pose.Pose()


app = FastAPI() # fastapi 객체 만들기

width = 1280
height = 720
check_height = height//2
    
sun=make_character.SunState()    
moon=make_character.MoonState()    


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"An error occurred: {str(exc)}")
    return {"message": "An internal server error occurred"}  # 기본 응답 설정


@app.get("/") 
def root(): 
    logging.info('/')
    return {'message': '접속 완료'}


@app.get("/reset") 
def reset(): 
    moon.reset()
    sun.reset()
    logging.info('/reset')
    return {'message': f'게임 값 초기화'}


# 이미지가 들어온다
@app.post("/tiger_hand1") 
async def tiger_hand1(file:UploadFile): 
    file_path = os.path.join('./server', f'tiger_hand1_{file.filename}')
    with open(file_path, 'wb') as buffer: # 이미지는 바이너리파일
        buffer.write(await file.read()) # 파일 저장
        
    img = cv2.imread(file_path)
    img = cv2.flip(img, 1)
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_lr in zip(results.multi_hand_landmarks, results.multi_handedness) : # hand_lr: 왼쪽 오른쪽 구별
            
            # 9번 좌표의 z 값이 가까워지면(더 큰 음수일수록 카메라에 더 가까이 위치)
            if hand_lr.classification[0].label == 'Left':
                hand_z = np.abs(np.round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z, 5))
                
                # 가까워지면
                if hand_z > 0.05:
                    sun.th_left_state = True
                else:
                    # 가까워졌다가 멀어 지면
                    if sun.th_left_state == True:
                        sun.th_left_state = False
                        print('left')
                        logging.info(f'tiger_hand1: left')
                        return {'result': 'LEFT'}
                    
            # 9번 좌표의 z 값이 가까워지면
            if hand_lr.classification[0].label == 'Right':
                hand_z = np.abs(np.round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z, 5))
                if hand_z > 0.05:
                    sun.th_right_state = True
                else:
                    if sun.th_right_state == True:
                        sun.th_right_state = False
                        print('tiger_hand1: right')
                        logging.info(f'right')
                        return {'result': 'RIGHT'}
    
    logging.info(f'tiger_hand1: None')
    # 빈 문자열
    return {'result': ''}

@app.post("/massage1") 
async def massage1(file:UploadFile): 
    left_distance12, left_distance9, right_distance9, right_distance12 = 0,0,0,0
    file_path = os.path.join('./server', f'massage1_{file.filename}')
    with open(file_path, 'wb') as buffer: # 이미지는 바이너리파일
        buffer.write(await file.read()) # 파일 저장
        
    img = cv2.imread(file_path)
    result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if result.multi_hand_landmarks:
        for hand_landmarks, hand_lr in zip(result.multi_hand_landmarks, result.multi_handedness) :
            wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
            point_12 = np.array([hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y, hand_landmarks.landmark[12].z])
            point_9 = np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[9].z])

        
            if hand_lr.classification[0].label == 'Left':
                # 12번에서 wrist까지, 9번에서 wrist까지 길이
                left_distance12 = np.linalg.norm(point_12-wrist)
                left_distance9 = np.linalg.norm(point_9-wrist)
            elif hand_lr.classification[0].label == 'Right':
                # 12번에서 wrist까지, 9번에서 wrist까지 길이
                right_distance12 = np.linalg.norm(point_12-wrist)
                right_distance9 = np.linalg.norm(point_9-wrist)
                
            
            # 두 손 모두 12가 더 길면 펴져있는 상태
            if left_distance12>left_distance9 and right_distance12>right_distance9 and sun.ms_state == True:
                sun.ms_cnt +=1
                sun.ms_state = False
            
            
            # 9가 더 길면 접혀있는 상태
            if left_distance12<left_distance9 and right_distance12<right_distance9:
                sun.ms_state = True
    print(sun.ms_cnt)
    logging.info(f'massage1: {sun.ms_cnt}')         
    return {'result': sun.ms_cnt} # cnt 값 전달

@app.post("/tether1") 
async def tether1(file:UploadFile): 
    file_path = os.path.join('./server', f'tether1_{file.filename}')
    with open(file_path, 'wb') as buffer: # 이미지는 바이너리파일
        buffer.write(await file.read()) # 파일 저장
        
    img = cv2.imread(file_path)
    
    result = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # rgb로 넣어줘야 한다 
   
    # 인식이 됐으면
    if result.pose_landmarks:
        left_y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y * height
        right_y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y * height 
        if  left_y < 300:
            sun.tt_left_state = True
        if right_y < 300:
            sun.tt_right_state = True
            
        # 화면 절반까지 내려오면
        if  left_y >= check_height and sun.tt_left_state == True:
            if sun.tt_last_hand != 'left': # 같은 손 계속 들지 않게
                sun.tt_last_hand = 'left'
                print("right")
                sun.tt_cnt += 1
            sun.tt_left_state = False
        elif right_y >= check_height and sun.tt_right_state == True:
            sun.tt_right_state = False
            if sun.tt_last_hand != 'right':
                sun.tt_last_hand = 'right'
                print("left")
                sun.tt_cnt += 1
    
    print(sun.tt_cnt)
    logging.info(f'tether1: {sun.tt_cnt}')
    return {'result': sun.tt_cnt} # cnt 값 전달



# 이미지가 들어온다
@app.post("/tiger_hand2") 
async def tiger_hand2(file:UploadFile): 
    file_path = os.path.join('./server', f'tiger_hand2_{file.filename}')
    with open(file_path, 'wb') as buffer: # 이미지는 바이너리파일
        buffer.write(await file.read()) # 파일 저장
        
    img = cv2.imread(file_path)
    img = cv2.flip(img, 1)
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_lr in zip(results.multi_hand_landmarks, results.multi_handedness) : # hand_lr: 왼쪽 오른쪽 구별
            
            # 9번 좌표의 z 값이 가까워지면(더 큰 음수일수록 카메라에 더 가까이 위치)
            if hand_lr.classification[0].label == 'Left':
                hand_z = np.abs(np.round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z, 5))
                
                # 가까워지면
                if hand_z > 0.05:
                    moon.th_left_state = True
                else:
                    # 가까워졌다가 멀어 지면
                    if moon.th_left_state == True:
                        moon.th_left_state = False
                        print('left')
                        logging.info(f'tiger_hand2: left')
                        return {'result': 'LEFT'}
                    
            # 9번 좌표의 z 값이 가까워지면
            if hand_lr.classification[0].label == 'Right':
                hand_z = np.abs(np.round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z, 5))
                if hand_z > 0.05:
                    moon.th_right_state = True
                else:
                    if moon.th_right_state == True:
                        moon.th_right_state = False
                        print('right')
                        logging.info(f'tiger_hand2: right')
                        return {'result': 'RIGHT'}
                    
    logging.info(f'tiger_hand2: None')
    return {'result': ''} # 빈 문자열, LEFT, RIGHT 리턴

@app.post("/massage2") 
async def massage2(file:UploadFile): 
    left_distance12, left_distance9, right_distance9, right_distance12 = 0,0,0,0
    file_path = os.path.join('./server', f'massage2_{file.filename}')
    with open(file_path, 'wb') as buffer: # 이미지는 바이너리파일
        buffer.write(await file.read()) # 파일 저장
        
    img = cv2.imread(file_path)
    result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if result.multi_hand_landmarks:
        for hand_landmarks, hand_lr in zip(result.multi_hand_landmarks, result.multi_handedness) :
            wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
            point_12 = np.array([hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y, hand_landmarks.landmark[12].z])
            point_9 = np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[9].z])

        
            if hand_lr.classification[0].label == 'Left':
                # 12번에서 wrist까지, 9번에서 wrist까지 길이
                left_distance12 = np.linalg.norm(point_12-wrist)
                left_distance9 = np.linalg.norm(point_9-wrist)
            elif hand_lr.classification[0].label == 'Right':
                # 12번에서 wrist까지, 9번에서 wrist까지 길이
                right_distance12 = np.linalg.norm(point_12-wrist)
                right_distance9 = np.linalg.norm(point_9-wrist)
                
            
            # 두 손 모두 12가 더 길면 펴져있는 상태
            if left_distance12>left_distance9 and right_distance12>right_distance9 and moon.ms_state == True:
                moon.ms_cnt +=1
                moon.ms_state = False
            
            
            # 9가 더 길면 접혀있는 상태
            if left_distance12<left_distance9 and right_distance12<right_distance9:
                moon.ms_state = True
    print(moon.ms_cnt)
    logging.info(f'massage2: {moon.ms_cnt}')                 
    return {'result': moon.ms_cnt} # cnt 값 전달

@app.post("/tether2") 
async def tether2(file:UploadFile): 
    file_path = os.path.join('./server', f'tether2_{file.filename}')
    with open(file_path, 'wb') as buffer: # 이미지는 바이너리파일
        buffer.write(await file.read()) # 파일 저장
        
    img = cv2.imread(file_path)
    
    result = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # rgb로 넣어줘야 한다 
    if result.pose_landmarks:
        left_y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y * height
        right_y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y * height 
        if  left_y < 300:
            moon.tt_left_state = True
        if right_y < 300:
            moon.tt_right_state = True
            
        # 화면 절반까지 내려오면
        check_height
        if  left_y >= check_height and moon.tt_left_state == True:
            if moon.tt_last_hand != 'left': # 같은 손 계속 들지 않게
                moon.tt_last_hand = 'left'
                print("right")
                moon.tt_cnt += 1
            moon.tt_left_state = False
        elif right_y >= check_height and moon.tt_right_state == True:
            moon.tt_right_state = False
            if moon.tt_last_hand != 'right':
                moon.tt_last_hand = 'right'
                print("left")
                moon.tt_cnt += 1
                
    print(moon.tt_cnt)
    logging.info(f'tether2: {moon.tt_cnt}')
    return {'result': moon.tt_cnt} # cnt 값 전달


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8788)