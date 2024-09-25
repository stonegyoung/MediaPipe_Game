import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions import hands as mp_hands
from uuid import uuid4


left_state, right_state = False, False

hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

width = 1280
height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

while cap.isOpened():
    ret, img = cap.read()
    
    img = cv2.flip(img, 1) # 좌우 반전
    img_original = img.copy()
    
    if not ret:
        break
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # mediapipe에서 읽기 위해 rgb로 바꾼다
    
    results = hands.process(img_rgb) # 미디어 파이프에 집어 넣어서 손가락 좌표 받기
    
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_lr in zip(results.multi_hand_landmarks, results.multi_handedness) : # hand_lr: 왼쪽 오른쪽 구별
            
            # 0번 좌표(손 맨 아래)에 왼쪽인지 오른쪽인지 써준다
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST] # []안에 좌표를 가져오거나 이름을 가져오거나
            
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS) # background img에 그리기. HAND_CONNECTIONS: 손가락 좌표 각각을 연결해라
            
            if hand_lr.classification[0].label == 'Left':
                cv2.putText(img, 'Left', (int(wrist.x * width), int(wrist.y * height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2) # 글자 집어넣기(좌표, 폰트, 폰트 스케일, 색, 두께)  
            elif hand_lr.classification[0].label == 'Right':
                cv2.putText(img, 'Right', (int(wrist.x * width), int(wrist.y * height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2) # 글자 집어넣기(좌표, 폰트, 폰트 스케일, 색, 두께) 
    
            # 9번 좌표의 z 값이 가까워지면(더 큰 음수일수록 카메라에 더 가까이 위치)
            if hand_lr.classification[0].label == 'Left':
                hand_z = np.abs(np.round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z, 5)*1000)
                
                cv2.putText(img, str(hand_z), (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                
                if hand_z > 100:
                    left_state = True
                elif hand_z < 50:
                    if left_state == True:
                        cv2.putText(img, 'LEFT', (600,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 5)
                        left_state = False
                    
            # 9번 좌표의 z 값이 가까워지면
            if hand_lr.classification[0].label == 'Right':
                hand_z = np.abs(np.round(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z, 5)*1000)
                cv2.putText(img, str(hand_z), (1080,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                if hand_z > 100:
                    right_state = True
                elif hand_z < 50:
                    if right_state == True:
                        cv2.putText(img, 'RIGHT', (600,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 5)
                        right_state = False
    cv2.imshow('tiger hand', img)
    
    if cv2.waitKey(1) == ord('q'):
        print("저장되었습니다")
        cv2.imwrite(f'./server/tiger_hand_{str(uuid4())}.jpg', img_original)
  
    if cv2.waitKey(1) == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
    
    
