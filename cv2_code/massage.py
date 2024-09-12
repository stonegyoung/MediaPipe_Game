import cv2
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
import numpy as np
from uuid import uuid4


hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)  # 웹캠 입력

width = 1280
height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

left_distance12, left_distance9, right_distance9, right_distance12 = 0,0,0,0
state = False
cnt = 0
save_idx = 0

while cap.isOpened():
    ret, img = cap.read()
    
    img = cv2.flip(img, 1)
    img_original = img.copy()
    
    if not ret:
        break

    # BGR 이미지를 RGB로 변환 후 손 추적
    result = hands.process( cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if result.multi_hand_landmarks:
        for hand_landmarks, hand_lr in zip(result.multi_hand_landmarks, result.multi_handedness) :
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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
            if left_distance12>left_distance9 and right_distance12>right_distance9 and state == True:
                cv2.putText(img, f'paper', (600,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 2)
                cnt +=1
                state = False
            
            
            # 9가 더 길면 접혀있는 상태
            if left_distance12<left_distance9 and right_distance12<right_distance9:
                cv2.putText(img, f'rock', (600,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 2)
                state = True
                
            
    cv2.putText(img, str(cnt), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 5)     
    cv2.imshow('', img)
  
    if cv2.waitKey(1) == ord('q'):
        print('저장되었습니다')
        cv2.imwrite(f'./server/massage_{str(uuid4())}.jpg', img_original)
        save_idx+=1
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
