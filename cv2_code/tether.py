import cv2
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
import mediapipe as mp
from uuid import uuid4

pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

width = 1280
height = 720
check_height = height//2
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


left_state, right_state = False, False
last_hand = ''
cnt = 0
while cap.isOpened():
    ret, img = cap.read()
    
    if not ret:
        break
    
    img = cv2.flip(img, 1) # 좌우반전
    img_original = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 결과 33개 좌표 값
    result = pose.process(img_rgb) # rgb로 넣어줘야 한다 

    cv2.circle(img, (640,200), 10, (0,0,255), -1)
    
    # 인식이 됐으면
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            img,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing_styles.get_default_pose_landmarks_style()
        )
        left_y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y * height
        right_y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y * height 
        if  left_y < 300:
            left_state = True
        if right_y < 300:
            right_state = True
            
        # 화면 절반까지 내려오면
        if  left_y >= check_height and left_state == True:
            if last_hand != 'left': # 같은 손 계속 들지 않게
                last_hand = 'left'
                print("right")
                cnt += 1
            left_state = False
        elif right_y >= check_height and right_state == True:
            right_state = False
            if last_hand != 'right':
                last_hand = 'right'
                print("left")
                cnt += 1
                        
    cv2.putText(img, str(cnt), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 5)   
    cv2.imshow('tether', img)
        
    if cv2.waitKey(1)==27:
        break
    if cv2.waitKey(1) == ord('q'):
        cv2.imwrite(f'./server/tether_{str(uuid4())}.jpg', img_original)
        print("저장되었습니다")
cv2.destroyAllWindows()