import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

cap = cv2.VideoCapture(0)
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
   static_image_mode=False,
    model_complexity=2,            # 1 hoặc 2 = chính xác hơn
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.3,  # giảm để dễ nhận tay hơn
    min_tracking_confidence=0.6
)

lm_list=[]
label = "dautim4"

have_frame = 400
def make_landmark_pose(results):
    """Trả về list [x, y, z, visibility] cho 33 pose landmarks"""
    if not results.pose_landmarks:
        return None
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
        
    return c_lm
def make_landmark_hand_left(result):
    """Trả về list [x, y, z] cho 21 left hand landmarks"""
    if not result.left_hand_landmarks:
        return None
    c_lm_hl = []
    for lm_hl in result.left_hand_landmarks.landmark:              
        c_lm_hl.extend([lm_hl.x, lm_hl.y, lm_hl.z])
        
    return c_lm_hl

def make_landmark_hand_right(results):
    """Trả về list [x, y, z] cho 21 right hand landmarks"""
    if not results.right_hand_landmarks:
        return None
    c_lm_hr = []
    for lm_hr in results.right_hand_landmarks.landmark:
        c_lm_hr.extend([lm_hr.x, lm_hr.y, lm_hr.z])
        
    return c_lm_hr

def draw_all_landmarks(mp_drawing, results, img):
    """Vẽ tất cả landmarks lên frame"""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1)
        )
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
        )
    
    return img
    
while len(lm_list)<have_frame:
    
    ret, frame = cap.read()
    if not ret:
        break
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(frameRGB)
    frame_count = 0
    frame = draw_all_landmarks(mp_draw, result, frame)
    
    
    
    pose_lm = make_landmark_pose(result)
    left_hand_lm = make_landmark_hand_left(result)
    right_hand_lm = make_landmark_hand_right(result)
        
    if pose_lm:
        row = []
        row.extend(pose_lm)
        row.extend(left_hand_lm if left_hand_lm else [0.0] * 63)
        row.extend(right_hand_lm if right_hand_lm else [0.0] * 63)
        
            
        lm_list.append(row)
        frame_count += 1
    
    cv2.imshow("warningheart",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
df = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")
cap.release()
cv2.destroyAllWindows()