import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import os

def extract_landmarks(video_path, output_csv = "normal.csv"):
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

#nap video
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    lm_list = []
    label = os.path.basename(video_path).split('_')[0]  # Lấy nhãn từ tên file video

    with mp_holistic.Holistic(
        static_image_mode = False,
        model_complexity = 2,
        enable_segmentation = False,
        refine_face_landmarks = True,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            #chuyen BGR sang RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #Lấy landmarks
            pose_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else [0]
            left_hand_landmarks = results.left_hand_landmarks.landmark if results.left_hand_landmarks else [0]
            right_hand_landmarks = results.right_hand_landmarks.landmark if results.right_hand_landmarks else [0]

            frame_lm = []
            for lm in pose_landmarks:
                frame_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
            for lm in left_hand_landmarks:
                frame_lm.extend([lm.x, lm.y, lm.z])
            for lm in right_hand_landmarks:
                frame_lm.extend([lm.x, lm.y, lm.z])

            if frame_lm:
                frame_lm.append(label)
                lm_list.append(frame_lm)

            frame_idx += 1
    cap.release()
    df = pd.DataFrame(lm_list)
    df.to_csv(output_csv, index=False)
    print(f"Landmarks extracted and saved to {output_csv}")
