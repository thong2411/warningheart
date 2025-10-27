import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# --- Cấu hình ---
VIDEO_SOURCE = 0  # 0 = webcam, hoặc thay bằng đường dẫn video "video_test.mp4"
TIMESTEPS = 30

# --- Load model ---
model = load_model("model.h5")

# --- Khởi tạo MediaPipe Holistic ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- Bộ đệm lưu 30 frame landmark ---
sequence = []

def extract_landmarks(results):
    """Lấy toàn bộ tọa độ tay trái + tay phải (x, y, z)"""
    
    row = []
    # Pose
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        row.extend([0.0]*132)  # 33 điểm × 3

    # Tay trái
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])
    else:
        row.extend([0.0]*63)

    # Tay phải
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])
    else:
        row.extend([0.0]*63)
    return row

# --- Mở video ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        # Vẽ keypoint lên màn hình
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        # --- Lấy landmarks ---
        row = extract_landmarks(results)
        sequence.append(row)

        # Giữ lại 30 frame gần nhất
        if len(sequence) > TIMESTEPS:
            sequence.pop(0)

        # Khi có đủ 30 frame thì dự đoán
        if len(sequence) == TIMESTEPS:
            X_input = np.expand_dims(sequence, axis=0)  # (1, 30, n_features)
            y_pred = model.predict(X_input, verbose=0)[0][0]

            # Dự đoán
            label = "Đau tim" if y_pred > 0.5 else "Bình thường"
            color = (0, 0, 255) if label == "Đau tim" else (0, 255, 0)
            cv2.putText(frame, f"{label} ({y_pred:.2f})", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

        cv2.imshow("Heart Action Detection", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()