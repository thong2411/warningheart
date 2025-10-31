import cv2
import numpy as np
import tensorflow as tf
import os

#doc video và chuyen thanh tensor
def load_video(path, max_frames=0, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(frame)
            if len(frames) >= max_frames:
                break
    finally:
        cap.release()
    video_tensor = tf.convert_to_tensor(np.array(frames), dtype=tf.float32)
    return video_tensor

#Nap video va gan nhan
def load_dataset(video_dir):
    videos = []
    labels = []

    for file in os.listdir(video_dir):
        if file.endwith('.mp4'):
            label = 0 if 'normal' in file else 1 #0 = normal, 1 = unnormal
            path = os.path.join(video_dir, file)
            vid = load_video(path)
            videos.append(vid)
            labels.append(label)
            print(f"Loaded: {file} -> label = {label}")
    
    return np.array(videos), np.array(labels)

#Tao mo hinh CNN + LSTM
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            input_shape=input_shape
        ),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

#chay train model
if __name__ == "__main__":
    data_dir = "data"  # thu muc chua video
    X, y = load_dataset(data_dir)

    print("Shape X:", X.shape)  # in thong tin ve kich thuoc video tensor
    print("Shape y:", y.shape)

    model = create_model((X.shape[1], X.shape[2], X.shape[3], X.shape[4]))

    # train
    model.fit(X, y, epochs=10, batch_size=2)

    # luu model
    model.save("video_model.h5")