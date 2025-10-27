import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#đọc dữ liệu
warningheart1 = pd.read_csv("/content/dautim1.txt")
warningheart2 = pd.read_csv("/content/dautim2.txt")
warningheart3 = pd.read_csv("/content/dautim3.txt")  # Sửa tên biến
warningheart4 = pd.read_csv("/content/dautim4.txt")
warningheart = pd.concat([warningheart1, warningheart2, warningheart3, warningheart4])
normal1 = pd.read_csv("/content/normal1.csv")
normal2 = pd.read_csv("/content/normal2.csv")
normal3 = pd.read_csv("/content/normal3.csv")
normal4 = pd.read_csv("/content/normal4.csv")
normal5 = pd.read_csv("/content/normal5.csv")

data_normal = pd.concat([normal1, normal2, normal3, normal4, normal5])
X = []
y = []

#LSTM timestep để input trong both_side (=10 là = 10 điều kiện(label))
timesteps = 30




warningheart_values = warningheart.iloc[:,1:].values


scaler = StandardScaler()
datascaler_wh = scaler.fit_transform(warningheart_values)
datascaler_normal = scaler.fit_transform(data_normal)


n_sample_1 = len(datascaler_wh)

#lấy vòng lặp trong cứ 10 timesteps để huấn luyện
for i in range(timesteps, n_sample_1):
    X.append(datascaler_wh[i-timesteps:i, :])
    y.append(1)

#gán giá trị 0 khi bình thường
for i in range(timesteps, len(datascaler_normal)):
    X.append(datascaler_normal[i-timesteps:i, :])
    y.append(0)

X, y = np.array(X), np.array(y)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(Y_train),
    y=Y_train
)
class_weight_dict = dict(enumerate(class_weights))

model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=32, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=16))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation="sigmoid"))
model.compile(optimizer="adam", metrics=['accuracy'], loss="binary_crossentropy")
model.summary()

#callback
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

mc = ModelCheckpoint(
    'best_model_mediapipe.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

model.fit(
    X_train, Y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, Y_test),
    callbacks=[es, mc, rlr],
    class_weight=class_weight_dict,
    verbose=1
)

model.save("model.h5")
