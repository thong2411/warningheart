import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
#đọc dữ liệu
warningheart = pd.read_csv("cambientim/dautim1.txt")
dataset_normal = pd.read_csv("cambientim/normal.txt")
X = []
y = []

scaler = StandardScaler()
datascaler_wh = scaler.fit_transform(warningheart)
datascaler_normal = scaler.transform(dataset_normal)
#LSTM timestep để input trong both_side (=10 là = 10 điều kiện(label))
timesteps = 30

dataset_warningheart = datascaler_wh.iloc[:,1:].values
n_sample_1 = len(dataset_warningheart)
#lấy vòng lặp trong cứ 10 timesteps để huấn luyện 
for i in range(timesteps,n_sample_1):
    X.append(dataset_warningheart[i-timesteps:i,:])
    y.append(1)
#gán giá trị 0 khi bình thường
for i in range(timesteps, len(dataset_normal)):
    X.append(dataset_normal[i-timesteps:i, :])
    y.append(0)

    
X, y = np.array(X), np.array(y)

    
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2,stratify=y,random_state=42)
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(Y_train),
    y=Y_train
)
class_weight_dict = dict(enumerate(class_weights))


model  = Sequential()
model.add(LSTM(units = 64, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(units = 64, return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(units = 32, return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(units = 16))
model.add(Dropout(0.3))
model.add(Dense(units = 1, activation="sigmoid"))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "binary_crossentropy")
model.summary()
#call back
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True,verbose=1)

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