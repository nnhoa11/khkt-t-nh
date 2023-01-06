import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import TensorBoard

X = []
Y = []
NAME = "inmoov-hand-detect-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

no_timestep = 2


n_sample = 1000


for i in range(0,4):
    data = pd.read_csv("pose" + str(i + 1) + ".txt")
    dataset = data.iloc[:,1:].values
    for index in range(no_timestep, n_sample):
        X.append(dataset[index - no_timestep:index, :])
        Y.append(i)

X, Y = np.array(X), np.array(Y)

Y = to_categorical(Y).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print(X.shape)

model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 4, activation="softmax"))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "binary_crossentropy")

model.fit(X_train, y_train, epochs=30,batch_size=32, validation_data=(X_test, y_test), callbacks=[tensorboard])
model.save("model4-right.h5")
