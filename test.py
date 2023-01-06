import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

model = tf.keras.models.load_model("model4.h5")


X = []
Y = []

no_timestep = 2

n_sample = 100
for i in range(0,4):
    data = pd.read_csv("pose" + str(i + 1) + "-test.txt")
    dataset = data.iloc[:,1:].values
    for index in range(no_timestep, n_sample):
        X.append(dataset[index - no_timestep:index, :])
        Y.append(i)


X, Y = np.array(X), np.array(Y)

Y = to_categorical(Y).astype(int)
print(X.shape, Y.shape)
print(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1)

score = model.evaluate(X, Y, verbose=0)
print('Test loss: %.4f'% score[0])
print('Test accuracy %.4f'% score[1])