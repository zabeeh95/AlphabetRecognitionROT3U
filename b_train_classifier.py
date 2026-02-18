import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dropout, Flatten, Conv2D, MaxPool2D, Dense
from tensorflow.keras.optimizers import Adam

data = pd.read_csv("data/A_Z Handwritten Data.csv").astype('float32')

# total samples are 372,450 and total of 14,500 samples per word
# column 1- contains 784 elements in 28x28 shape
X = data.drop('0', axis=1)

# column 0 contains labels of alphabets
y = data['0'].astype('int32')
print(y.unique())
# Split data into training and testing sets
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.20, random_state=75)

# Reshape data for the CNN model   -- shape 28x28x1
X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_train = X_train / 255.0
X_test = X_test / 255.0

""" Sequential API TF"""
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (5, 5), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(26, activation='softmax'),
])
model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=256, verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Evaluation error CNN Error: %.2f%%" % (100 - scores[1] * 100))

model.save('data/output/model.keras')

# Print accuracy and loss
print("The validation accuracy is:", history.history['val_accuracy'][-1])
print("The training accuracy is:", history.history['accuracy'][-1])
print("The validation loss is:", history.history['val_loss'][-1])
print("The training loss is:", history.history['loss'][-1])

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
