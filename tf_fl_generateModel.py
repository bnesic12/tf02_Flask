import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import load_model
import numpy as np
import sys
import os

print("Python version:")
print(sys.version) # 3.7.7 (default, Mar 26 2020, 10:32:53) [Clang 4.0.1 (tags/RELEASE_401/final)]
print(f"Tensorflow version: {tf.version}") # tensorflow_core._api.v2.version
print(f"Keras Version: {tf.keras.__version__}") # 2.2.4-tf
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE") # GPU is NOT AVAILABLE






# Get data. This is a school example of cloth items images:
# - 60,000 training images
# - 10,000 test images
# See https://github.com/zalandoresearch/fashion-mnist
mnist = keras.datasets.fashion_mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

class_names=['top','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']

#Normalize values:
X_train = X_train/255.0
X_test = X_test/255.0

###Build the model with TF2.0
model=Sequential()

#The following flattens 2-dim 28x28=784 into 1-dim 784 array of input neurons;
#to be used as input layer. BTW, 28x28 is num of pixels for each image
model.add(Flatten(input_shape=(28,28)))

#The next layer of neurons has 128 neurons with activation ReLu function
#This is hidden layer
model.add(Dense(128, activation='relu'))

#Output layer of 10 neurons with activation SoftMax function
model.add(Dense(10, activation = 'softmax'))

# Layer0--Layer1: 784x(128+1)=100352 connections
# Layer1--Layer2: (128+1)x10=1290 connections

### Compile model with following params:
### - Loss Function - how the model is accurate during training
### - optimizer
### - Metrics
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

### Train model
# Weights get updated 10 times
model.fit(X_train, y_train, epochs=10)

# Save trained model into a file:
model.save(os.path.join("./trainResults", "clothesModel01.h5"))
# Now, load the model back into another model instance
modelFromFile = load_model(os.path.join("./trainResults", "clothesModel01.h5"))

###Evaluate accuracy
test_loss, test_acc = modelFromFile.evaluate(X_test, y_test)
print(f"Accuracy = {test_acc}")  # Accuracy = 0.8873999714851379
#

### Prediction
from sklearn.metrics import accuracy_score
# Predict between discrete set of values ie categories (not continuous values)
# ie whether it is 'top','trouser','pullover',...
y_pred = modelFromFile.predict_classes(X_test)
print(f"prediction for first image is {y_pred[0]}") # prediction for first image is 9
print(f"accuracy of whole predition is {accuracy_score(y_test, y_pred)}") # accuracy of whole predition is 0.8874
# Predict continuous values, ie what is probability for each item is to be 'top','trouser','pullover',...
pred = modelFromFile.predict(X_test)
print(f"Item categories: {class_names}")
print(f"Predictions for 0-th element: {pred[0]}")
print(f"Max prediction for 0-th element is: {np.argmax(pred[0])}") # 9
# The following are prediction results for item [0]:
#   top:        5.4126527e-08
#   trouser:    4.7941390e-11
#    ...
#   ankle boot: 9.9725133e-01  <- obviously, this is the outcome of prediction, ie the image is 9 ie 'ankle boot'