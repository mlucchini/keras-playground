from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import numpy

K.set_image_dim_ordering('th')
numpy.random.seed(7)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print("Shape of images before transformation: %s" % (X_train[0].shape,))

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

print("Shape of images after transformation: %s" % (X_train[0].shape,))

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

def baseline_model():
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

tensor_board = TensorBoard(log_dir='./logs/mnist-cnn', histogram_freq=0, write_graph=True)
model = baseline_model()
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=20, batch_size=200, verbose=2, callbacks=[tensor_board])
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline error: %.2f%%" % (100 - scores[1] * 100))
