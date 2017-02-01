from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from util.plot_util import CifarPlot
import util.signal_util
import numpy as np

K.set_image_dim_ordering('th')
np.random.seed(7)
epochs = 25
lrate = 0.01

def plot_images(x):
    plot = CifarPlot(x)
    plot.show()

def normalise_data(x):
    x = x.astype('float32')
    x /= 255.0
    return x

def generate_model(num_classes):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lrate, momentum=0.9, decay=lrate/epochs, nesterov=False), metrics=['accuracy'])
    print(model.summary())
    return model

def train_model(model, x, y, x_test, y_test):
    model.fit(x, y, validation_data=(x_test, y_test), nb_epoch=epochs, batch_size=32)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % scores[1] * 100)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# plot_images(X_train)

X_train = normalise_data(X_train)
X_test = normalise_data(X_test)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
model = generate_model(y_test.shape[1])
train_model(model, X_train, y_train, X_test, y_test)
