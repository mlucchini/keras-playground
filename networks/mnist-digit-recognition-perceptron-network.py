from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy

numpy.random.seed(7)

def plot_image(plotArgs, x, index):
    plt.subplot(plotArgs)
    plt.imshow(x[index], cmap=plt.get_cmap('gray'))

def plot_first_images(x):
    plot_image(221, x, 0)
    plot_image(222, x, 1)
    plot_image(223, x, 2)
    plot_image(224, x, 3)
    plt.show()

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# plot_first_images(X_train)

print("Shape of images before transformation: %s" % (X_train[0].shape,))

num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

print("Shape of images after transformation: %s" % (X_train[0].shape,))

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))
    model.add(Dense(num_classes, init='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

tensor_board = TensorBoard(log_dir='./logs/mnist-perceptron', histogram_freq=0, write_graph=True)
model = baseline_model()
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=20, batch_size=200, verbose=2, callbacks=[tensor_board])
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline error: %.2f%%" % (100 - scores[1] * 100))
