import sys

import numpy as np
from keras.datasets import reuters
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

from util.spark_util import TrainerConf, LocalTrainer, DistributedTrainer

epochs = 20
batch_size = 32
train_test_split = 0.2
max_words = 1000
metrics = ['accuracy']
optimizer = 'adagrad'


def main(use_spark):

    def generate_data():
        (x_train, y_train), (x_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2)
        nb_classes = np.max(y_train) + 1
        tokenizer = Tokenizer(nb_words=max_words)
        x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
        x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)
        return x_train, y_train, x_test, y_test, nb_classes

    def generate_model(nb_classes):
        m = Sequential()
        m.add(Dense(128, input_shape=(max_words,)))
        m.add(Activation('relu'))
        m.add(Dropout(0.2))
        m.add(Dense(128))
        m.add(Activation('relu'))
        m.add(Dropout(0.2))
        m.add(Dense(nb_classes))
        m.add(Activation('softmax'))
        m.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
        return m

    def train_model(model, x_train, y_train):
        conf = TrainerConf(model, epochs, batch_size, train_test_split, optimizer, metrics, 1, 2)
        trainer = DistributedTrainer(conf) if use_spark else LocalTrainer(conf)
        trainer.fit(x_train, y_train)

    X_train, Y_train, X_test, Y_test, nb_classes = generate_data()
    model = generate_model(nb_classes)
    train_model(model, X_train, Y_train)
    score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)

    print()
    print('Test score: %1.4f' % score[0])
    print('Test accuracy: %1.4f' % score[1])

if __name__ == "__main__":
    use_spark = len(sys.argv) > 1 and sys.argv[1] == 'spark'
    main(use_spark)
