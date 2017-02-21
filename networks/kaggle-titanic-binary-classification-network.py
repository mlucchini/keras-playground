import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential

np.random.seed(7)

root_path = '../data/kaggle/titanic'


def get_data(filepath):
    df = pd.read_csv(filepath)
    df = normalise_data(df)
    x = df.values[:, [2, 4, 5, 6, 7, 9, 10, 11]]
    y = df.values[:, 1]
    return x, y


def get_test_data(x_filepath, y_filepath):
    x_df = pd.read_csv(x_filepath)
    y_df = pd.read_csv(y_filepath)
    x_df = normalise_data(x_df)
    x = x_df.values[:, [1, 3, 4, 5, 6, 8, 9, 10]]
    y = y_df.values[:, 1]
    return x, y


def normalise_data(x):
    x['Sex'] = x['Sex'].apply(lambda x: 0 if x == 'male' else 1)
    x['Age'] = x['Age'].apply(lambda x: x / 100)
    x['Cabin'] = x['Cabin'].apply(lambda x: ord(x[0]) if isinstance(x, basestring) and len(x) > 0 else 0)
    x['Embarked'] = x['Embarked'].apply(lambda x: 0 if x == 'C' else 1 if x == 'Q' else 2)
    return x


def get_model(x):
    nb_inputs = x.shape[1]
    m = Sequential()
    m.add(Dense(nb_inputs * 10, input_dim=nb_inputs, init='uniform', activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(nb_inputs * 3, init='uniform', activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(1, init='uniform', activation='sigmoid'))
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(m.summary())
    return m

X, Y = get_data(root_path + '/train.csv')
X_test, Y_test = get_test_data(root_path + '/test.csv', root_path + '/gender_submission.csv')
model = get_model(X)

model.fit(X, Y, nb_epoch=100, batch_size=10, verbose=1)

scores = model.evaluate(X, Y, batch_size=10)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
