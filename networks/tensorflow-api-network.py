from keras import backend
from keras.layers import Dense
from keras.layers import Dropout
from keras.metrics import categorical_crossentropy as accuracy
from keras.objectives import categorical_crossentropy
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

session = tf.Session()
backend.set_session(session)

img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

X = Dense(128, activation='relu')(img)
X = Dropout(0.5)(X)
X = Dense(128, activation='relu')(X)
X = Dropout(0.5)(X)
preds = Dense(10, activation='softmax')(X)

loss = tf.reduce_mean(categorical_crossentropy(labels, preds))
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
with session.as_default():
    tf.global_variables_initializer().run()
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1],
                                  backend.learning_phase(): 1})

acc_value = accuracy(labels, preds)
with session.as_default():
    print acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels,
                                    backend.learning_phase(): 0})

print backend.learning_phase()
