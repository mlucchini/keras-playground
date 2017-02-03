from keras.applications.vgg16 import VGG16
from quiver_engine.server import launch

model = VGG16()
launch(model, '/tmp/quiver', './data/mnist-test')
