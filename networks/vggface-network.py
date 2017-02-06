import copy
from scipy import misc
import os
import numpy as np
from util.vggface import VGGFace


def predict(model, file_path):
    image = misc.imread(file_path)
    image = misc.imresize(image, (224, 224)).astype(np.float32)
    aux = copy.copy(image)
    image[:, :, 0] = aux[:, :, 2]
    image[:, :, 2] = aux[:, :, 0]
    image[:, :, 0] -= 93.5940
    image[:, :, 1] -= 104.7624
    image[:, :, 2] -= 129.1863
    image = np.expand_dims(image, axis=0)

    result = model.predict(image)
    idx = np.argmax(result[0])
    print("%s: %d, acc: %1.4f" % (file_path, idx, result[0][idx]))

vggface = VGGFace()
print(vggface.summary())

images_file_paths = [os.path.join(root, f) for root, _, files in os.walk('data') for f in files if f.startswith('vggface-')]
for image_path in images_file_paths:
    predict(vggface, image_path)
