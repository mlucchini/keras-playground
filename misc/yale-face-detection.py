import os
import re

import cv2
from skimage import io


def show_images(images):
    for image in images:
        io.imshow(image)
        io.show()

haar_cascade_config_file_name = '/usr/local/Cellar/opencv/' + cv2.__version__ + '/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
faces_directory = 'data/yalefaces'

images_file_paths = [os.path.join(root, f) for root, _, files in os.walk(faces_directory) for f in files if f.startswith('subject') ]
images_data = [io.imread(path, as_grey=True) for path in images_file_paths]
images_labels = [re.sub('.*subject([0-9]+).*', '\g<1>', path) for path in images_file_paths]

face_detect_classifier = cv2.CascadeClassifier(haar_cascade_config_file_name)
faces_coordinates = [face_detect_classifier.detectMultiScale(image_data)[0] for image_data in images_data]
faces_data = [images_data[idx][y: y + h, x:x + w] for idx, (x, y, w, h) in enumerate(faces_coordinates)]

show_images(faces_data)
