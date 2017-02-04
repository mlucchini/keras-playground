import cv2

haar_cascade_config_file_name = '/usr/local/Cellar/opencv/' + cv2.__version__ + '/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
face_classifier = cv2.CascadeClassifier(haar_cascade_config_file_name)
video_capture = cv2.VideoCapture(0)


def get_face_name(_):
    return 'Me?'

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),  (0, 255, 0), 2)
        face = frame[y: y + h, x: x + w]
        name = get_face_name(face)
        cv2.putText(frame, name, (x + 5, y - 5), cv2.cv.CV_FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == (ord('q')):
        break

video_capture.release()
cv2.destroyAllWindows()
