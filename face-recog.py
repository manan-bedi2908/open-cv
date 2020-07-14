import urllib.request
import cv2
import numpy as np

url = "http://192.168.0.100:8080/shot.jpg"
classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    img_resp = urllib.request.urlopen(url)
    img_arr = np.array(bytearray(img_resp.read()), dtype = np.uint8)
    frame = cv2.imdecode(img_arr, -1)
    faces = classifier.detectMultiScale(frame)

    for face in faces:
        x, y, w, h = face
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)

    cv2.imshow("My Window", frame)

    key = cv2.waitKey(1000)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()