import cv2

cap = cv2.VideoCapture("https://192.168.0.100:8080")
classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:

    ret, frame = cap.read()

    if ret:
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