import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('D:\\opencv\\trainner\\trainner.yml')
face_cascade = cv2.CascadeClassifier('D:\\opencv\\opencv\\sources\\data\\haarcascades'
                                     '\\haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
idnum = 0

cam = cv2.VideoCapture(0)
cam.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

names = ['xkx', 'sy', 'lzj']

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH))
    )

    if len(faces) != 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if confidence < 80:
                idum = names[idnum]
                confidence = "{0}%".format(round(100 - confidence))
            else:
                idum = "unknown"
                confidence = "{0}%".format(round(100 - confidence))
            cv2.putText(img, str(idum), (x + 5, y - 5), font, 1, (0, 0, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (0, 0, 255), 2)

    cv2.imshow('camera', img)
    k = cv2.waitKey(10)
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()