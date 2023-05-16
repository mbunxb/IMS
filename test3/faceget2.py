import cv2

cap = cv2.VideoCapture(1)
face_detector = cv2.CascadeClassifier('D:\\opencv\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')
face_id = input('User data input,Look at the camera and wait ...\n')
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if ret is True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + w), (255, 0, 0))
        count += 1
        cv2.imwrite("D:\\opencv\\imgs\\" + str(face_id) + '.' + str(count) + '.jpg', gray[y:y + h, x:x + w])
        cv2.imshow('image', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif count >= 100:
        break

cap.release()
cv2.destroyAllWindows()