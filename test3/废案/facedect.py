# -----检测、校验并输出结果-----
import cv2


# 准备好识别方法
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 使用之前训练好的模型
recognizer.read('D:\\opencv\\trainner\\trainner.yml')

# 再次调用人脸分类器
cascade_path = "D:\\opencv\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# 加载一个字体，用于识别后，在图片上标注出对象的名字
font = cv2.FONT_HERSHEY_SIMPLEX

idnum = 0
# 设置好与ID号码对应的用户名，如下，如0对应的就是初始

names = ['xkx', 'cym', 'lzj', 'sy', 'user3']

# 调用摄像头
cam = cv2.VideoCapture(1)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    if not ret:  # 如果读取不正常，重新获取一帧图片
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 识别人脸
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(int(minW), int(minH))
    )
    # 进行校验

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        # 计算出一个检验结果
        if confidence < 100:
            idum = names[idnum]
            confidence = "{0}%".format(round(100 - confidence))
        else:
            idum = "unknown"
            confidence = "{0}%".format(round(100 - confidence))

        # 输出检验结果以及用户名
        cv2.putText(img, str(idum), (x + 5, y - 5), font, 1, (0, 0, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (0, 0, 255), 2)

        # 展示结果
        cv2.imshow('camera', img)
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break  # q/Q 键中断循环

# 释放资源
cam.release()
cv2.destroyAllWindows()
