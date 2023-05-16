import os
import cv2
import numpy as np
from PIL import Image

path = "D:\\opencv\\imgs\\"
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('D:\\opencv\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')


def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        img = Image.open(image_path).convert('L')
        img_np = np.array(img, 'uint8')
        if os.path.split(image_path)[-1].split(".")[-1] != 'jpg':
            continue

        id = int(os.path.split(image_path)[-1].split(".")[0])
        faces = detector.detectMultiScale(img_np)

        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y + h, x:x + w])
            ids.append(id)
    return face_samples, ids


faces, ids = get_images_and_labels(path)
recognizer.train(faces, np.array(ids))
recognizer.save('D:\\opencv\\trainner\\trainner.yml')
