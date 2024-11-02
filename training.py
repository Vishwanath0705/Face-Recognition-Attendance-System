import cv2
import numpy as np
from PIL import Image
import os

face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(face_cascade_path)

path = r"D:\\projects1\\face_attendance_system\\datasets\\vishwa"
recognizer = cv2.face.LBPHFaceRecognizer_create()   

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[2])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids

print("Training Faces. It will take a few seconds")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))  

trainer_path = r'D:\\projects1\\face_attendance_system\\datasets\\vishwa\\trainer\\trainer.yml'
os.makedirs(os.path.dirname(trainer_path), exist_ok=True)
recognizer.write(trainer_path)
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
