import cv2
import cv2.data
import numpy as np
import os
import pandas as pd
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
import io
from PIL import Image

face_cascade_path = cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
dataset_path =  r"D:\\projects1\\face_attendance_system\datasets\\vishwa"
excel_path = r"D:\\projects1\\face_attendance_system\\attendance.xlsx"

face_cascade = cv2.CascadeClassifier(face_cascade_path)

dataset_images = []
names = []

for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg"):
        img_path = os.path.join(dataset_path,filename)
        img = cv2.imread(img_path)
        if img is not None:
            dataset_images.append(img)
            name = filename.split('.')[1]
            names.append(name)

def preprocess_image(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image
    return cv2.GaussianBlur(gray,(5,5),0)

def log_attendance(name,excel_path):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    if os.path.exists(excel_path):
        attendace_df = pd.read_excel(excel_path,engine='openpyxl')
    else:
        attendace_df = pd.DataFrame(columns=["Name","Date","Time"])

    new_entry = {"Name":name,"Date":date,"Time":time}
    attendace_df = pd.concat([attendace_df,pd.DataFrame([new_entry])],ignore_index=True)
    attendace_df.to_excel(excel_path,index=False,engine='openpyxl')

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

threshold = 0.5
recognized_names = set()
attendance_marked = False

while True:
    ret,frame = cam.read()
    if not ret:
        print("Failed to capture frame")
        break

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,4)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = preprocess_image(gray[y:y + h, x:x + w])

        best_match = "Unknown"
        best_score = threshold

        for dataset_img, name in zip(dataset_images, names):
            dataset_img_resized = cv2.resize(dataset_img, (w, h))
            dataset_img_gray = preprocess_image(dataset_img_resized)
            score = ssim(roi_gray, dataset_img_gray)

            if score > best_score:
                best_score = score
                best_match = name

        if best_match != "Unknown" and best_match not in recognized_names:
            recognized_names.add(best_match)
            log_attendance(best_match, excel_path=excel_path)
            attendance_marked = True

        cv2.putText(frame, best_match, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    if attendance_marked:
        cv2.putText(frame,"Attendance Marked",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        attendance_marked = False
        break

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()