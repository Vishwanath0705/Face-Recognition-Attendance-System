import cv2
import cv2.data
import os

face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

eyes_cascade_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
eyes_cascade = cv2.CascadeClassifier(eyes_cascade_path)

smile_cascade_path = cv2.data.haarcascades + "haarcascade_smile.xml"
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

dataset_path = r"create a dataset folder and copy the path"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)


cap = cv2.VideoCapture(0)
face_id = input("Enter the name of the person: ")
print("Initializing face capture. Please look at the camera...")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Draw rectangle around face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eyes_cascade.detectMultiScale(roi_gray)
        eye_detected = len(eyes) > 0
        
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        smile_detected = len(smiles) > 0

        if eye_detected and smile_detected:
            count += 1
            file_path = os.path.join(dataset_path, f"User.{face_id}.{count}.jpg")
            cv2.imwrite(file_path, roi_color)  
            print(f"Image {count} captured and saved.")

        cv2.putText(frame, f"Captured: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('image', frame)

    k = cv2.waitKey(1)
    if k == ord('q'):  
        break
    if count >= 50:  
        break

print("Exiting program and cleaning up.")
cap.release()
cv2.destroyAllWindows()
