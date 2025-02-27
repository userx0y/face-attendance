import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from datetime import datetime
import threading
import pandas as pd
#paths
DATASET_PATH = "dataset"
RECORDS_PATH = "records"
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ensure required folders exists
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(RECORDS_PATH, exist_ok=True)

#Load Haarcascade for face detections
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

#initialize LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

class FaceAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Attendance System")
        self.root.geometry("500x300")

        # system Variables
        self.running = False
        self.log_file = None
        self.detected_faces = set()

        #UI Setup
        self.create_widgets()

        #load trained model if exists
        self.load_trained_model()

    def create_widgets(self):
        ttk.Label(self.root, text="Face Attendance System", font=("Arial", 16)).pack(pady=10)
        self.btn_register = ttk.Button(self.root, text="Register New Face", command=self.register_user)
        self.btn_register.pack(pady=5)
        self.btn_toggle = ttk.Button(self.root, text="Start Detection", command=self.toggle_detection)
        self.btn_toggle.pack(pady=5)
        self.status_label = ttk.Label(self.root, text="Status: Ready", font=("Arial", 12))
        self.status_label.pack(pady=10)

    def register_user(self):
        reg_no = simpledialog.askstring("Register User", "Enter Registration Number:")
        if not reg_no:
            return

        user_folder = os.path.join(DATASET_PATH, reg_no)
        
        if os.path.exists(user_folder):
            messagebox.showwarning("Warning", "User with this registration number already exists!")
            return
        
        os.makedirs(user_folder, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Webcam could not be opened!")
            return

        face_count = 0
        while face_count < 10:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                file_path = os.path.join(user_folder, f"{reg_no}_{face_count}.jpg")
                cv2.imwrite(file_path, face_img)
                face_count += 1

            cv2.imshow("Face Registration", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Success", "User registered successfully!")
        self.train_faces()

    def train_faces(self):
        images, labels = [], []
        for reg_no in os.listdir(DATASET_PATH):
            reg_path = os.path.join(DATASET_PATH, reg_no)
            if os.path.isdir(reg_path):
                for img_file in os.listdir(reg_path):
                    img_path = os.path.join(reg_path, img_file)
                    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    images.append(gray_img)
                    labels.append(int(reg_no))

        if images:
            recognizer.train(images, np.array(labels))
            recognizer.save("face_model.xml")  #Save the trained model
            messagebox.showinfo("Success", "Face recognition model trained and saved!")

    def load_trained_model(self):
        if os.path.exists("face_model.xml"):
            recognizer.read("face_model.xml")

    def toggle_detection(self):
        if self.running:
            self.running = False
            self.btn_toggle.config(text="Start Detection")
            self.status_label.config(text="Status: Stopped")
        else:
            self.running = True
            self.detected_faces.clear()  #reset detected facess oneach detection start
            self.btn_toggle.config(text="Stop Detection")
            self.status_label.config(text="Status: Running")
            self.log_file = os.path.join(RECORDS_PATH, datetime.now().strftime('%d-%m-%Y_%H-%M-%S') + ".csv")
            pd.DataFrame(columns=["Name", "Timestamp"]).to_csv(self.log_file, index=False)
            threading.Thread(target=self.run_detection, daemon=True).start()

    def run_detection(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Webcam could not be opened!")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                try:
                    label, confidence = recognizer.predict(face_img)
                    name = str(label) if confidence < 60 else "Unknown"
                    if name != "Unknown" and name not in self.detected_faces:
                        self.detected_faces.add(name)
                        self.log_attendance(name)
                except Exception:
                    pass

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.running = False
        self.btn_toggle.config(text="Start Detection")
        self.status_label.config(text="Status: Stopped")

    def log_attendance(self, name):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df = pd.read_csv(self.log_file)
        df = pd.concat([df, pd.DataFrame([{"Name": name, "Timestamp": timestamp}])], ignore_index=True)
        df.to_csv(self.log_file, index=False)
# 
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAttendanceApp(root)
    root.mainloop()