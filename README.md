# Face Attendance System
A face recognition-based attendance system using OpenCV and LBPH Face Recognizer.

## 📌 Features
Register new faces with unique registration numbers.
- Prevents overwriting if the registration number already exists.
- Detects faces and logs attendance to CSV files.
- Ignores unrecognized faces in attendance logging.
- Uses OpenCV’s Haarcascade for face detection.
- Saves trained face model for future use.

## 🛠️ Installation
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/userx0y/face-attendance.git
cd face-attendance


virtual environment (optional)
python -m venv attendance_env
attendance_sys_env\Scripts\activate (on windows)

structure of the project:-
attendance/
│── dataset/              # Stores registered face images
│── records/              # Stores attendance log CSVs
│── face_attendance.py    # Main program
│── face_model.xml        # Saved trained model
│── requirements.txt      # Dependencies
│── README.md             # Project documentation

