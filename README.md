Eye Blink Counter

This project is an Eye Blink Counter built using OpenCV and MediaPipe in Python. The program detects eye blinks in real-time using a webcam feed and tracks the total blink count. It’s useful for applications such as monitoring driver drowsiness or studying human attention levels.

Features

Real-Time Eye Detection: Uses MediaPipe's Face Mesh for accurate facial landmark detection.

Eye Aspect Ratio (EAR): Calculates the Eye Aspect Ratio to detect eye blinks.

Blink Counting: Counts blinks and displays the total on the video feed.

Visual Feedback: Draws landmarks on the eyes to provide real-time visual feedback.

Requirements

Python 3.x

OpenCV

MediaPipe

SciPy

To install these packages, run:

pip install opencv-python mediapipe scipy

How It Works

Face Detection: The program initializes the webcam and uses MediaPipe’s FaceDetection and FaceMesh modules to detect the face and eye landmarks.

Eye Aspect Ratio Calculation: The EAR is calculated based on six eye landmarks per eye. When the EAR falls below a threshold (indicating a closed eye), it registers as a blink.

Blink Counting: Each time a blink is detected, the total blink count is updated and displayed on the screen.

Usage

git clone https://github.com/muralikarthikv/Eye_Blink_Counter.git

cd Eye_Blink_Counter

python main.py
