import cv2
import mediapipe as mp
from scipy.spatial import distance

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Variables to keep track of the blink count
count = 0
total = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            face_results = face_mesh.process(img_rgb)
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # Extract coordinates for left and right eyes
                    left_eye_indices = [362, 385, 387, 263, 373, 380]
                    right_eye_indices = [33, 160, 158, 133, 153, 144]

                    left_eye = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in left_eye_indices]
                    right_eye = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in right_eye_indices]

                    # Convert normalized coordinates to pixel coordinates
                    left_eye = [(int(x * img.shape[1]), int(y * img.shape[0])) for (x, y) in left_eye]
                    right_eye = [(int(x * img.shape[1]), int(y * img.shape[0])) for (x, y) in right_eye]

                    leftEAR = eye_aspect_ratio(left_eye)
                    rightEAR = eye_aspect_ratio(right_eye)

                    ear = (leftEAR + rightEAR) / 2.0

                    if ear < 0.3:
                        count += 1
                    else:
                        if count >= 3:
                            total += 1
                        count = 0

                    # Draw the landmarks
                    for (x, y) in left_eye + right_eye:
                        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

    cv2.putText(img, "Blink Count: {}".format(total), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Video', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()