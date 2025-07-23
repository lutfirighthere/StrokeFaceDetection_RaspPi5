## Owen Kim OpenCV mini project (Picamera2 version for Raspberry Pi 5)
## Pre-stroke prediction based on face detection.

from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import dlib
from scipy.stats import entropy

# Initialize PiCamera2
picam2 = Picamera2()
picam2.start()

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Must be present

def calculate_kl_divergence(upper, lower):
    hist_upper, _ = np.histogram(upper[:, 0], bins=10, range=(0, 1), density=True)
    hist_lower, _ = np.histogram(lower[:, 0], bins=10, range=(0, 1), density=True)
    hist_upper += 1e-6
    hist_lower += 1e-6
    return entropy(hist_upper, hist_lower)

def calculate_eye_tilt(inner, outer):
    dx, dy = outer[0] - inner[0], outer[1] - inner[1]
    return np.degrees(np.arctan2(dy, dx))

def process_eye_landmarks(landmarks, img, left=True):
    eye_pts = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in (range(36, 42) if left else range(42, 48))])
    for i in range(5):
        cv.circle(img, tuple(eye_pts[i]), 2, (0, 255, 0), -1)
        cv.line(img, tuple(eye_pts[i]), tuple(eye_pts[i + 1]), (255, 0, 0), 1)
    cv.circle(img, tuple(eye_pts[-1]), 2, (0, 255, 0), -1)
    cv.line(img, tuple(eye_pts[-1]), tuple(eye_pts[0]), (255, 0, 0), 1)
    upper_lid = eye_pts[[1, 2]]
    lower_lid = eye_pts[[4, 5]]
    kl_score = calculate_kl_divergence(upper_lid, lower_lid)
    eye_tilt = calculate_eye_tilt(eye_pts[0], eye_pts[3])
    return kl_score, eye_tilt

def face_landmarks(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        x1, y1 = landmarks.part(48).x, landmarks.part(48).y
        x2, y2 = landmarks.part(54).x, landmarks.part(54).y
        x3, y3 = landmarks.part(30).x, landmarks.part(30).y
        x4, y4 = landmarks.part(28).x, landmarks.part(28).y

        cv.circle(img, (x1, y1), 2, (0, 255, 0), -1)
        cv.circle(img, (x2, y2), 2, (0, 255, 0), -1)
        cv.circle(img, (x3, y3), 2, (0, 255, 0), -1)
        cv.circle(img, (x4, y4), 2, (0, 255, 0), -1)

        cv.line(img, (x1, y1), (x3, y3), (255, 0, 0), 1)
        cv.line(img, (x2, y2), (x3, y3), (255, 0, 0), 1)
        cv.line(img, (x1, y1), (x4, y4), (255, 0, 0), 1)
        cv.line(img, (x2, y2), (x4, y4), (255, 0, 0), 1)

        A, B, C, D = np.array([x1, y1]), np.array([x2, y2]), np.array([x3, y3]), np.array([x4, y4])
        AD, AC = D - A, C - A
        BD, BC = D - B, C - B
        angle_degrees_DAC = np.degrees(np.arccos(np.dot(AD, AC) / (np.linalg.norm(AD) * np.linalg.norm(AC))))
        angle_degrees_CBD = np.degrees(np.arccos(np.dot(BD, BC) / (np.linalg.norm(BD) * np.linalg.norm(BC))))
        S_mouthangle = np.abs(angle_degrees_CBD - angle_degrees_DAC)
        print("mouth: ", S_mouthangle)

        kl_left, tilt_left = process_eye_landmarks(landmarks, img, left=True)
        kl_right, tilt_right = process_eye_landmarks(landmarks, img, left=False)
        eye_tilt_diff = abs(tilt_left - tilt_right)
        print("eye: ", eye_tilt_diff)

        cv.circle(img, (270, 200), 2, (0, 0, 255), 1)
        cv.circle(img, (370, 200), 2, (0, 0, 255), 1)
        cv.line(img, (300, 300), (340, 300), (0, 0, 255), 2)

        face_width = abs(landmarks.part(16).x - landmarks.part(0).x)
        norm_mouth = S_mouthangle / face_width
        norm_eyes = eye_tilt_diff / face_width

        if norm_mouth < 0.035 and norm_eyes < 0.02:
            cv.putText(img, "no stroke", (50, 50), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv.putText(img, "stroke", (50, 50), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

# Main loop
while True:
    frame = picam2.capture_array()
    frame = cv.flip(frame, 1)
    face_landmarks(frame)
    cv.imshow("Face Landmarks", frame)
    if cv.waitKey(1) == 27:
        break

cv.destroyAllWindows()
