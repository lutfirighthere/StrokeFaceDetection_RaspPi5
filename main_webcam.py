from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import dlib
from scipy.stats import entropy

picam2 = Picamera2()
picam2.start()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

def estimate_head_pose(landmarks, img):
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),     # Nose tip
        (landmarks.part(8).x, landmarks.part(8).y),       # Chin
        (landmarks.part(36).x, landmarks.part(36).y),     # Left eye left corner
        (landmarks.part(45).x, landmarks.part(45).y),     # Right eye right corner
        (landmarks.part(48).x, landmarks.part(48).y),     # Left mouth corner
        (landmarks.part(54).x, landmarks.part(54).y)      # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -63.6, -12.5),         # Chin
        (-43.3, 32.7, -26.0),        # Left eye
        (43.3, 32.7, -26.0),         # Right eye
        (-28.9, -28.9, -24.1),       # Left mouth
        (28.9, -28.9, -24.1)         # Right mouth
    ])

    size = img.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)

    rmat, _ = cv.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv.RQDecomp3x3(rmat)
    yaw = angles[1]
    pitch = angles[0]
    roll = angles[2]
    return yaw, pitch, roll

def face_landmarks(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        # --- Mouth: lines first, then dots on top ---
        x1, y1 = landmarks.part(48).x, landmarks.part(48).y
        x2, y2 = landmarks.part(54).x, landmarks.part(54).y
        x3, y3 = landmarks.part(30).x, landmarks.part(30).y

        mouth_pts = [(x1, y1), (x3, y3), (x2, y2)]
        # Draw blue lines first (triangle)
        cv.line(img, mouth_pts[0], mouth_pts[1], (255, 0, 0), 2)
        cv.line(img, mouth_pts[1], mouth_pts[2], (255, 0, 0), 2)
        cv.line(img, mouth_pts[2], mouth_pts[0], (255, 0, 0), 2)
        # Draw neon green dots on top
        for pt in mouth_pts:
            cv.circle(img, pt, 4, (0, 255, 0), -1)

        # Optionally: draw angle label
        cv.putText(img, f"Mouth angle: {np.abs(np.degrees(np.arctan2(y2-y1, x2-x1))):.2f}",
                   (x2+10, y2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 255), 1)

        # --- Your existing calculation/logic ---
        x4, y4 = landmarks.part(28).x, landmarks.part(28).y
        A, B, C, D = np.array([x1, y1]), np.array([x2, y2]), np.array([x3, y3]), np.array([x4, y4])
        AD, AC = D - A, C - A
        BD, BC = D - B, C - B
        angle_degrees_DAC = np.degrees(np.arccos(np.dot(AD, AC) / (np.linalg.norm(AD) * np.linalg.norm(AC))))
        angle_degrees_CBD = np.degrees(np.arccos(np.dot(BD, BC) / (np.linalg.norm(BD) * np.linalg.norm(BC))))
        S_mouthangle = np.abs(angle_degrees_CBD - angle_degrees_DAC)

        kl_left, tilt_left = process_eye_landmarks(landmarks, img, left=True)
        kl_right, tilt_right = process_eye_landmarks(landmarks, img, left=False)
        eye_tilt_diff = abs(tilt_left - tilt_right)

        face_width = abs(landmarks.part(16).x - landmarks.part(0).x)
        norm_mouth = S_mouthangle / face_width
        norm_eyes = eye_tilt_diff / face_width

        yaw, pitch, roll = estimate_head_pose(landmarks, img)
        cv.putText(img, f"Yaw: {yaw:.2f}", (10, 230), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if abs(yaw) > 17.5:
            cv.putText(img, "angle too large", (50, 50), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
        elif norm_mouth < 0.035 and norm_eyes < 0.02:
            cv.putText(img, "no stroke", (50, 50), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv.putText(img, "stroke", (50, 50), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

try:
    while True:
        frame = picam2.capture_array()
        frame = cv.flip(frame, 1)
        face_landmarks(frame)
        cv.imshow("Face Landmarks", frame)
        if cv.waitKey(1) == 27:
            break
except KeyboardInterrupt:
    print("Keyboard interrupt detected, exiting cleanly.")
finally:
    cv.destroyAllWindows()
    picam2.stop()
