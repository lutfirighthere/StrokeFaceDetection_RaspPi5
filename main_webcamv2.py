from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import dlib
from scipy.stats import entropy

# Initialize camera
picam2 = Picamera2()
picam2.start()

# Load face detector and landmark predictor
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
    idx = range(36, 42) if left else range(42, 48)
    eye_pts = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in idx])
    # Draw eye outline
    for i in range(5):
        cv.circle(img, tuple(eye_pts[i]), 2, (0, 255, 0), -1)
        cv.line(img, tuple(eye_pts[i]), tuple(eye_pts[i + 1]), (255, 0, 0), 1)
    cv.circle(img, tuple(eye_pts[-1]), 2, (0, 255, 0), -1)
    cv.line(img, tuple(eye_pts[-1]), tuple(eye_pts[0]), (255, 0, 0), 1)
    # Compute KL divergence between upper and lower lids
    upper_lid = eye_pts[[1, 2]]
    lower_lid = eye_pts[[4, 5]]
    kl_score = calculate_kl_divergence(upper_lid, lower_lid)
    # Compute tilt of eye
    eye_tilt = calculate_eye_tilt(eye_pts[0], eye_pts[3])
    return kl_score, eye_tilt

def estimate_head_pose(landmarks, img):
    # 2D image points
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
        (landmarks.part(8).x, landmarks.part(8).y),    # Chin
        (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
        (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
        (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
        (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
    ], dtype="double")

    # Corresponding 3D model points
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
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv.SOLVEPNP_ITERATIVE
    )

    # Convert rotation vector to Euler angles
    rmat, _ = cv.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv.RQDecomp3x3(rmat)
    yaw = angles[1]
    pitch = angles[0]
    roll = angles[2]
    return yaw, pitch, roll, rotation_vector, translation_vector, camera_matrix

def draw_axes(img, camera_matrix, rot_vec, trans_vec, origin_pt):
    # Draw 3D axes to show orientation
    axis = np.float32([[50,0,0], [0,50,0], [0,0,50]])
    imgpts, _ = cv.projectPoints(axis, rot_vec, trans_vec, camera_matrix, np.zeros((4,1)))
    origin = tuple(int(x) for x in origin_pt)
    for i, pt in enumerate(imgpts):
        pt = tuple(int(x) for x in pt.ravel())
        color = (0,0,255) if i==0 else ((0,255,0) if i==1 else (255,0,0))
        cv.line(img, origin, pt, color, 2)

def face_landmarks(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        # --- Mouth triangle & angle ---
        A = np.array([landmarks.part(48).x, landmarks.part(48).y])
        B = np.array([landmarks.part(54).x, landmarks.part(54).y])
        C = np.array([landmarks.part(30).x, landmarks.part(30).y])
        D = np.array([landmarks.part(28).x, landmarks.part(28).y])

        mouth_pts = [tuple(A), tuple(C), tuple(B)]
        # Draw triangle
        cv.polylines(img, [np.array(mouth_pts)], isClosed=True, color=(255,0,0), thickness=2)
        # Dots
        for pt in mouth_pts:
            cv.circle(img, pt, 4, (0,255,0), -1)

        # Mouth angles
        angle_DAC = np.degrees(np.arccos(np.dot(D-A, C-A) /
                             (np.linalg.norm(D-A)*np.linalg.norm(C-A))))
        angle_CBD = np.degrees(np.arccos(np.dot(D-B, C-B) /
                             (np.linalg.norm(D-B)*np.linalg.norm(C-B))))
        S_mouthangle = abs(angle_CBD - angle_DAC)

        # --- Eye processing ---
        kl_left, tilt_left   = process_eye_landmarks(landmarks, img, left=True)
        kl_right, tilt_right = process_eye_landmarks(landmarks, img, left=False)
        eye_tilt_diff = abs(tilt_left - tilt_right)

        # Normalize by face width
        face_width = abs(landmarks.part(16).x - landmarks.part(0).x)
        norm_mouth = S_mouthangle / face_width
        norm_eyes  = eye_tilt_diff / face_width

        # --- Head pose estimation ---
        yaw, pitch, roll, rot_vec, trans_vec, cam_mtx = estimate_head_pose(landmarks, img)
        cv.putText(img, f"Yaw: {yaw:.1f}",   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv.putText(img, f"Pitch: {pitch:.1f}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv.putText(img, f"Roll: {roll:.1f}",  (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        # Draw axes at nose tip
        nose_pt = (landmarks.part(30).x, landmarks.part(30).y)
        draw_axes(img, cam_mtx, rot_vec, trans_vec, nose_pt)

        # --- Adaptive thresholding based on yaw ---
        angle_weight = max(0.0, 1.0 - abs(yaw) / 20.0)
        mouth_thresh = 0.035 * angle_weight
        eye_thresh   = 0.020 * angle_weight

        # Decision logic
        if abs(yaw) > 25 or abs(pitch) > 20:
            cv.putText(img, "Face rotated", (50, 110), cv.FONT_HERSHEY_COMPLEX, 0.6, (0,255,255), 2)
        elif norm_mouth < mouth_thresh and norm_eyes < eye_thresh:
            cv.putText(img, "No stroke", (50, 110), cv.FONT_HERSHEY_COMPLEX, 0.6, (0,255,0), 2)
        else:
            cv.putText(img, "Stroke detected", (50, 110), cv.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 2)

try:
    while True:
        frame = picam2.capture_array()
        frame = cv.flip(frame, 1)
        face_landmarks(frame)
        cv.imshow("Stroke Detection", frame)
        if cv.waitKey(1) & 0xFF == 27:  # ESC to exit
            break
except KeyboardInterrupt:
    print("Exiting on user interrupt.")
finally:
    cv.destroyAllWindows()
    picam2.stop()
