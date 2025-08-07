import cv2
import dlib
import numpy as np
import requests
import bz2
import os
import time
from collections import deque
from picamera2 import Picamera2

# --- Configuration & Enhancements ---
DEBUG = True
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# 1. ENHANCEMENT: Frame-to-Frame Stability (Debouncing)
# We use a deque to store the status of the last N frames.
STATUS_BUFFER_SIZE = 15  # Use last 15 frames for stability
STATUS_CONFIRMATION_THRESHOLD = 10 # At least 10 of the last 15 frames must agree

# 2. ENHANCEMENT: Adaptive Thresholds
# Base thresholds for a front-facing person
HEAD_TILT_THRESHOLD = 17.0  # Degrees for roll/tilt (slightly increased)
MOUTH_ASYMMETRY_THRESHOLD = 0.07 # Normalized score
EYE_ASYMMETRY_THRESHOLD = 0.12   # Normalized score (ratio of EARs)


# --- Utility Functions ---

def download_and_extract_shape_predictor():
    """Downloads and extracts the dlib facial landmark predictor model if it doesn't exist."""
    if os.path.exists(SHAPE_PREDICTOR_PATH):
        if DEBUG: print(f"'{SHAPE_PREDICTOR_PATH}' already exists.")
        return
    
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    print(f"Downloading '{SHAPE_PREDICTOR_PATH}' from {url}...")
    
    try:
        req = requests.get(url, stream=True)
        req.raise_for_status()
        with open(SHAPE_PREDICTOR_PATH + ".bz2", "wb") as f_bz2:
            for chunk in req.iter_content(chunk_size=8192):
                f_bz2.write(chunk)
        print("Download complete. Decompressing...")
        with bz2.BZ2File(SHAPE_PREDICTOR_PATH + ".bz2", 'rb') as f_in, open(SHAPE_PREDICTOR_PATH, 'wb') as f_out:
            f_out.write(f_in.read())
        os.remove(SHAPE_PREDICTOR_PATH + ".bz2")
        print("Model ready.")
    except Exception as e:
        print(f"Error downloading or extracting model: {e}")
        exit()

def get_eye_aspect_ratio(eye_points):
    """Calculates the Eye Aspect Ratio (EAR)."""
    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])
    h = np.linalg.norm(eye_points[0] - eye_points[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0

def get_mouth_asymmetry(landmarks):
    """Calculates a normalized mouth asymmetry score."""
    nose_tip = np.array([landmarks.part(30).x, landmarks.part(30).y])
    left_mouth_corner = np.array([landmarks.part(48).x, landmarks.part(48).y])
    right_mouth_corner = np.array([landmarks.part(54).x, landmarks.part(54).y])
    
    dist_left = abs(left_mouth_corner[1] - nose_tip[1])
    dist_right = abs(right_mouth_corner[1] - nose_tip[1])
    asymmetry_raw = abs(dist_left - dist_right)
    
    face_width = np.linalg.norm(np.array([landmarks.part(0).x, landmarks.part(0).y]) - np.array([landmarks.part(16).x, landmarks.part(16).y]))
    return asymmetry_raw / face_width if face_width > 0 else 0


# --- 3D Head Pose Estimation ---

def estimate_head_pose(landmarks, frame_shape):
    """Estimates head pose (yaw, pitch, roll)."""
    h, w = frame_shape[:2]
    
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ]) / 4.5
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y), (landmarks.part(8).x, landmarks.part(8).y),
        (landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(45).x, landmarks.part(45).y),
        (landmarks.part(48).x, landmarks.part(48).y), (landmarks.part(54).x, landmarks.part(54).y)
    ], dtype="double")
    
    focal_length = w
    center = (w / 2, h / 2)
    cam_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    
    success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
    yaw, pitch, roll = angles[1], angles[0], angles[2]
    return yaw, pitch, roll, rot_vec, trans_vec, cam_matrix


# --- Drawing & Visualization ---

def draw_3d_axes(img, rot_vec, trans_vec, cam_matrix, origin_pt):
    """Draws 3D X, Y, Z axes on the image."""
    axis = np.float32([[50, 0, 0], [0, 50, 0], [0, 0, -50]]).reshape(-1, 3)
    img_pts, _ = cv2.projectPoints(axis, rot_vec, trans_vec, cam_matrix, np.zeros((4,1)))
    o, colors = tuple(np.int32(origin_pt)), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for i, p in enumerate(img_pts):
        cv2.line(img, o, tuple(np.int32(p.ravel())), colors[i], 3)

def draw_3d_bbox(img, rot_vec, trans_vec, cam_matrix):
    """Draws a 3D bounding box around the face."""
    box_pts = np.float32([[-75,-100,-50], [75,-100,-50], [75,100,-50], [-75,100,-50], [-75,-100,50], [75,-100,50], [75,100,50], [-75,100,50]])
    img_pts, _ = cv2.projectPoints(box_pts, rot_vec, trans_vec, cam_matrix, np.zeros((4,1)))
    img_pts = np.int32(img_pts).reshape(-1, 2)
    cv2.drawContours(img, [img_pts[:4]], -1, (255, 255, 0), 2)
    cv2.drawContours(img, [img_pts[4:]], -1, (0, 255, 255), 2)
    for i in range(4): cv2.line(img, tuple(img_pts[i]), tuple(img_pts[i+4]), (255, 0, 255), 2)

def draw_info_panel(img, status, color, yaw, pitch, roll, mouth_asym, eye_asym):
    """Puts all the text information on the screen."""
    x_offset, y_offset, font, scale = 10, 20, cv2.FONT_HERSHEY_SIMPLEX, 0.6
    cv2.putText(img, f"Status: {status}", (x_offset, y_offset), font, 0.7, color, 2)
    y_offset += 30; cv2.putText(img, f"Roll (Tilt): {roll:.1f}", (x_offset, y_offset), font, scale, (255, 255, 0), 1)
    y_offset += 25; cv2.putText(img, f"Yaw: {yaw:.1f}", (x_offset, y_offset), font, scale, (255, 255, 0), 1)
    y_offset += 25; cv2.putText(img, f"Mouth Asym: {mouth_asym:.3f}", (x_offset, y_offset), font, scale, (255, 255, 0), 1)
    y_offset += 25; cv2.putText(img, f"Eye Asym: {eye_asym:.3f}", (x_offset, y_offset), font, scale, (255, 255, 0), 1)


# --- Main Processing Function ---

def process_frame(frame, detector, predictor, status_buffer):
    """Processes a single video frame for detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    
    # Default values if no face is detected
    yaw, pitch, roll, mouth_asymmetry, eye_asymmetry = 0, 0, 0, 0, 0
    final_status, final_color = "No Face Detected", (100, 100, 100)

    if len(faces) > 0:
        face = faces[0] # Process only the first detected face
        landmarks = predictor(gray, face)
        
        # 1. Calculate Asymmetry Metrics
        left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        ear_left = get_eye_aspect_ratio(left_eye_pts)
        ear_right = get_eye_aspect_ratio(right_eye_pts)
        eye_asymmetry = abs(ear_left - ear_right) / max(ear_left, ear_right, 1e-6)
        mouth_asymmetry = get_mouth_asymmetry(landmarks)

        # 2. Estimate 3D Head Pose
        yaw, pitch, roll, rot_vec, trans_vec, cam_matrix = estimate_head_pose(landmarks, frame.shape)
        
        # 3. Decision Logic with Enhancements
        # Adaptive Thresholds: Make thresholds more lenient if the head is turned
        rot_factor = max(np.cos(np.radians(abs(yaw))) * np.cos(np.radians(abs(pitch))), 0.5)
        adj_mouth_th = MOUTH_ASYMMETRY_THRESHOLD / rot_factor
        adj_eye_th = EYE_ASYMMETRY_THRESHOLD / rot_factor

        is_tilted = abs(roll) > HEAD_TILT_THRESHOLD
        is_asymmetric = (mouth_asymmetry > adj_mouth_th or eye_asymmetry > adj_eye_th)
        
        # Determine the status for the CURRENT frame
        current_status = "Normal"
        if is_tilted: current_status = "Head Tilted"
        elif is_asymmetric: current_status = "Asymmetry Detected"
        
        # Debouncing: Determine the FINAL status based on the buffer
        status_buffer.append(current_status)
        if status_buffer.count("Asymmetry Detected") >= STATUS_CONFIRMATION_THRESHOLD:
            final_status, final_color = "Asymmetry Detected", (0, 0, 255)
        elif status_buffer.count("Head Tilted") >= STATUS_CONFIRMATION_THRESHOLD:
            final_status, final_color = "Head Tilted", (0, 255, 255)
        else:
            final_status, final_color = "Normal", (0, 255, 0)
            
        # 4. Draw Visualizations
        nose_tip_2d = (landmarks.part(30).x, landmarks.part(30).y)
        draw_3d_axes(frame, rot_vec, trans_vec, cam_matrix, nose_tip_2d)
        draw_3d_bbox(frame, rot_vec, trans_vec, cam_matrix)
        for i in range(68): cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 255, 0), -1)

    draw_info_panel(frame, final_status, final_color, yaw, pitch, roll, mouth_asymmetry, eye_asymmetry)
    return frame

# --- Main Execution ---

def main():
    download_and_extract_shape_predictor()
    print("Initializing detectors...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    
    print("Initializing camera...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()

    # Initialize enhancement variables
    status_buffer = deque(maxlen=STATUS_BUFFER_SIZE)
    prev_time = 0

    print("Starting video stream. Press 'ESC' to exit.")
    try:
        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            frame = cv2.flip(frame, 1)

            processed_frame = process_frame(frame, detector, predictor, status_buffer)

            # Calculate and display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (processed_frame.shape[1] - 80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
            
            cv2.imshow("Facial Asymmetry Detection", processed_frame)
            if cv2.waitKey(1) & 0xFF == 27: break
    except KeyboardInterrupt:
        print("\nExiting cleanly on user interrupt.")
    finally:
        print("Shutting down...")
        cv2.destroyAllWindows()
        picam2.stop()

if __name__ == '__main__':
    main()
