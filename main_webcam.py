import cv2
import dlib
import numpy as np
import requests
import bz2
import os
import time
from collections import deque
from picamera2 import Picamera2

# --- GUI Imports ---
import tkinter as tk
from PIL import Image, ImageTk

# --- Configuration & Enhancements ---
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
DEBUG = True

# Frame-to-Frame Stability (Debouncing)
STATUS_BUFFER_SIZE = 15
STATUS_CONFIRMATION_THRESHOLD = 9

# --- Thresholds to Tune ---
HEAD_TILT_THRESHOLD = 17.0
MOUTH_ASYMMETRY_THRESHOLD = 0.03
EYE_ASYMMETRY_THRESHOLD = 0.7
# **ACTION REQUIRED**: This is the value you need to find and adjust!
# Run the script, observe the printed ratio, and set this value accordingly.
TONGUE_OUT_THRESHOLD = 0.35

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

def is_tongue_out(landmarks):
    """Detects if the tongue is sticking out using a ratio of inner mouth height to outer mouth width."""
    inner_lip_top = np.array([landmarks.part(62).x, landmarks.part(62).y])
    inner_lip_bottom = np.array([landmarks.part(66).x, landmarks.part(66).y])
    inner_mouth_height = np.linalg.norm(inner_lip_top - inner_lip_bottom)

    mouth_corner_left = np.array([landmarks.part(48).x, landmarks.part(48).y])
    mouth_corner_right = np.array([landmarks.part(54).x, landmarks.part(54).y])
    outer_mouth_width = np.linalg.norm(mouth_corner_left - mouth_corner_right)
    
    if outer_mouth_width == 0:
        return False

    ratio = inner_mouth_height / outer_mouth_width
    
    # This print will help you find the correct threshold
    print(f"Tongue Ratio: {ratio:.4f}")
    
    return ratio > TONGUE_OUT_THRESHOLD

# --- 3D Head Pose Estimation & Drawing ---

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
    _, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
    return angles[1], angles[0], angles[2], rot_vec, trans_vec, cam_matrix

def draw_3d_axes(img, rot_vec, trans_vec, cam_matrix, origin_pt):
    """Draws 3D X, Y, Z axes on the image."""
    axis = np.float32([[50,0,0], [0,50,0], [0,0,-50]]).reshape(-1,3)
    img_pts, _ = cv2.projectPoints(axis, rot_vec, trans_vec, cam_matrix, np.zeros((4,1)))
    o = tuple(np.int32(origin_pt))
    colors = [(0,0,255),(0,255,0),(255,0,0)] # BGR for Red, Green, Blue axes
    cv2.line(img, o, tuple(np.int32(img_pts[0].ravel())), colors[0], 3)
    cv2.line(img, o, tuple(np.int32(img_pts[1].ravel())), colors[1], 3)
    cv2.line(img, o, tuple(np.int32(img_pts[2].ravel())), colors[2], 3)

def draw_3d_bbox(img, rot_vec, trans_vec, cam_matrix):
    """Draws a 3D bounding box around the face."""
    box_pts = np.float32([[-90,-130,-120], [90,-130,-120], [90,110,-120], [-90,110,-120], 
                             [-90,-130,20], [90,-130,20], [90,110,20], [-90,110,20]])
    img_pts, _ = cv2.projectPoints(box_pts, rot_vec, trans_vec, cam_matrix, np.zeros((4,1)))
    img_pts = np.int32(img_pts).reshape(-1,2)
    cv2.drawContours(img, [img_pts[:4]], -1, (255,255,0), 2) # Back plane
    cv2.drawContours(img, [img_pts[4:]], -1, (0,255,255), 2) # Front plane
    for i in range(4): cv2.line(img, tuple(img_pts[i]), tuple(img_pts[i+4]), (255,0,255), 2)

def draw_info_panel(img, status, color, yaw, pitch, roll, mouth_asym, eye_asym):
    """Puts all the text information on the screen."""
    x, y, font, scale = 10, 20, cv2.FONT_HERSHEY_SIMPLEX, 0.6
    cv2.putText(img, f"Status: {status}", (x,y), font, 0.7, color, 2)
    y+=30; cv2.putText(img, f"Roll (Tilt): {roll:.1f}", (x,y), font, scale, (255,255,0), 1)
    y+=25; cv2.putText(img, f"Yaw: {yaw:.1f}", (x,y), font, scale, (255,255,0), 1)
    y+=25; cv2.putText(img, f"Mouth Asym: {mouth_asym:.3f}", (x,y), font, scale, (255,255,0), 1)
    y+=25; cv2.putText(img, f"Eye Asym: {eye_asym:.3f}", (x,y), font, scale, (255,255,0), 1)

# --- Main Processing Function ---

def process_frame(frame, detector, predictor, status_buffer):
    """Processes a single video frame to detect facial states and draw overlays."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    yaw, pitch, roll, mouth_asym, eye_asym = 0, 0, 0, 0, 0
    final_status, final_color = "No Face Detected", (100, 100, 100)

    if faces:
        landmarks = predictor(gray, faces[0])
        
        # --- Calculate All Metrics ---
        eye_asym = abs(get_eye_aspect_ratio(np.array([(p.x, p.y) for p in landmarks.parts()[36:42]])) - 
                       get_eye_aspect_ratio(np.array([(p.x, p.y) for p in landmarks.parts()[42:48]])))
        mouth_asym = get_mouth_asymmetry(landmarks)
        tongue_detected = is_tongue_out(landmarks)
        yaw, pitch, roll, rvec, tvec, cam_matrix = estimate_head_pose(landmarks, frame.shape)
        
        # --- Decision Logic with Adaptive Thresholds ---
        rot_factor = max(np.cos(np.radians(abs(yaw))) * np.cos(np.radians(abs(pitch))), 0.5)
        is_asymmetric = (mouth_asym > MOUTH_ASYMMETRY_THRESHOLD / rot_factor or 
                         eye_asym > EYE_ASYMMETRY_THRESHOLD / rot_factor)
        
        current_status = "Normal"
        if tongue_detected:
            current_status = "Tongue Out"
        elif abs(roll) > HEAD_TILT_THRESHOLD:
            current_status = "Head Tilted"
        elif is_asymmetric:
            current_status = "Asymmetry Detected"
        
        # --- Debounce Status for Stability ---
        status_buffer.append(current_status)
        if status_buffer.count("Tongue Out") >= STATUS_CONFIRMATION_THRESHOLD:
            final_status, final_color = "Tongue Out", (0, 165, 255) # Orange
        elif status_buffer.count("Asymmetry Detected") >= STATUS_CONFIRMATION_THRESHOLD:
            final_status, final_color = "Asymmetry Detected", (0, 0, 255)
        elif status_buffer.count("Head Tilted") >= STATUS_CONFIRMATION_THRESHOLD:
            final_status, final_color = "Head Tilted", (0, 255, 255)
        else:
            final_status, final_color = "Normal", (0, 255, 0)
        
        # --- Draw Visualizations ---
        draw_3d_axes(frame, rvec, tvec, cam_matrix, (landmarks.part(30).x, landmarks.part(30).y))
        draw_3d_bbox(frame, rvec, tvec, cam_matrix)
        for i in range(68):
            cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 255, 0), -1)
        if tongue_detected:
            for i in range(60, 68):
                cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 3, (255, 0, 0), -1)

    draw_info_panel(frame, final_status, final_color, yaw, pitch, roll, mouth_asym, eye_asym)
    return frame

# --- Main Application Class (GUI) ---

class App:
    def __init__(self, window_title="Facial Asymmetry Detection"):
        self.root = tk.Tk()
        self.root.title(window_title)

        download_and_extract_shape_predictor()
        print("Initializing detectors...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        self.status_buffer = deque(maxlen=STATUS_BUFFER_SIZE)
        
        print("Initializing camera...")
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"size": (640, 480)}))
        self.picam2.start()

        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()

        self.delay = 15 # ms for screen update
        self.update()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def update(self):
        """Captures a frame, processes it, and displays it on the canvas."""
        frame = self.picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        frame = cv2.flip(frame, 1)

        processed_frame = process_frame(frame, self.detector, self.predictor, self.status_buffer)
        
        img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.root.after(self.delay, self.update)

    def on_closing(self):
        """Handles graceful shutdown of the camera and GUI."""
        print("Shutting down...")
        self.picam2.stop()
        self.root.destroy()

if __name__ == '__main__':
    App()
