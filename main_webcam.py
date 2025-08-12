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

# --- Sensitivity Parameters ---
STD_MULTIPLIER = 1.75
SENSITIVITY_FACTOR = 0.85

# --- Default Thresholds ---
DEFAULT_HEAD_TILT = 17.0
DEFAULT_MOUTH_ASYMMETRY = 0.035
DEFAULT_EYE_ASYMMETRY = 0.075
DEFAULT_TONGUE_OUT = 0.4

# --- Maximum Thresholds ---
MAX_HEAD_TILT = 30.0
MAX_MOUTH_ASYMMETRY = 0.10
MAX_EYE_ASYMMETRY = 0.10
MAX_TONGUE_OUT = 0.50

# --- Minimum Thresholds ---
MIN_HEAD_TILT = 10.0
MIN_MOUTH_ASYMMETRY = 0.030
MIN_EYE_ASYMMETRY = 0.030
MIN_TONGUE_OUT = 0.25

# --- Active Thresholds ---
HEAD_TILT_THRESHOLD = max(min(DEFAULT_HEAD_TILT, MAX_HEAD_TILT), MIN_HEAD_TILT)
MOUTH_ASYMMETRY_THRESHOLD = max(min(DEFAULT_MOUTH_ASYMMETRY, MAX_MOUTH_ASYMMETRY), MIN_MOUTH_ASYMMETRY)
EYE_ASYMMETRY_THRESHOLD = max(min(DEFAULT_EYE_ASYMMETRY, MAX_EYE_ASYMMETRY), MIN_EYE_ASYMMETRY)
TONGUE_OUT_THRESHOLD = max(min(DEFAULT_TONGUE_OUT, MAX_TONGUE_OUT), MIN_TONGUE_OUT)

# --- Head Orientation Limits ---
MAX_YAW_FOR_ASSESSMENT = 22.0  # degrees
MAX_ROLL_FOR_ASSESSMENT = 7.0  # degrees

# --- Debounce Params ---
STATUS_BUFFER_SIZE = 15
STATUS_CONFIRMATION_THRESHOLD = 7

SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
DEBUG = True

def download_and_extract_shape_predictor():
    if os.path.exists(SHAPE_PREDICTOR_PATH):
        if DEBUG:
            print(f"'{SHAPE_PREDICTOR_PATH}' already exists.")
        return
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    print(f"Downloading shape predictor from {url}…")
    try:
        req = requests.get(url, stream=True)
        req.raise_for_status()
        with open(SHAPE_PREDICTOR_PATH + ".bz2", "wb") as f:
            for chunk in req.iter_content(8192):
                f.write(chunk)
        with bz2.BZ2File(SHAPE_PREDICTOR_PATH + ".bz2", 'rb') as f_in, open(SHAPE_PREDICTOR_PATH, 'wb') as f_out:
            f_out.write(f_in.read())
        os.remove(SHAPE_PREDICTOR_PATH + ".bz2")
        print("Model ready.")
    except Exception as e:
        print(f"Failed to download model: {e}")
        exit()

def get_eye_aspect_ratio(eye_points):
    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])
    h = np.linalg.norm(eye_points[0] - eye_points[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0

def get_mouth_asymmetry(landmarks):
    nose = np.array([landmarks.part(30).x, landmarks.part(30).y])
    left = np.array([landmarks.part(48).x, landmarks.part(48).y])
    right = np.array([landmarks.part(54).x, landmarks.part(54).y])
    dl = abs(left[1] - nose[1])
    dr = abs(right[1] - nose[1])
    raw = abs(dl - dr)
    width = np.linalg.norm(
        np.array([landmarks.part(0).x, landmarks.part(0).y]) -
        np.array([landmarks.part(16).x, landmarks.part(16).y])
    )
    return raw / width if width > 0 else 0

def is_tongue_out(landmarks):
    top = np.array([landmarks.part(62).x, landmarks.part(62).y])
    bottom = np.array([landmarks.part(66).x, landmarks.part(66).y])
    height = np.linalg.norm(top - bottom)
    left = np.array([landmarks.part(48).x, landmarks.part(48).y])
    right = np.array([landmarks.part(54).x, landmarks.part(54).y])
    width = np.linalg.norm(left - right)
    if width == 0:
        return False
    return (height / width) > TONGUE_OUT_THRESHOLD

def calibrate_camera(picam2, detector, predictor, samples=60, pause=0.2, min_face_frac=0.2):
    global HEAD_TILT_THRESHOLD, MOUTH_ASYMMETRY_THRESHOLD
    global EYE_ASYMMETRY_THRESHOLD, TONGUE_OUT_THRESHOLD

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    h_frame, w_frame = picam2.capture_array().shape[:2]

    def valid_face(faces):
        if len(faces) != 1:
            return False
        f = faces[0]
        fw, fh = f.right() - f.left(), f.bottom() - f.top()
        return fw >= min_face_frac * w_frame and fh >= min_face_frac * h_frame

    neutral_rolls, neutral_mouths, neutral_eyes = [], [], []
    tongue_ratios = []
    stroke_mouths, stroke_eyes = [], []

    # Phase 1 – Neutral
    print("Calibration 1/3: Neutral pose…")
    while len(neutral_rolls) < samples:
        frame = picam2.capture_array()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        cv2.putText(bgr, f"Neutral {len(neutral_rolls)+1}/{samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Calibration", bgr)
        cv2.waitKey(int(pause * 1000))

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        if not valid_face(faces):
            continue
        lm = predictor(gray, faces[0])
        _, _, roll, _, _, _ = estimate_head_pose(lm, frame.shape)
        neutral_rolls.append(abs(roll))
        neutral_mouths.append(get_mouth_asymmetry(lm))
        le = get_eye_aspect_ratio(np.array([(p.x, p.y) for p in lm.parts()[36:42]]))
        re = get_eye_aspect_ratio(np.array([(p.x, p.y) for p in lm.parts()[42:48]]))
        neutral_eyes.append(abs(le - re))

    # Phase 2 – Tongue Out
    print("Calibration 2/3: Tongue out…")
    while len(tongue_ratios) < samples:
        frame = picam2.capture_array()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        cv2.putText(bgr, f"Tongue {len(tongue_ratios)+1}/{samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Calibration", bgr)
        cv2.waitKey(int(pause * 1000))

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        if not valid_face(faces):
            continue
        lm = predictor(gray, faces[0])
        top = np.array([lm.part(62).x, lm.part(62).y])
        bot = np.array([lm.part(66).x, lm.part(66).y])
        left = np.array([lm.part(48).x, lm.part(48).y])
        right = np.array([lm.part(54).x, lm.part(54).y])
        h, w = np.linalg.norm(top - bot), np.linalg.norm(left - right)
        if w > 0:
            tongue_ratios.append(h / w)

    # Phase 3 – Simulated Stroke
    print("Calibration 3/3: Simulate droop…")
    while len(stroke_mouths) < samples:
        frame = picam2.capture_array()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        cv2.putText(bgr, f"Simulate {len(stroke_mouths)+1}/{samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow("Calibration", bgr)
        cv2.waitKey(int(pause * 1000))

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        if not valid_face(faces):
            continue
        lm = predictor(gray, faces[0])
        stroke_mouths.append(get_mouth_asymmetry(lm))
        le = get_eye_aspect_ratio(np.array([(p.x, p.y) for p in lm.parts()[36:42]]))
        re = get_eye_aspect_ratio(np.array([(p.x, p.y) for p in lm.parts()[42:48]]))
        stroke_eyes.append(abs(le - re))

    cv2.destroyWindow("Calibration")

    # Compute thresholds
    nr_med, nr_std = np.median(neutral_rolls), np.std(neutral_rolls)
    hm_med, hm_str = np.median(neutral_mouths), np.median(stroke_mouths)
    he_med, he_str = np.median(neutral_eyes), np.median(stroke_eyes)
    tg_med = np.median(tongue_ratios)

    raw_head_tilt = nr_med + STD_MULTIPLIER * nr_std
    raw_mouth = (hm_med + hm_str) / 2
    raw_eye = (he_med + he_str) / 2
    raw_tongue = (tg_med + hm_med) / 2

    HEAD_TILT_THRESHOLD = min(raw_head_tilt, MAX_HEAD_TILT) * SENSITIVITY_FACTOR
    MOUTH_ASYMMETRY_THRESHOLD = min(raw_mouth, MAX_MOUTH_ASYMMETRY) * SENSITIVITY_FACTOR
    EYE_ASYMMETRY_THRESHOLD = min(raw_eye, MAX_EYE_ASYMMETRY) * SENSITIVITY_FACTOR
    TONGUE_OUT_THRESHOLD = min(raw_tongue, MAX_TONGUE_OUT) * SENSITIVITY_FACTOR

    print("\n=== Calibration complete ===")
    print(f"HEAD_TILT_THRESHOLD = {HEAD_TILT_THRESHOLD:.2f}°")
    print(f"MOUTH_ASYMMETRY_THRESHOLD = {MOUTH_ASYMMETRY_THRESHOLD:.3f}")
    print(f"EYE_ASYMMETRY_THRESHOLD = {EYE_ASYMMETRY_THRESHOLD:.3f}")
    print(f"TONGUE_OUT_THRESHOLD = {TONGUE_OUT_THRESHOLD:.3f}")
    print(f"Yaw limit for asymmetry = {MAX_YAW_FOR_ASSESSMENT:.1f}°")
    print(f"Roll limit for asymmetry = {MAX_ROLL_FOR_ASSESSMENT:.1f}°")
    print("============================\n")

def estimate_head_pose(landmarks, frame_shape):
    h, w = frame_shape[:2]
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ]) / 4.5
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),
        (landmarks.part(8).x, landmarks.part(8).y),
        (landmarks.part(36).x, landmarks.part(36).y),
        (landmarks.part(45).x, landmarks.part(45).y),
        (landmarks.part(48).x, landmarks.part(48).y),
        (landmarks.part(54).x, landmarks.part(54).y),
    ], dtype="double")
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype="double")
    dist = np.zeros((4, 1))
    _, rvec, tvec = cv2.solvePnP(model_points, image_points, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    rot, _ = cv2.Rodrigues(rvec)
    ang, *_ = cv2.RQDecomp3x3(rot)
    return ang[1], ang[0], ang[2], rvec, tvec, cam

def draw_3d_axes(img, rvec, tvec, cam, origin):
    axis = np.float32([[50, 0, 0], [0, 50, 0], [0, 0, -50]]).reshape(-1, 3)
    pts, _ = cv2.projectPoints(axis, rvec, tvec, cam, np.zeros((4, 1)))
    pts = pts.reshape(-1, 2)
    o = (int(origin[0]), int(origin[1]))
    cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for i in range(3):
        p = (int(pts[i][0]), int(pts[i][1]))
        cv2.line(img, o, p, cols[i], 3)

def draw_3d_bbox(img, rvec, tvec, cam):
    box = np.float32([
        [-90, -130, 20], [90, -130, 20], [90, 110, 20], [-90, 110, 20],
        [-90, -130, 140], [90, -130, 140], [90, 110, 140], [-90, 110, 140]
    ])
    pts, _ = cv2.projectPoints(box, rvec, tvec, cam, np.zeros((4, 1)))
    pts = np.int32(pts).reshape(-1, 2)
    cv2.drawContours(img, [pts[4:]], -1, (0, 255, 255), 2)
    for i in range(4):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 4]), (255, 0, 255), 2)

def draw_info_panel(img, status, color, yaw, pitch, roll, mouth_asym, eye_asym):
    x, y = 10, 20
    font, s = cv2.FONT_HERSHEY_SIMPLEX, 0.6
    txt = "Stroke Detected" if status == "Asymmetry Detected" else "No Stroke"
    col = (0, 0, 255) if status == "Asymmetry Detected" else (0, 255, 0)
    cv2.putText(img, txt, (x, y), font, 0.7, col, 2)
    y += 30
    cv2.putText(img, f"Status: {status}", (x, y), font, s, color, 2)
    y += 25
    cv2.putText(img, f"Roll: {roll:.1f}", (x, y), font, s, (255, 255, 0), 1)
    y += 20
    cv2.putText(img, f"Yaw: {yaw:.1f}", (x, y), font, s, (255, 255, 0), 1)
    y += 20
    cv2.putText(img, f"Mouth Asym: {mouth_asym:.3f}", (x, y), font, s, (255, 255, 0), 1)
    y += 20
    cv2.putText(img, f"Eye Asym: {eye_asym:.3f}", (x, y), font, s, (255, 255, 0), 1)

def process_frame(frame, detector, predictor, buf):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    yaw = pitch = roll = mouth_asym = eye_asym = 0
    status, color = "No Face", (100, 100, 100)
    cur = "Normal"

    if faces:
        lm = predictor(gray, faces[0])
        le = get_eye_aspect_ratio(np.array([(p.x, p.y) for p in lm.parts()[36:42]]))
        re = get_eye_aspect_ratio(np.array([(p.x, p.y) for p in lm.parts()[42:48]]))
        eye_asym = abs(le - re)
        mouth_asym = get_mouth_asymmetry(lm)
        tongue = is_tongue_out(lm)
        yaw, pitch, roll, rvec, tvec, cam = estimate_head_pose(lm, frame.shape)

        # --- Asymmetry Check Logic ---
        asym = False
        if abs(yaw) <= MAX_YAW_FOR_ASSESSMENT and abs(roll) <= MAX_ROLL_FOR_ASSESSMENT:
            rotf = max(np.cos(np.radians(abs(yaw))), 0.7)
            mouth_thresh = MOUTH_ASYMMETRY_THRESHOLD / rotf
            eye_thresh = EYE_ASYMMETRY_THRESHOLD / rotf
            if mouth_asym > mouth_thresh or eye_asym > eye_thresh:
                asym = True

        # Determine status
        if asym:
            cur = "Asymmetry Detected"
        elif tongue:
            cur = "Tongue Out"
        elif abs(roll) > HEAD_TILT_THRESHOLD:
            cur = "Head Tilted"

        # Debounce
        buf.append(cur)
        if buf.count("Asymmetry Detected") >= STATUS_CONFIRMATION_THRESHOLD:
            status, color = "Asymmetry Detected", (0, 0, 255)
        elif buf.count("Head Tilted") >= STATUS_CONFIRMATION_THRESHOLD:
            status, color = "Head Tilted", (0, 255, 255)
        elif buf.count("Tongue Out") >= STATUS_CONFIRMATION_THRESHOLD:
            status, color = "Tongue Out", (0, 165, 255)
        else:
            status, color = "Normal", (0, 255, 0)

        # Draw
        draw_3d_axes(frame, rvec, tvec, cam, (lm.part(30).x, lm.part(30).y))
        draw_3d_bbox(frame, rvec, tvec, cam)
        for i in range(68):
            cv2.circle(frame, (lm.part(i).x, lm.part(i).y), 2, (0, 255, 0), -1)
        if cur == "Tongue Out":
            for i in range(60, 68):
                cv2.circle(frame, (lm.part(i).x, lm.part(i).y), 3, (255, 0, 0), -1)

    draw_info_panel(frame, status, color, yaw, pitch, roll, mouth_asym, eye_asym)
    return frame

class App:
    def __init__(self, title="Facial Asymmetry / Stroke Detection"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.withdraw()

        download_and_extract_shape_predictor()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        self.status_buffer = deque(maxlen=STATUS_BUFFER_SIZE)

        print("Initializing camera…")
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"size": (640, 480)}))
        self.picam2.start()
        time.sleep(1.0)

        calibrate_camera(self.picam2, self.detector, self.predictor)

        self.root.deiconify()
        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()

        self.delay = 15
        self.update()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def update(self):
        frame = self.picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        frame = cv2.flip(frame, 1)
        out = process_frame(frame, self.detector, self.predictor, self.status_buffer)
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.root.after(self.delay, self.update)

    def on_closing(self):
        print("Shutting down…")
        self.picam2.stop()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == '__main__':
    App()
