from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import dlib
from scipy.stats import entropy

# ================= DEBUG CONFIG =================
DEBUG = True  # Set to False to disable all debug
MAX_PRINTS = 10
# ================================================

class DebugPrinter:
    def __init__(self, max_count):
        self.counts = {}
        self.max_count = max_count

    def print(self, key, *msgs):
        if not DEBUG:
            return
        cnt = self.counts.get(key, 0)
        if cnt < self.max_count:
            print(*msgs)
            self.counts[key] = cnt + 1

debug = DebugPrinter(MAX_PRINTS)

# Initialize camera
picam2 = Picamera2()
picam2.start()

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

frame_count = 0

def calculate_kl_divergence(upper, lower):
    debug.print("kl_div_args", "  [DEBUG] calculate_kl_divergence upper:", upper, "lower:", lower)
    hist_upper, _ = np.histogram(upper[:, 0], bins=10, range=(0, 1), density=True)
    hist_lower, _ = np.histogram(lower[:, 0], bins=10, range=(0, 1), density=True)
    hist_upper += 1e-6
    hist_lower += 1e-6
    kl = entropy(hist_upper, hist_lower)
    debug.print("kl_div_result", f"  [DEBUG] KL divergence = {kl:.6f}")
    return kl

def calculate_eye_tilt(inner, outer):
    dx, dy = outer[0] - inner[0], outer[1] - inner[1]
    tilt = np.degrees(np.arctan2(dy, dx))
    debug.print("eye_tilt", f"  [DEBUG] calculate_eye_tilt inner={inner}, outer={outer}, tilt={tilt:.2f}°")
    return tilt

def process_eye_landmarks(landmarks, img, left=True):
    side = "Left" if left else "Right"
    debug.print(f"{side}_eye_start", f"[DEBUG] process_eye_landmarks ({side} eye)")
    idx = range(36, 42) if left else range(42, 48)
    eye_pts = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in idx])
    debug.print(f"{side}_eye_pts", f"  eye_pts:", eye_pts.tolist())
    # draw
    for i in range(5):
        cv.circle(img, tuple(eye_pts[i]), 2, (0, 255, 0), -1)
        cv.line(img, tuple(eye_pts[i]), tuple(eye_pts[i+1]), (255, 0, 0), 1)
    cv.circle(img, tuple(eye_pts[-1]), 2, (0, 255, 0), -1)
    cv.line(img, tuple(eye_pts[-1]), tuple(eye_pts[0]), (255, 0, 0), 1)
    # metrics
    upper = eye_pts[[1, 2]]
    lower = eye_pts[[4, 5]]
    kl_score = calculate_kl_divergence(upper, lower)
    eye_tilt = calculate_eye_tilt(eye_pts[0], eye_pts[3])
    debug.print(f"{side}_eye_metrics", f"  [DEBUG] {side} eye KL={kl_score:.4f}, tilt={eye_tilt:.2f}°")
    return kl_score, eye_tilt

def estimate_head_pose(landmarks, img):
    debug.print("headpose_start", "[DEBUG] estimate_head_pose")
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),  # nose
        (landmarks.part(8).x,  landmarks.part(8).y),   # chin
        (landmarks.part(36).x, landmarks.part(36).y),  # left eye
        (landmarks.part(45).x, landmarks.part(45).y),  # right eye
        (landmarks.part(48).x, landmarks.part(48).y),  # left mouth
        (landmarks.part(54).x, landmarks.part(54).y)   # right mouth
    ], dtype="double")
    debug.print("headpose_imgpts", "  image_points:", image_points.tolist())

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
    ])
    debug.print("headpose_modelpts", "  model_points:", model_points.tolist())

    h, w = img.shape[:2]
    focal = w
    center = (w/2, h/2)
    camera_matrix = np.array([
        [focal, 0, center[0]],
        [0, focal, center[1]],
        [0, 0, 1]
    ], dtype="double")
    debug.print("headpose_cam_mtx", "  camera_matrix:\n", camera_matrix)

    dist = np.zeros((4,1))
    success, rot_vec, trans_vec = cv.solvePnP(
        model_points, image_points, camera_matrix, dist,
        flags=cv.SOLVEPNP_ITERATIVE
    )
    debug.print("headpose_solvepnpsuccess", f"  solvePnP success: {success}")

    rmat, _ = cv.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv.RQDecomp3x3(rmat)
    yaw, pitch, roll = angles[1], angles[0], angles[2]
    debug.print("headpose_angles", f"  [DEBUG] head pose — yaw={yaw:.2f}°, pitch={pitch:.2f}°, roll={roll:.2f}°")
    return yaw, pitch, roll, rot_vec, trans_vec, camera_matrix

def draw_axes(img, cm, rv, tv, origin):
    axis = np.float32([[50,0,0],[0,50,0],[0,0,50]])
    imgpts, _ = cv.projectPoints(axis, rv, tv, cm, np.zeros((4,1)))
    org = tuple(origin)
    for i, pt in enumerate(imgpts):
        p = tuple(int(x) for x in pt.ravel())
        col = (0,0,255) if i==0 else ((0,255,0) if i==1 else (255,0,0))
        cv.line(img, org, p, col, 2)

def face_landmarks(img):
    global frame_count
    frame_count += 1
    debug.print("frame_capture", f"[DEBUG] Captured frame #{frame_count}")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = detector(gray)
    debug.print("faces_detected", f"[DEBUG] Faces detected: {len(faces)}")
    for i, face in enumerate(faces, 1):
        debug.print("face_rect", f" [DEBUG] Face #{i} rect: {face}")
        landmarks = predictor(gray, face)
        debug.print("landmarks_extracted", " [DEBUG] Landmarks extracted")

        # mouth
        A = np.array([landmarks.part(48).x, landmarks.part(48).y])
        B = np.array([landmarks.part(54).x, landmarks.part(54).y])
        C = np.array([landmarks.part(30).x, landmarks.part(30).y])
        D = np.array([landmarks.part(28).x, landmarks.part(28).y])
        debug.print("mouth_points", f"  [DEBUG] mouth pts A={A}, B={B}, C={C}, D={D}")
        angle_DAC = np.degrees(np.arccos(np.dot(D-A, C-A) /
                             (np.linalg.norm(D-A)*np.linalg.norm(C-A))))
        angle_CBD = np.degrees(np.arccos(np.dot(D-B, C-B) /
                             (np.linalg.norm(D-B)*np.linalg.norm(C-B))))
        S_mouth = abs(angle_CBD - angle_DAC)
        debug.print("mouth_angle", f"  [DEBUG] mouth Δangle: {S_mouth:.2f}°")

        # eyes
        kl_l, tilt_l = process_eye_landmarks(landmarks, img, left=True)
        kl_r, tilt_r = process_eye_landmarks(landmarks, img, left=False)
        eye_diff = abs(tilt_l - tilt_r)
        debug.print("eye_diff", f"  [DEBUG] eye tilt diff: {eye_diff:.2f}°")

        # normalize
        face_w = abs(landmarks.part(16).x - landmarks.part(0).x)
        norm_m = S_mouth / face_w
        norm_e = eye_diff / face_w
        debug.print("normalize", f"  [DEBUG] normalized: mouth={norm_m:.4f}, eyes={norm_e:.4f}")

        # head pose
        yaw, pitch, roll, rv, tv, cm = estimate_head_pose(landmarks, img)
        nose = (landmarks.part(30).x, landmarks.part(30).y)
        draw_axes(img, cm, rv, tv, nose)

        # adaptive thresholds
        weight = max(0.0, 1 - abs(yaw)/20.0)
        th_m = 0.035 * weight
        th_e = 0.020 * weight
        debug.print("thresholds", f"  [DEBUG] angle weight={weight:.3f}, thresholds mouth={th_m:.4f}, eyes={th_e:.4f}")

        # decision
        if abs(yaw) > 25 or abs(pitch) > 20:
            result, col = "Face rotated", (0,255,255)
        elif norm_m < th_m and norm_e < th_e:
            result, col = "No stroke", (0,255,0)
        else:
            result, col = "Stroke detected", (0,0,255)
        debug.print("decision", f"  [DEBUG] Decision: {result}")
        cv.putText(img, result, (50, 110), cv.FONT_HERSHEY_COMPLEX, 0.6, col, 2)

try:
    while True:
        frame = picam2.capture_array()
        frame = cv.flip(frame, 1)
        face_landmarks(frame)
        cv.imshow("Stroke Detection (DEBUG)", frame)
        if cv.waitKey(1) & 0xFF == 27:
            break
except KeyboardInterrupt:
    print("Exiting on user interrupt.")
finally:
    cv.destroyAllWindows()
    picam2.stop()
