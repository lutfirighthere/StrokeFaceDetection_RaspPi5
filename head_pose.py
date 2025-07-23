import numpy as np
import cv2

def get_head_pose(landmarks):
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
        (landmarks.part(8).x, landmarks.part(8).y),    # Chin
        (landmarks.part(36).x, landmarks.part(36).y),  # Left eye
        (landmarks.part(45).x, landmarks.part(45).y),  # Right eye
        (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth
        (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
    ])

    focal_length = 1
    center = (0, 0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    _, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    rmat, _ = cv2.Rodrigues(rotation_vector)
    yaw = np.arctan2(rmat[1][0], rmat[0][0]) * 180.0 / np.pi
    return yaw
