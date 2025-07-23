##Owen Kim OpenCV mini project
## Pre-stroke prediction based on face detection.
 

import cv2 as cv
import numpy as np
import dlib
from scipy.stats import entropy


cap = cv.imread('stroke/4.jpg')


# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from dlib


# Function to compute KL divergence
def calculate_kl_divergence(upper, lower):
    #Compute KL divergence between upper and lower eyelid distributions.
    hist_upper, _ = np.histogram(upper[:, 0], bins=10, range=(0, 1), density=True)
    hist_lower, _ = np.histogram(lower[:, 0], bins=10, range=(0, 1), density=True)

    hist_upper += 1e-6  # Prevent division by zero
    hist_lower += 1e-6

    return entropy(hist_upper, hist_lower)

def calculate_eye_tilt(inner, outer):
    """ Compute eye tilt angle in degrees using arctan. """
    dx, dy = outer[0] - inner[0], outer[1] - inner[1]
    return np.degrees(np.arctan2(dy, dx))

def process_eye_landmarks(landmarks, img, left=True):
    """ Process eye landmarks, draw them, and compute symmetry metrics. """
    eye_pts = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(36, 42)]) if left else \
              np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(42, 48)])

    # Draw landmarks & connecting lines
    for i in range(5):
        cv.circle(img, tuple(eye_pts[i]), 2, (0, 255, 0), -1)
        cv.line(img, tuple(eye_pts[i]), tuple(eye_pts[i + 1]), (255, 0, 0), 1)
    cv.circle(img, tuple(eye_pts[-1]), 2, (0, 255, 0), -1)
    cv.line(img, tuple(eye_pts[-1]), tuple(eye_pts[0]), (255, 0, 0), 1)

    # Upper and lower eyelid points
    upper_lid = eye_pts[[1, 2]]
    lower_lid = eye_pts[[4, 5]]

    # Calculate KL divergence
    kl_score = calculate_kl_divergence(upper_lid, lower_lid)

    # Calculate tilt angle
    eye_tilt = calculate_eye_tilt(eye_pts[0], eye_pts[3])

    return kl_score, eye_tilt


def face_landmarks(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Draw landmarks

        # Landmarks for Mouth Score
        #A
        x1, y1 = landmarks.part(48).x, landmarks.part(48).y
        cv.circle(img, (x1, y1), 2, (0, 255, 0), -1)
        #B
        x2, y2 = landmarks.part(54).x, landmarks.part(54).y
        cv.circle(img, (x2, y2), 2, (0, 255, 0), -1)
        #C
        x3, y3 = landmarks.part(30).x, landmarks.part(30).y
        cv.circle(img, (x3, y3), 2, (0, 255, 0), -1)
        #D
        x4, y4 = landmarks.part(28).x, landmarks.part(28).y
        cv.circle(img, (x4, y4), 2, (0, 255, 0), -1)

        cv.line(img, (x1, y1), (x3, y3), (255, 0, 0), 1)
        cv.line(img, (x2, y2), (x3, y3), (255, 0, 0), 1)
        cv.line(img, (x1, y1), (x4, y4), (255, 0, 0), 1)
        cv.line(img, (x2, y2), (x4, y4), (255, 0, 0), 1)

        A = np.array([x1, y1])
        B = np.array([x2, y2])
        C = np.array([x3, y3])
        D = np.array([x4, y4])

        # angle DAC
        AD = D-A
        AC = C-A
        dot_product = np.dot(AD, AC)
        mag_AD = np.linalg.norm(AD)
        mag_AC = np.linalg.norm(AC)
        cos_angle1 = dot_product / (mag_AD * mag_AC)
        angle_radians1 = np.arccos(cos_angle1)
        angle_degrees_DAC = np.degrees(angle_radians1)

        # angle CBD
        BD = D-B
        BC = C-B
        dot_product = np.dot(BD, BC)
        mag_BD = np.linalg.norm(BD)
        mag_BC = np.linalg.norm(BC)
        cos_angle2 = dot_product / (mag_BD * mag_BC)
        angle_radians2 = np.arccos(cos_angle2)
        angle_degrees_CBD = np.degrees(angle_radians2)



        S_mouthangle = np.abs(angle_degrees_CBD-angle_degrees_DAC)
        print("mouth: ", S_mouthangle)


        

        kl_left, tilt_left = process_eye_landmarks(landmarks, img, left=True)
        kl_right, tilt_right = process_eye_landmarks(landmarks, img, left=False)
        
        eye_tilt_diff = abs(tilt_left - tilt_right)
        
        #print("KL Left", kl_left,  "KL Right", kl_right, "Tilt Left", tilt_left, "Tilt Right", tilt_right, "Tilt Difference", eye_tilt_diff)
        print("eye: ", eye_tilt_diff)


        #cv.circle(img, (270,200), 2, (0, 0, 255), 1)
        #cv.circle(img, (370,200), 2, (0, 0, 255), 1)
        #cv.line(img, (300, 300), (340, 300), (0, 0, 255), 2)
        
        if S_mouthangle<2 and eye_tilt_diff<10:
            cv.putText(img, "no stroke", (50, 50), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv.putText(img, "stroke", (50, 50), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        


frame = cv.flip(cap, 1)
face_landmarks(frame)

cv.imshow("Faces", frame)
cv.waitKey(0)
cv.destroyAllWindows()