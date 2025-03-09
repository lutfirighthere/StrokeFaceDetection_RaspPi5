Pre-Stroke Prediction Using Face Detection
Owen Kim - OpenCV Mini Project
This project uses OpenCV, dlib, and NumPy to detect facial landmarks and predict potential signs of a stroke based on facial asymmetry. The model specifically evaluates:

Eye Asymmetry - Compares the tilt and closure of both eyes.
Mouth Angle Deviation - Analyzes mouth angles to detect drooping.
The program uses facial landmark detection to evaluate eye and mouth symmetry and provides a simple "Stroke" or "No Stroke" prediction.

✅ How It Works
Eye Asymmetry Detection:

Detects the tilt angle of both eyes.
Measures the KL divergence (distribution difference) between the upper and lower eyelid.
Large differences may indicate facial droop, a common stroke symptom.
Mouth Angle Deviation:

Measures the angles formed by the mouth corners and nose bridge.
Significant deviation in angles may indicate facial asymmetry, often linked to strokes.
Stroke Prediction:

If mouth angle deviation is high and eye tilt difference is noticeable, the system flags a possible stroke.
Otherwise, it indicates no stroke.
