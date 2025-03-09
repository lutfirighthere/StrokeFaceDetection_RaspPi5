# 🚑 Pre-Stroke Prediction Using Face Detection
## **Owen Kim - OpenCV Mini Project**  

Stroke is the **third** leading cause of death globally.  
Early detection of **facial drooping** — a common symptom of a stroke — can significantly increase the chances of **survival and recovery**.  

By providing **real-time stroke prediction**, this project aims to showcase how **AI and computer vision** can contribute to **early medical intervention** and potentially **save lives**.  

This project uses **OpenCV**, **dlib**, and **NumPy** to detect facial landmarks and predict potential signs of a stroke based on **facial asymmetry**. The model specifically evaluates:  

- ✅ **Eye Asymmetry** - Compares the tilt angle and closure of both eyes.  
- ✅ **Mouth Angle Deviation** - Analyzes the deviation in mouth angles to detect drooping.  

The program leverages **facial landmark detection** to evaluate eye and mouth symmetry and provides a simple prediction of:  
- 🟢 **No Stroke** if the face appears symmetrical.  
- 🔴 **Stroke** if noticeable facial asymmetry is detected.  

---

## ✅ How It Works  

### **Eye Asymmetry Detection**  
- Detects the tilt angle of both eyes.  
- Measures the **KL divergence** (distribution difference) between the upper and lower eyelid.  
- Large differences in eye closure or tilt may indicate **facial droop**, a common stroke symptom.  

### **Mouth Angle Deviation**  
- Measures the angles formed by the mouth corners and the nose bridge.  
- A significant deviation in mouth angles could indicate **facial asymmetry**, often linked to strokes.  

### **Stroke Prediction**  
- If the mouth angle deviation is **high** and the eye tilt difference is **noticeable**, the system flags a possible stroke.  
- Otherwise, it classifies the face as **No Stroke**.  

---

## 💻 Run the Project  

### ✅ Option 1: With From photos in folders
### ✅ Option 2: With Webcam (Live Feed)  

---

## Drew Inspiration from this research paper
### Prehospital Cerebrovascular Accident Detection using Artificial Intelligence Powered Mobile Devices
https://www.researchgate.net/publication/346063581_Prehospital_Cerebrovascular_Accident_Detection_using_Artificial_Intelligence_Powered_Mobile_Devices

