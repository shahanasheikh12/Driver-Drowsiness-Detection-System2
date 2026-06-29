# Driver Drowsiness Detection System (DDDS)

## Project Title

**AI-Based Driver Drowsiness Detection System**

## Author

**Shahana Sheikh**

## Affiliation

**Computer Engineering**

## Date

**29/06/2026**

---

# Abstract

This project focuses on developing an **AI-Based Driver Drowsiness Detection System (DDDS)** that monitors a driver's facial features in real time to detect signs of fatigue and drowsiness. Driver drowsiness is one of the major causes of road accidents worldwide. The system uses **Computer Vision** and **Artificial Intelligence** techniques to analyze eye movements, blinking patterns, yawning, and head posture using a webcam.

The application utilizes **MediaPipe Face Mesh** to detect facial landmarks and calculates **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)** to determine the driver's alertness level. When signs of drowsiness are detected, the system immediately triggers visual and audio alerts to warn the driver. Session data is also stored for future analysis.

The results show that the system provides accurate real-time monitoring, reduces the risk of fatigue-related accidents, and enhances overall road safety.

---

# 1. Introduction

Road accidents caused by driver fatigue continue to increase every year. Long driving hours, lack of sleep, and continuous concentration often result in reduced alertness, leading to serious accidents.

The objective of this project is to develop an intelligent desktop application capable of detecting driver drowsiness using artificial intelligence and computer vision techniques. The system continuously monitors the driver's face through a webcam, calculates fatigue indicators, and generates instant alerts whenever drowsiness is detected.

The proposed system is simple, efficient, affordable, and suitable for real-world implementation without requiring expensive hardware.

---

# 2. Literature Review

* Traditional drowsiness detection systems require specialized sensors and expensive hardware.
* Camera-based monitoring systems provide a cost-effective solution.
* MediaPipe Face Mesh enables accurate facial landmark detection in real time.
* Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) are widely used techniques for fatigue detection.
* Adaptive thresholding improves detection accuracy for different users.

---

# 3. Methodology

The proposed system follows these steps:

* Capture live video using a webcam.
* Detect facial landmarks using MediaPipe Face Mesh.
* Calculate Eye Aspect Ratio (EAR).
* Calculate Mouth Aspect Ratio (MAR).
* Detect eye closure, blink rate, and yawning.
* Compute a fatigue score based on multiple parameters.
* Trigger audio and visual alerts if fatigue exceeds a threshold.
* Log session data into a CSV file for analysis.

The system also performs adaptive calibration to learn the user's normal eye-opening ratio, making the detection more personalized and accurate.

---

# 4. Implementation

### Programming Language

* Python

### Libraries

* OpenCV
* MediaPipe
* NumPy
* Pandas
* SciPy
* Pillow
* Playsound
* Tkinter

### Tools Used

* Visual Studio Code
* Git & GitHub
* Webcam
* CSV Logger

### Technologies

* Computer Vision
* Artificial Intelligence
* Facial Landmark Detection
* Adaptive Fatigue Analysis

---

# 5. Results and Discussion

The Driver Drowsiness Detection System successfully monitors the driver's facial movements in real time.

Major achievements include:

* Real-time eye and mouth tracking.
* Accurate drowsiness detection using EAR and MAR.
* Adaptive calibration for personalized monitoring.
* Instant audio and visual alerts.
* Automatic fatigue score calculation.
* Session data logging into CSV files.
* Professional monitoring dashboard.

Testing demonstrated that the system accurately detects prolonged eye closure and yawning under normal lighting conditions while maintaining smooth real-time performance.

---

# 6. Limitations

* Performance depends on lighting conditions.
* Accuracy may reduce if the driver's face is partially covered.
* Glasses or sunglasses may affect landmark detection.
* Webcam quality influences detection accuracy.
* Extreme head rotations can reduce tracking performance.

---

# 7. Future Scope

* Night vision enhancement.
* Head pose estimation using AI.
* Mobile application support.
* Cloud-based monitoring dashboard.
* SMS and emergency contact notifications.
* Multi-driver profile management.
* Deep Learning-based fatigue prediction.
* Integration with smart vehicles and IoT systems.

---

# 8. Conclusion

The AI-Based Driver Drowsiness Detection System provides an intelligent and practical solution for reducing road accidents caused by driver fatigue. By combining Computer Vision, Artificial Intelligence, and real-time facial landmark analysis, the system continuously monitors the driver's alertness and immediately provides warnings whenever drowsiness is detected.

The project demonstrates that affordable webcam-based monitoring can significantly improve road safety while maintaining high accuracy and real-time performance.

---

# References

[1] MediaPipe Face Mesh Documentation

[2] OpenCV Documentation

[3] Eye Aspect Ratio (EAR) Research Paper by Soukupová & Čech

[4] World Health Organization (WHO) – Road Safety Reports

[5] Python Official Documentation – https://www.python.org/

[6] OpenCV Official Documentation – https://opencv.org/

[7] MediaPipe Documentation – https://developers.google.com/mediapipe
