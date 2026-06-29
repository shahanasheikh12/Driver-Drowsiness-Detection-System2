# DDDS v2.0 — Driver Drowsiness Detection System

DDDS v2.0 is a professional real-time driver drowsiness detection application. It utilizes a webcam feed and MediaPipe's FaceMesh model (468 landmarks) to monitor driver facial geometry. By calculating Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and blink rates, it computes a driver fatigue score and sounds visual and audio alerts when critical levels are reached.

The project offers **two dashboard interfaces**:
1. A **desktop application** built using Python's native Tkinter library.
2. A **web dashboard** hosted on `localhost:5000` via a Flask server.

---

## Key Features

* **Real-time Landmark Tracking**: Utilizes MediaPipe FaceMesh to map eye contours and lips.
* **Adaptive Baseline Calibration**: Auto-calibrates to the driver's unique open-eye baseline in the first 90 frames (~3 seconds at 30 fps).
* **Multi-Weight Fatigue Model**:
  * **EAR (60%)** - Eye closure and micro-sleep detection.
  * **MAR (25%)** - Yawn opening detection.
  * **Blinks (15%)** - Abnormal blink-rate calculations (too slow or rapid struggling).
* **Multi-Level Alarm System**: Cooldown-managed audio beep alert and colored border feedback (Green / Amber / Red).
* **CSV Logging**: Tracks session states and exports metrics to `data/sessions.csv` every second.
* **Web UI (Localhost)**: Serves a live MJPEG stream overlayed with charts, status badges, and dynamic buttons.

---

## Installation & Setup

### Prerequisites
* Python 3.10 or 3.12+
* Web Camera (built-in or USB external camera)

### 1. Clone the Repository
```bash
git clone https://github.com/shaha/drawines.git
cd drawines/ddds
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Dashboards

### Option A: Desktop Tkinter Dashboard
To run the standard desktop dashboard window:
```bash
python main.py
```

### Option B: Localhost Web Dashboard
To host the Flask web dashboard on localhost:
```bash
python web_app.py
```
Open your browser and navigate to: **[http://localhost:5000](http://localhost:5000)**

---

## Project Structure
```
ddds/
├── assets/
│   └── alert.wav               # Alarm sound file
├── data/
│   └── sessions.csv            # Session statistics log file
├── alert.py                    # Alert level trigger manager
├── calibrator.py               # Personalized baseline calibrator
├── detector.py                 # Core EAR, MAR, and blink algorithms
├── logger.py                   # CSV writer and summary logs
├── main.py                     # Desktop runner & detection loop
├── ui.py                       # Tkinter dark UI widgets
├── verify.py                   # Automated tests suite
└── web_app.py                  # Flask web dashboard server
```
