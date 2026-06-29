# PRD — Driver Drowsiness Detection System (DDDS) v2.0 Simple
## Antigravity Build Document

**Status:** Ready to Build  
**Stack:** Python + OpenCV + MediaPipe + TFLite + Tkinter/PyQt5  
**Complexity:** Beginner-Intermediate | Final Year Project  
**Estimated Build Time:** 2–3 weeks  

---

## What Are We Building?

A **Python desktop app** that:
1. Opens webcam feed in real time
2. Detects driver drowsiness using facial landmarks (eyes + mouth + head)
3. Shows a live dashboard with fatigue score
4. Plays a voice alert when driver is drowsy
5. Logs session data to a CSV file

That's it. No cloud. No fleet. No mobile. Simple, clean, impressive.

---

## AI/ML Features (Simple Versions)

| Feature | What it does | How simple |
|---------|-------------|-----------|
| EAR Detection | Detects eye closure ratio | 10 lines of math |
| MAR Detection | Detects yawn | 10 lines of math |
| Fatigue Score | Weighted formula of EAR + MAR + blinks | Simple formula, no training |
| Adaptive Baseline | Learns YOUR normal EAR in first 30s | Running average |
| Alert System | Audio + on-screen alert | playsound library |
| Session Logger | Saves fatigue data to CSV | pandas |

> **No model training needed.** All ML features use rule-based formulas on MediaPipe landmarks. This is still "AI/ML" because it uses computer vision + adaptive algorithms.

---

## Project Folder Structure

```
ddds/
├── main.py                  ← Entry point, webcam loop
├── detector.py              ← EAR, MAR, fatigue score logic
├── alert.py                 ← Audio + on-screen alerts
├── calibrator.py            ← Adaptive baseline calibration
├── logger.py                ← CSV session logging
├── ui.py                    ← Tkinter dashboard UI
├── assets/
│   ├── alert.mp3            ← Alarm sound
│   └── alert_hindi.mp3      ← Hindi voice alert
├── data/
│   └── sessions.csv         ← Auto-generated logs
├── requirements.txt
└── README.md
```

---

## requirements.txt

```
opencv-python==4.9.0.80
mediapipe==0.10.9
numpy==1.26.4
pandas==2.2.1
playsound==1.3.0
scipy==1.13.0
Pillow==10.3.0
```

---

## Core Logic (for Antigravity reference)

### EAR Formula
```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
- p1..p6 = 6 eye landmark points from MediaPipe
- Normal EAR ≈ 0.25–0.30
- Drowsy EAR < 0.20 (or below adaptive baseline × 0.75)
```

### MAR Formula
```
MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (2 * ||p1-p5||)
- p1..p8 = 8 mouth landmark points
- Yawn MAR > 0.6
```

### Fatigue Score (0–100)
```
fatigue_score = (
  (ear_score × 0.5) +
  (mar_score × 0.2) +
  (blink_rate_score × 0.2) +
  (head_pose_score × 0.1)
)
- Each sub-score normalized 0–100
- Thresholds: Green < 40, Amber 40–65, Red > 65
```

---

## Antigravity Phase Prompts

---

### PHASE 1 — Project Setup + Webcam + MediaPipe

```
Build a Python project called "DDDS" (Driver Drowsiness Detection System).

Setup:
- Create the folder structure: main.py, detector.py, alert.py, calibrator.py, logger.py, ui.py
- Create requirements.txt with: opencv-python, mediapipe, numpy, pandas, playsound, scipy, Pillow

In main.py:
- Open webcam using OpenCV (cap = cv2.VideoCapture(0))
- For each frame, use MediaPipe FaceMesh (max_num_faces=1, refine_landmarks=True)
- Draw the face mesh on the frame
- Show the live feed in a window titled "DDDS — Driver Monitor"
- Press Q to quit

In detector.py:
- Write a function get_ear(landmarks, eye_indices) that:
  - Takes MediaPipe face landmarks and a list of 6 landmark indices for one eye
  - Calculates Eye Aspect Ratio: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
  - Returns float EAR value
- Use MediaPipe left eye indices: [362, 385, 387, 263, 373, 380]
- Use MediaPipe right eye indices: [33, 160, 158, 133, 153, 144]
- Calculate average EAR of both eyes
- Write a function get_mar(landmarks, mouth_indices) that:
  - Calculates Mouth Aspect Ratio similarly
  - Use mouth indices: [61, 291, 39, 181, 0, 17, 269, 405]
  - Returns float MAR value
- Print EAR and MAR values to console for each frame
```

---

### PHASE 2 — Adaptive Calibration + Fatigue Score

```
In calibrator.py:
- Create a class BaselineCalibrator:
  - __init__: sets calibration_frames=90 (3 seconds at 30fps), ear_values=[], calibrated=False, baseline_ear=0.27
  - Method update(ear): appends ear to ear_values if len < calibration_frames
  - When len reaches calibration_frames: set baseline_ear = mean(ear_values), calibrated=True, print "Calibration complete. Baseline EAR: {value}"
  - Method get_threshold(): returns baseline_ear * 0.75
  - Method is_ready(): returns calibrated bool

In detector.py — add fatigue score calculation:
- Create function calculate_fatigue_score(ear, mar, baseline_ear, blink_count, session_seconds):
  - ear_score: if ear < baseline_ear*0.75 → 100, elif ear < baseline_ear*0.85 → 60, else → 0
  - mar_score: if mar > 0.6 → 80, elif mar > 0.5 → 40, else → 0
  - blink_score: blinks_per_min = (blink_count / session_seconds) * 60; if blinks_per_min < 8 → 70, elif blinks_per_min > 25 → 50, else → 0
  - fatigue_score = (ear_score * 0.6) + (mar_score * 0.25) + (blink_score * 0.15)
  - Return fatigue_score (float 0–100)

- Also add blink counter: 
  - Track when EAR drops below threshold and comes back up
  - Count as one blink
  - Return blink_count

In main.py:
- Create BaselineCalibrator instance
- Show "CALIBRATING... keep eyes open" text overlay for first 3 seconds
- After calibration: show EAR, MAR, fatigue_score values as text overlay on frame
```

---

### PHASE 3 — Alert System

```
In alert.py:
- Import playsound, threading
- Create class AlertManager:
  - __init__: sets alert_cooldown=5 (seconds), last_alert_time=0, alert_active=False
  - Method trigger_alert(level, fatigue_score):
    - Check cooldown (time.time() - last_alert_time > alert_cooldown)
    - If cooldown passed:
      - If level == "RED": play "assets/alert.mp3" in a new thread (so it doesn't block)
      - If level == "AMBER": print "⚠️ Drowsiness detected — fatigue score: {fatigue_score}"
      - Update last_alert_time = time.time()
  - Method get_alert_level(fatigue_score):
    - Return "GREEN" if fatigue_score < 40
    - Return "AMBER" if 40 <= fatigue_score < 65
    - Return "RED" if fatigue_score >= 65

In main.py — add alert logic:
- After calculating fatigue_score:
  - Call alert_manager.get_alert_level(fatigue_score) 
  - If level is AMBER or RED: call alert_manager.trigger_alert(level, fatigue_score)
  - Color the frame border based on level:
    - GREEN → draw green rectangle border on frame
    - AMBER → draw yellow border
    - RED → draw red border + flash effect (alternate red/white every 5 frames)
```

---

### PHASE 4 — Live Dashboard UI (Tkinter)

```
In ui.py:
- Create a Tkinter dashboard window (separate from the OpenCV window) with:
  - Title: "DDDS — Driver Safety Monitor"
  - Dark background: #0D1117
  - Left panel (40% width): Live webcam feed embedded using PIL ImageTk
  - Right panel (60% width): Stats display

  Right panel should show these live-updating labels:
  - "FATIGUE SCORE" → big number (0–100) with color: green/amber/red
  - "STATUS" → text: ALERT / DROWSY / NORMAL with matching color
  - "EAR" → current eye aspect ratio value
  - "BLINKS" → blink count this session
  - "SESSION TIME" → elapsed time HH:MM:SS
  - "ALERT LEVEL" indicator (colored circle: green/amber/red)

  Bottom bar:
  - "START SESSION" button → begins monitoring
  - "STOP & SAVE LOG" button → stops and saves CSV
  - "CALIBRATE" button → re-runs calibration

- Use a threading approach: OpenCV runs in background thread, updates a shared frame variable
- Tkinter mainloop updates the UI every 100ms using root.after(100, update_ui)
```

---

### PHASE 5 — Session Logger + Final Polish

```
In logger.py:
- Create class SessionLogger:
  - __init__: creates data/sessions.csv if not exists with columns:
    [timestamp, session_id, ear, mar, fatigue_score, alert_level, blink_count, elapsed_seconds]
  - Method log_frame(ear, mar, fatigue_score, alert_level, blink_count, elapsed_seconds):
    - Append row to CSV every 30 frames (1 second of data)
  - Method save_session_summary(session_id):
    - Read all rows for this session_id
    - Calculate: avg_fatigue, max_fatigue, total_alerts, total_time
    - Print summary to console
    - Return summary dict

In main.py — wire everything together:
- On "START SESSION" button: generate session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
- Log every second to CSV
- On "STOP & SAVE" button: call save_session_summary and show popup with stats

Final polish in main.py:
- Add FPS counter overlay (top-right)
- Add "FACE NOT DETECTED" warning when no face found
- Add startup splash screen for 2 seconds showing: "DDDS v2.0 — Starting up..."
- Ensure clean shutdown: release cap, close all windows on Q or window close
- Add try/except around entire main loop with error logging
```

---

## What to Demo at Final Year Presentation

1. **Live detection** — Show EAR dropping when eyes close → fatigue score rising → alert
2. **Calibration** — Show it learning YOUR baseline vs a default
3. **Night simulation** — Dim the lights, show it still detects (CLAHE preprocessing bonus)
4. **Session log CSV** — Open CSV, show real logged data
5. **AI angle** — Explain the adaptive baseline = personalized AI; fatigue score formula = ML feature engineering

---

## Bonus Features (Add if time permits)

- **Hindi TTS alert:** Use `gtts` to pre-generate `alert_hindi.mp3` with "Savdhan! Neend aa rahi hai."
- **Head pose detection:** Add pitch/yaw/roll from MediaPipe face mesh pose landmarks → if chin drops > 20° add to fatigue score
- **Matplotlib graph:** Show fatigue score over time as a live line chart in the dashboard

---

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| MediaPipe not detecting face | Ensure good lighting; lower min_detection_confidence to 0.5 |
| playsound not working on Linux | Use `pygame.mixer` instead |
| Tkinter + OpenCV thread crash | Always update UI from main thread using root.after() |
| EAR always low (glasses) | Adjust calibration multiplier from 0.75 to 0.70 |
| High CPU usage | Process every 2nd frame: `if frame_count % 2 == 0: run_detection()` |

---

*PRD Version: 2.0 Simple | For use with Google Antigravity (Claude Sonnet + Thinking mode)*
