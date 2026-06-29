# DDDS v2.0 Dashboard Gallery

Welcome to the visual gallery of the **Driver Drowsiness Detection System (DDDS) v2.0**. This document details the responsive layouts, styling palettes, and interactive states of both application dashboards.

---

## 🖥️ Tkinter Desktop Dashboard Layout

The desktop application is built with a deep-navy glassmorphism layout optimized for 1200×700 resolutions.

| Left Side: Camera Canvas (500px) | Right Side: Stats Panel & Controls (700px) |
| :--- | :--- |
| **Webcam Canvas (480×400)**:<br>• Displays real-time webcam feed.<br>• MediaPipe Face Mesh mesh tesselations (tesselations: grey; contours: white).<br>• Dynamic BGR bounding border reflecting alert level.<br>• Pulsing yellow warning overlay when face is absent. | **Header Section**:<br>• `DDDS` title in bold cyan (#00D4FF).<br>• Subtitle "Driver Safety Monitor" in muted grey.<br><br>**Fatigue Score Ring**:<br>• Custom animated circle showing fatigue score (0–100).<br>• Large score text coloring matching the current alert level. |
| **Status Overlay Label**:<br>• Displays calibration progress: `CALIBRATING... 45/90`.<br>• Displays live camera performance: `CALIBRATED | FPS: 29`. | **Live Metrics Grid**:<br>• 2×2 layout cards showing EAR, MAR, Blinks, and elapsed session time.<br><br>**Controls**:<br>• **Start Session** (Cyan)<br>• **Stop & Save** (Gray)<br>• **Recalibrate** (Outline) |

---

## 🌐 Web Browser Dashboard (localhost:5000)

The web dashboard is fully responsive, styled with a modern dark-theme interface utilizing the Inter font family.

```
+-------------------------------------------------------------------------------+
|  DDDS  Driver Safety Monitor v2.0                                   ● NORMAL  |
+-------------------------------------------------------------------------------+
|  +----------------------------------+   +----------------------------------+  |
|  | [LIVE CAMERA FEED]       FPS: 30 |   | FATIGUE SCORE                    |  |
|  |                                  |   |                                  |  |
|  |           (Webcam Feed)          |   |       /=======\                  |  |
|  |                                  |   |      /   0     \  ALERT LEVEL    |  |
|  |     - MediaPipe Tesselations     |   |     |   /100    | (O) Green      |  |
|  |     - BGR Alert Level Border     |   |      \         /  ( ) Amber      |  |
|  |     - Center Warnings            |   |       \=======/   ( ) Red        |  |
|  |                                  |   +----------------------------------+  |
|  |                                  |   +----------------------------------+  |
|  |                                  |   | LIVE METRICS                     |  |
|  |                                  |   | +--------------+ +-------------+ |  |
|  |                                  |   | | EAR: 0.282   | | MAR: 0.082  | |  |
|  |                                  |   | +--------------+ +-------------+ |  |
|  |                                  |   | | BLINKS: 4    | | TIME: 01:45 | |  |
|  |                                  |   | +--------------+ +-------------+ |  |
|  +----------------------------------+   +----------------------------------+  |
|                                         | Calibrated (baseline: 0.285)     |  |
|                                         +----------------------------------+  |
|                                         | [Start Session] [Stop] [Recalib] |  |
|                                         +----------------------------------+  |
+-------------------------------------------------------------------------------+
```

---

## 🚨 Visual States & Border Colors

The dashboards change visually based on the calculated fatigue level:

```
[ GREEN STATE: Safe ]
┌──────────────────────────────────────┐
│  Thin Green Border (3px)             │
│  Badge: * NORMAL                     │
└──────────────────────────────────────┘

[ AMBER STATE: Warning ]
┌──────────────────────────────────────┐
│  Medium Orange Border (6px)          │
│  Badge: ! DROWSY                     │
└──────────────────────────────────────┘

[ RED STATE: Critical Alert ]
┌──────────────────────────────────────┐
│  Thick Red Border (10px)             │
│  Top 20px Red tint overlay strip     │
│  Badge: !! CRITICAL                  │
└──────────────────────────────────────┘
```
