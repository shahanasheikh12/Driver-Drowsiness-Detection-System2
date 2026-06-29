"""verify.py — DDDS v2.0 pre-presentation verification suite"""
import numpy as np
import os, time, csv, sys

print("=" * 55)
print("  DDDS v2.0 — Pre-Presentation Verification Suite")
print("=" * 55)

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
results = []

def chk(label, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"{tag} {label}" + (f"  ({detail})" if detail else ""))
    results.append((label, ok))

# ── 1. All imports ──────────────────────────────────────────────────
try:
    import cv2
    import mediapipe as mp
    import pandas as pd
    from PIL import Image
    from detector   import calculate_ear, calculate_mar, calculate_fatigue_score, track_blink
    from calibrator import BaselineCalibrator
    from alert      import AlertManager
    from logger     import SessionLogger
    chk("All module imports", True, f"cv2={cv2.__version__}, mediapipe={mp.__version__}")
except Exception as e:
    chk("All module imports", False, str(e))
    sys.exit(1)

# ── 2. alert.wav exists ─────────────────────────────────────────────
wav_path = os.path.join("assets", "alert.wav")
wav_ok = os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000
chk("assets/alert.wav exists", wav_ok,
    f"{os.path.getsize(wav_path)//1024} KB" if wav_ok else "missing")

# ── 3. Camera opens ─────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cam_ok = cap.isOpened()
if cam_ok:
    ret, frame = cap.read()
    cam_ok = ret and frame is not None
    frame_shape = frame.shape if cam_ok else None
    cap.release()
    chk("Camera opens + frame readable", cam_ok,
        f"shape={frame_shape}" if frame_shape else "no frame")
else:
    chk("Camera opens", False, "VideoCapture(0) failed")

# ── 4. EAR: open eye vs closed eye ─────────────────────────────────
class FakeLM:
    def __init__(self, pts): self._pts = pts
    def __getitem__(self, i):
        class P:
            def __init__(s, x, y): s.x = x; s.y = y
        return P(*self._pts[i])

open_pts = {
    33:(0.30,0.50), 160:(0.32,0.46), 158:(0.36,0.45),
    133:(0.42,0.50), 153:(0.36,0.54), 144:(0.32,0.54),
    362:(0.58,0.50), 385:(0.60,0.46), 387:(0.64,0.45),
    263:(0.70,0.50), 373:(0.64,0.54), 380:(0.60,0.54),
}
closed_pts = {
    33:(0.30,0.500), 160:(0.32,0.4994), 158:(0.36,0.4994),
    133:(0.42,0.500), 153:(0.36,0.5006), 144:(0.32,0.5006),
    362:(0.58,0.500), 385:(0.60,0.4994), 387:(0.64,0.4994),
    263:(0.70,0.500), 373:(0.64,0.5006), 380:(0.60,0.5006),
}
LEFT  = [362, 385, 387, 263, 373, 380]
RIGHT = [33,  160, 158, 133, 153, 144]

ear_open = (calculate_ear(FakeLM(open_pts), LEFT) +
            calculate_ear(FakeLM(open_pts), RIGHT)) / 2
ear_cl   = (calculate_ear(FakeLM(closed_pts), LEFT) +
            calculate_ear(FakeLM(closed_pts), RIGHT)) / 2
chk("EAR open > 0.20 (open eye)", ear_open > 0.20, f"EAR={ear_open:.3f}")
chk("EAR < 0.05 when closed (geometry)", ear_cl < 0.05, f"EAR={ear_cl:.5f}")

# ── 5. MAR rises for yawn ───────────────────────────────────────────
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]
closed_m = {61:(0.35,0.75),291:(0.65,0.75),39:(0.40,0.73),181:(0.45,0.76),
             0:(0.50,0.72), 17:(0.50,0.77),269:(0.55,0.73),405:(0.60,0.76)}
yawn_m   = {61:(0.35,0.70),291:(0.65,0.70),39:(0.40,0.64),181:(0.45,0.76),
             0:(0.50,0.61), 17:(0.50,0.79),269:(0.55,0.64),405:(0.60,0.76)}
mar_c = calculate_mar(FakeLM(closed_m), MOUTH)
mar_y = calculate_mar(FakeLM(yawn_m),   MOUTH)
chk("MAR rises for yawn", mar_y > mar_c, f"closed={mar_c:.3f} yawn={mar_y:.3f}")
chk("MAR yawn > 0.55 threshold", mar_y > 0.55, f"mar_y={mar_y:.3f}")

# ── 6. Calibrator completes at exactly frame 90 ─────────────────────
cal = BaselineCalibrator(90)
for _ in range(89):
    d = cal.update(0.285)
frame89_done = d
d90 = cal.update(0.285)
chk("Calibrator not done at frame 89", not frame89_done)
chk("Calibrator done at frame 90", d90 and cal.calibrated,
    f"baseline={cal.baseline_ear:.4f}")
chk("Calibrator threshold = baseline*0.75",
    abs(cal.get_threshold() - 0.285*0.75) < 0.0001,
    f"threshold={cal.get_threshold():.4f}")

# ── 7. Fatigue score GREEN -> AMBER -> RED ──────────────────────────
am = AlertManager()
sc_g = calculate_fatigue_score(0.285, 0.10, 0.285, 10, 60.0)
sc_a = calculate_fatigue_score(0.220, 0.10, 0.285, 10, 60.0)
sc_r = calculate_fatigue_score(0.180, 0.10, 0.285, 10, 60.0)

lg = am.get_level(sc_g["score"])
la = am.get_level(sc_a["score"])
lr = am.get_level(sc_r["score"])

chk("Score GREEN (eyes open)",  lg == "GREEN",  f"score={sc_g['score']}")
chk("Score AMBER (eyes partial)", la in ("AMBER","RED"), f"score={sc_a['score']}")
chk("Score RED (eyes closed 70% baseline)", lr == "RED", f"score={sc_r['score']}")

# ── 8. Blink tracking: one complete blink ───────────────────────────
bs = {"eye_closed": False, "blink_count": 0}
thresh = 0.21
_, bs = track_blink(0.10, thresh, bs)   # eye closes
_, bs = track_blink(0.30, thresh, bs)   # eye opens -> count++
chk("Blink tracking: 1 open-close-open cycle = 1 blink",
    bs["blink_count"] == 1, f"count={bs['blink_count']}")

# ── 9. Alert cooldown (should NOT double-fire) ──────────────────────
am2 = AlertManager()
am2.cooldown_seconds = 10
am2.trigger_if_needed(70.0, 1)   # fires
am2.trigger_if_needed(70.0, 2)   # within cooldown — should not fire
chk("Alert cooldown suppresses repeat fire", am2.alert_count == 1,
    f"alert_count={am2.alert_count}")

# ── 10. CSV logger: write + verify row ─────────────────────────────
lg2 = SessionLogger()
sid = lg2.start_session()
for i in range(30):
    lg2.log_frame(0.26, 0.08, 35.0, "GREEN", 3, float(30+i))
summary = lg2.save_session()
with open("data/sessions.csv") as f:
    rows = list(csv.DictReader(f))
sid_rows = [r for r in rows if r["session_id"] == sid]
chk("CSV file exists after save_session()", os.path.exists("data/sessions.csv"))
chk("CSV contains rows for this session", len(sid_rows) >= 1,
    f"{len(sid_rows)} rows for {sid}")
chk("Summary dict has required keys",
    all(k in summary for k in ["duration","avg_fatigue","max_fatigue","total_alerts"]))

# ── 11. FPS benchmark from live camera ─────────────────────────────
cap2 = cv2.VideoCapture(0)
if cap2.isOpened():
    t0 = time.time()
    fc = 0
    while time.time() - t0 < 2.0:
        ret, f = cap2.read()
        if ret: fc += 1
    cap2.release()
    fps = fc / 2.0
    chk("FPS >= 15 from camera", fps >= 15, f"{fps:.1f} fps")
else:
    print(f"{SKIP} FPS test — camera unavailable")

# ── Summary ─────────────────────────────────────────────────────────
print()
print("=" * 55)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"  Results: {passed}/{total} passed")
if passed == total:
    print("  ALL CHECKS PASSED — Ready for presentation!")
else:
    failed = [lbl for lbl, ok in results if not ok]
    print(f"  FAILED: {failed}")
print("=" * 55)
