# main.py — Phase 5: logging + FPS + error handling + WAV generation
# DDDS v2.0 — Driver Drowsiness Detection System

import os
import time
import threading
import numpy as np
import cv2
import mediapipe as mp
from scipy.io.wavfile import write as wav_write

from detector   import get_ear_mar, calculate_fatigue_score, track_blink
from calibrator import BaselineCalibrator
from alert      import AlertManager
from logger     import SessionLogger
from ui         import DDDSDashboard

# ── MediaPipe handles ────────────────────────────────────────────────
mp_face_mesh   = mp.solutions.face_mesh
mp_drawing     = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles


# ─────────────────────────────────────────────────────────────────────
# One-time asset generation
# ─────────────────────────────────────────────────────────────────────

def ensure_alert_wav() -> None:
    """
    Generates assets/alert.wav using a two-tone (880/660 Hz) sine wave
    if the file is missing. Uses only numpy + scipy — no external tools.
    """
    path = os.path.join(os.path.dirname(__file__), "assets", "alert.wav")
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sr, dur = 44100, 1.2
    t   = np.linspace(0, dur, int(sr * dur), endpoint=False)
    env = np.ones_like(t)
    fade = int(sr * 0.05)
    env[:fade], env[-fade:] = np.linspace(0, 1, fade), np.linspace(1, 0, fade)
    tone   = np.zeros_like(t)
    chunks = np.array_split(t, 6)
    offset = 0
    for i, chunk in enumerate(chunks):
        freq = 880 if i % 2 == 0 else 660
        tone[offset:offset + len(chunk)] = np.sin(2 * np.pi * freq * chunk)
        offset += len(chunk)
    wav_write(path, sr, (tone * env * 32767).astype(np.int16))
    print(f"[MAIN] alert.wav generated -> {path}")


# ─────────────────────────────────────────────────────────────────────
# Frame annotation helper (Phase 3 overlays, now + FPS top-right)
# ─────────────────────────────────────────────────────────────────────

def _annotate_frame(
    frame,
    ear: float, mar: float,
    fatigue: dict, alert_level: str,
    calibrator: BaselineCalibrator,
    alert_manager: AlertManager,
    blink_count: int, fps: float,
) -> None:
    """
    Annotates BGR frame in-place: calibration banner OR live metrics,
    border/badge overlays (Phase 3), and FPS counter top-right (Phase 5).
    """
    font  = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    lc    = alert_manager.get_level_color(alert_level)
    h, w  = frame.shape[:2]

    if not calibrator.calibrated:
        cv2.putText(frame, calibrator.get_status_text(),
                    (10, 35), font, 0.60, white, 2)
        cv2.putText(frame, "Keep eyes open normally",
                    (10, 58), font, 0.45, (180, 180, 180), 1)
    else:
        cv2.putText(frame, f"EAR   : {ear:.3f}",      (10, 50),  font, 0.55, white, 1)
        cv2.putText(frame, f"MAR   : {mar:.3f}",      (10, 72),  font, 0.55, white, 1)
        cv2.putText(frame, f"Blinks: {blink_count}",  (10, 94),  font, 0.55, white, 1)
        cv2.putText(frame, f"Score : {fatigue['score']:.0f}",
                    (10, 126), font, 0.80, lc, 2)
        # Border
        thick = {"GREEN": 3, "AMBER": 6, "RED": 10}.get(alert_level, 3)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), lc, thick)
        # RED tint strip
        if alert_level == "RED":
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (w, 20), (0, 0, 180), -1)
            cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
        # Status badge
        badge = {"GREEN": "* NORMAL", "AMBER": "! DROWSY",
                 "RED": "!! CRITICAL"}.get(alert_level, "NORMAL")
        (tw, _), _ = cv2.getTextSize(badge, font, 0.65, 2)
        cv2.putText(frame, badge, ((w - tw) // 2, 26), font, 0.65, lc, 2)
        # Sub-scores bottom-left
        cv2.putText(frame, f"ear={fatigue['ear_score']}  "
                    f"mar={fatigue['mar_score']}  "
                    f"blink={fatigue['blink_score']}",
                    (10, h - 12), font, 0.40, (130, 130, 130), 1)

    # FPS counter — top-right
    fps_txt = f"FPS:{fps:.0f}"
    (fw, _), _ = cv2.getTextSize(fps_txt, font, 0.50, 1)
    cv2.putText(frame, fps_txt, (w - fw - 8, 20), font, 0.50, (140, 140, 140), 1)


# ─────────────────────────────────────────────────────────────────────
# Background detection thread
# ─────────────────────────────────────────────────────────────────────

def run_detection(
    shared_state: dict,
    stop_event:   threading.Event,
    recal_event:  threading.Event,
) -> None:
    """
    Full detection pipeline in a background daemon thread.
    Phase 5 additions: FPS counter, SessionLogger integration,
    no-face pause (>3 s suspends logging), try/except error handling.
    """
    print("[MAIN] Detection thread starting...")
    try:
        _run_detection_inner(shared_state, stop_event, recal_event)
    except Exception as exc:
        msg = f"Detection error: {exc}"
        print(f"[MAIN] [ERR] {msg}")
        shared_state["calibration_status"] = msg
        shared_state["error"] = msg


def _run_detection_inner(
    shared_state: dict,
    stop_event:   threading.Event,
    recal_event:  threading.Event,
) -> None:
    """Inner loop — raises on error, caught by run_detection wrapper."""

    calibrator    = BaselineCalibrator(calibration_duration_frames=90)
    alert_manager = AlertManager()
    logger        = SessionLogger()
    blink_state   = {"eye_closed": False, "blink_count": 0}

    session_id    = logger.start_session()
    shared_state["session_id"] = session_id

    session_start  = time.time()
    frame_count    = 0
    no_face_since: float | None = None   # timestamp when face last lost

    # FPS tracking
    fps_prev  = time.time()
    fps       = 0.0

    ear            = mar = 0.0
    fatigue_result = {"score": 0.0, "ear_score": 0, "mar_score": 0, "blink_score": 0}
    alert_level    = "GREEN"

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.6, min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (index 0)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[MAIN] [OK] Webcam opened.")

    while not stop_event.is_set():

        # Recalibrate signal
        if recal_event.is_set():
            calibrator    = BaselineCalibrator(calibration_duration_frames=90)
            blink_state   = {"eye_closed": False, "blink_count": 0}
            fatigue_result = {"score": 0.0, "ear_score": 0, "mar_score": 0, "blink_score": 0}
            alert_level    = "GREEN"
            session_start  = time.time()
            no_face_since  = None
            recal_event.clear()
            print("[MAIN] Recalibration reset")

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frame_count += 1
        elapsed = time.time() - session_start

        # FPS calculation
        now      = time.time()
        fps      = 1.0 / max(now - fps_prev, 1e-6)
        fps_prev = now

        # MediaPipe inference
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)
        rgb.flags.writeable = True

        face_detected = bool(results.multi_face_landmarks)

        if face_detected:
            no_face_since = None
            face_lm = results.multi_face_landmarks[0]

            mp_drawing.draw_landmarks(
                image=frame, landmark_list=face_lm,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw_styles.get_default_face_mesh_tesselation_style(),
            )
            mp_drawing.draw_landmarks(
                image=frame, landmark_list=face_lm,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw_styles.get_default_face_mesh_contours_style(),
            )

            ear, mar = get_ear_mar(face_lm)
            calibrator.update(ear)

            if calibrator.calibrated:
                _, blink_state = track_blink(ear, calibrator.get_threshold(), blink_state)
                fatigue_result = calculate_fatigue_score(
                    ear=ear, mar=mar, baseline_ear=calibrator.baseline_ear,
                    blink_count=blink_state["blink_count"], session_seconds=elapsed,
                )
                alert_level = alert_manager.trigger_if_needed(
                    fatigue_result["score"], frame_count
                )
                # Log only when face is present + calibrated
                logger.log_frame(ear, mar, fatigue_result["score"],
                                 alert_level, blink_state["blink_count"], elapsed)

                if frame_count % 30 == 0:
                    print(f"[SCORE] t={elapsed:.1f}s  score={fatigue_result['score']}  "
                          f"level={alert_level}  fps={fps:.1f}  "
                          f"blinks={blink_state['blink_count']}")
        else:
            # No face detected
            if no_face_since is None:
                no_face_since = time.time()
            no_face_secs = time.time() - no_face_since

            # Pulsing yellow warning after >3 s without face
            h_f, w_f = frame.shape[:2]
            msg = "NO FACE DETECTED"
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
            pulse_alpha = 0.5 + 0.5 * abs(np.sin(time.time() * 3))
            color_y = (0, int(200 * pulse_alpha), int(220 * pulse_alpha))
            cv2.putText(frame, msg, ((w_f - tw) // 2, h_f // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color_y, 2)
            if no_face_secs > 3:
                cv2.putText(frame, f"({no_face_secs:.0f}s)",
                            ((w_f - tw) // 2 + 20, h_f // 2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_y, 1)

        # Annotate + push to shared_state
        _annotate_frame(frame, ear, mar, fatigue_result, alert_level,
                        calibrator, alert_manager, blink_state["blink_count"], fps)

        mins, secs = divmod(int(elapsed), 60)
        shared_state.update({
            "frame":              frame,
            "ear":                round(ear, 3),
            "mar":                round(mar, 3),
            "score":              fatigue_result["score"],
            "level":              alert_level,
            "blinks":             blink_state["blink_count"],
            "session_time":       f"{mins:02d}:{secs:02d}",
            "calibrated":         calibrator.calibrated,
            "calibration_status": calibrator.get_status_text(),
            "fps":                round(fps, 1),
            "face_detected":      face_detected,
        })

    # Save on stop
    summary = logger.save_session()
    shared_state["last_summary"] = summary
    face_mesh.close()
    cap.release()
    print(f"[MAIN] Detection stopped — {time.time()-session_start:.1f}s")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[MAIN] DDDS v2.0 - Phase 5 starting...")
    ensure_alert_wav()

    shared_state: dict = {
        "frame": None, "ear": 0.0, "mar": 0.0,
        "score": 0.0, "level": "GREEN",
        "blinks": 0, "session_time": "00:00",
        "calibrated": False,
        "calibration_status": "Press  Start  to  begin",
        "fps": 0.0, "face_detected": False,
        "last_summary": {}, "session_id": "",
        "error": "",
    }
    stop_event  = threading.Event()
    recal_event = threading.Event()

    dashboard = DDDSDashboard(
        shared_state = shared_state,
        stop_event   = stop_event,
        recal_event  = recal_event,
        detection_fn = run_detection,
    )
    dashboard.run()
