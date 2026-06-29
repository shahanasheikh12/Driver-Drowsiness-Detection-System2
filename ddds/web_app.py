"""
web_app.py — DDDS v2.0 Flask Web Dashboard (localhost)
Serves live MJPEG webcam stream + real-time stats via JSON polling.
Run: python web_app.py  →  http://localhost:5000
"""
import os, time, threading
import numpy as np
import cv2
from flask import Flask, Response, jsonify, render_template_string

from detector   import get_ear_mar, calculate_fatigue_score, track_blink
from calibrator import BaselineCalibrator
from alert      import AlertManager
from logger     import SessionLogger

# ── Flask app ─────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Shared state (detection thread writes, Flask routes read) ─────────
state: dict = {
    "frame":               None,
    "ear":                 0.0,
    "mar":                 0.0,
    "score":               0.0,
    "level":               "GREEN",
    "blinks":              0,
    "session_time":        "00:00",
    "fps":                 0.0,
    "calibrated":          False,
    "calibration_status":  "Press Start Session to begin",
    "face_detected":       False,
    "running":             False,
    "last_summary":        {},
    "error":               "",
}
_stop_event  = threading.Event()
_recal_event = threading.Event()
_lock        = threading.Lock()


# ── Detection thread ──────────────────────────────────────────────────

def _detection_loop() -> None:
    """Background detection thread writing to shared state dict."""
    import mediapipe as mp
    mp_fm   = mp.solutions.face_mesh
    mp_draw = mp.solutions.drawing_utils
    mp_ds   = mp.solutions.drawing_styles

    cap = None
    face_mesh = None

    try:
        calibrator    = BaselineCalibrator(90)
        alert_manager = AlertManager()
        logger        = SessionLogger()
        blink_state   = {"eye_closed": False, "blink_count": 0}
        logger.start_session()

        session_start  = time.time()
        frame_count    = 0
        fps_prev       = time.time()
        ear = mar      = 0.0
        fatigue_result = {"score": 0.0, "ear_score": 0, "mar_score": 0, "blink_score": 0}
        alert_level    = "GREEN"

        face_mesh = mp_fm.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.6, min_tracking_confidence=0.5,
        )
        
        # Try different camera indices in case 0 is locked
        for idx in [0, 1, 2]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"[WEB] Successfully opened camera index {idx}")
                break
            cap.release()
            cap = None

        if cap is None:
            with _lock:
                state["calibration_status"] = "ERROR: Cannot open camera"
                state["error"] = "Camera locked or not found"
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while not _stop_event.is_set():
            if _recal_event.is_set():
                calibrator    = BaselineCalibrator(90)
                blink_state   = {"eye_closed": False, "blink_count": 0}
                fatigue_result = {"score": 0.0, "ear_score": 0, "mar_score": 0, "blink_score": 0}
                alert_level   = "GREEN"
                session_start = time.time()
                _recal_event.clear()

            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame_count += 1
            elapsed = time.time() - session_start
            now = time.time()
            fps = 1.0 / max(now - fps_prev, 1e-6)
            fps_prev = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            face_detected = bool(results.multi_face_landmarks)

            if face_detected:
                fl = results.multi_face_landmarks[0]
                mp_draw.draw_landmarks(
                    image=frame, landmark_list=fl,
                    connections=mp_fm.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_ds.get_default_face_mesh_tesselation_style(),
                )
                mp_draw.draw_landmarks(
                    image=frame, landmark_list=fl,
                    connections=mp_fm.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_ds.get_default_face_mesh_contours_style(),
                )
                ear, mar = get_ear_mar(fl)
                calibrator.update(ear)
                if calibrator.calibrated:
                    _, blink_state = track_blink(ear, calibrator.get_threshold(), blink_state)
                    fatigue_result = calculate_fatigue_score(
                        ear=ear, mar=mar, baseline_ear=calibrator.baseline_ear,
                        blink_count=blink_state["blink_count"], session_seconds=elapsed,
                    )
                    alert_level = alert_manager.trigger_if_needed(fatigue_result["score"], frame_count)
                    logger.log_frame(ear, mar, fatigue_result["score"],
                                     alert_level, blink_state["blink_count"], elapsed)
            else:
                h, w = frame.shape[:2]
                msg = "NO FACE DETECTED"
                (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
                pulse = (0, int(200 * abs(np.sin(time.time()*3))), int(220 * abs(np.sin(time.time()*3))))
                cv2.putText(frame, msg, ((w-tw)//2, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.85, pulse, 2)

            # FPS overlay
            fps_txt = f"FPS:{fps:.0f}"
            (fw, _), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.putText(frame, fps_txt, (frame.shape[1]-fw-8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (140,140,140), 1)

            mins, secs = divmod(int(elapsed), 60)
            with _lock:
                state.update({
                    "frame":              frame,
                    "ear":                round(ear, 3),
                    "mar":                round(mar, 3),
                    "score":              fatigue_result["score"],
                    "level":              alert_level,
                    "blinks":             blink_state["blink_count"],
                    "session_time":       f"{mins:02d}:{secs:02d}",
                    "fps":                round(fps, 1),
                    "calibrated":         calibrator.calibrated,
                    "calibration_status": calibrator.get_status_text(),
                    "face_detected":      face_detected,
                    "error":              "",
                })

        summary = logger.save_session()
        with _lock:
            state["last_summary"] = summary

    except Exception as exc:
        print(f"[WEB] Exception in detection thread: {exc}")
        with _lock:
            state["error"] = str(exc)
            state["calibration_status"] = f"ERROR: {exc}"
    finally:
        with _lock:
            state["running"] = False
            if state["calibration_status"] == "Press Start Session to begin" or state["calibration_status"].startswith("CALIBRATING") or state["calibration_status"].startswith("CALIBRATED"):
                state["calibration_status"] = "Session stopped"
            state["frame"] = None
        if face_mesh is not None:
            try:
                face_mesh.close()
            except Exception:
                pass
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


# ── MJPEG stream ──────────────────────────────────────────────────────

def _gen_frames():
    """Yields MJPEG-encoded frames for /video_feed."""
    while True:
        with _lock:
            frame = state.get("frame")
        if frame is None:
            # Generate black placeholder frame when camera isn't running
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "CAMERA INACTIVE", (180, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (122, 138, 170), 2)
            ok, buf = cv2.imencode(".jpg", placeholder)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
            time.sleep(0.1)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
        time.sleep(0.033)   # ~30 fps ceiling


# ── Flask routes ──────────────────────────────────────────────────────

@app.route("/video_feed")
def video_feed():
    """MJPEG live stream endpoint."""
    return Response(_gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def api_stats():
    """JSON stats polled every second by the frontend."""
    with _lock:
        return jsonify({
            "ear":     state["ear"],
            "mar":     state["mar"],
            "score":   state["score"],
            "level":   state["level"],
            "blinks":  state["blinks"],
            "time":    state["session_time"],
            "fps":     state["fps"],
            "calib":   state["calibration_status"],
            "running": state["running"],
            "summary": state["last_summary"],
            "error":   state["error"],
        })


@app.route("/api/start")
def api_start():
    """Starts the detection thread."""
    with _lock:
        already_running = state["running"]
    
    if not already_running:
        _stop_event.clear()
        with _lock:
            state["running"] = True
            state["last_summary"] = {}
            state["error"] = ""
        t = threading.Thread(target=_detection_loop, daemon=True)
        t.start()
        return jsonify({"status": "started"})
    return jsonify({"status": "already_running"})


@app.route("/api/stop")
def api_stop():
    """Signals detection to stop."""
    _stop_event.set()
    return jsonify({"status": "stopping"})


@app.route("/api/recalibrate")
def api_recalibrate():
    """Requests a calibration reset."""
    _recal_event.set()
    return jsonify({"status": "recalibrating"})


# ── HTML dashboard (inline template) ─────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DDDS v2.0 — Driver Safety Monitor</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;700&display=swap');
  *{margin:0;padding:0;box-sizing:border-box}
  :root{
    --bg:#0A0E1A;--card:#141824;--panel:#1A2035;
    --cyan:#00D4FF;--green:#22C55E;--amber:#F59E0B;--red:#EF4444;
    --white:#F0F4FF;--gray:#7A8AAA;--dim:#2A3550;
  }
  body{background:var(--bg);color:var(--white);font-family:'Inter',sans-serif;min-height:100vh}
  /* ── Header ── */
  header{
    display:flex;align-items:center;justify-content:space-between;
    padding:16px 32px;border-bottom:1px solid var(--dim);
    background:rgba(10,14,26,0.95);backdrop-filter:blur(12px);
    position:sticky;top:0;z-index:100;
  }
  .logo{font-size:1.8rem;font-weight:900;color:var(--cyan);letter-spacing:-1px}
  .logo span{color:var(--white);font-weight:300;font-size:1rem;margin-left:10px}
  .pill{
    padding:6px 16px;border-radius:999px;font-size:.78rem;font-weight:700;
    letter-spacing:.05em;transition:all .3s;
  }
  .pill-green{background:rgba(34,197,94,.15);color:var(--green);border:1px solid rgba(34,197,94,.3)}
  .pill-amber{background:rgba(245,158,11,.15);color:var(--amber);border:1px solid rgba(245,158,11,.3)}
  .pill-red  {background:rgba(239,68,68,.15); color:var(--red);  border:1px solid rgba(239,68,68,.3);animation:pulse-red 1s ease-in-out infinite}
  @keyframes pulse-red{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.4)}50%{box-shadow:0 0 0 8px rgba(239,68,68,0)}}
  /* ── Layout ── */
  .main{display:grid;grid-template-columns:1fr 420px;gap:20px;padding:24px 32px;max-width:1400px;margin:0 auto}
  /* ── Camera ── */
  .cam-wrap{
    background:var(--card);border-radius:16px;overflow:hidden;
    border:1px solid var(--dim);position:relative;
  }
  .cam-wrap img{width:100%;display:block}
  .cam-label{
    position:absolute;top:12px;left:12px;
    background:rgba(10,14,26,.7);backdrop-filter:blur(4px);
    padding:4px 10px;border-radius:6px;font-size:.7rem;
    color:var(--gray);font-family:'JetBrains Mono',monospace;letter-spacing:.05em;
  }
  .fps-badge{
    position:absolute;top:12px;right:12px;
    background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.2);
    padding:4px 10px;border-radius:6px;font-size:.7rem;
    color:var(--cyan);font-family:'JetBrains Mono',monospace;
  }
  /* ── Stats panel ── */
  .stats{display:flex;flex-direction:column;gap:16px}
  .card{background:var(--card);border-radius:14px;padding:20px;border:1px solid var(--dim)}
  /* Score ring */
  .score-row{display:flex;align-items:center;gap:20px}
  .ring-wrap{position:relative;width:120px;height:120px;flex-shrink:0}
  svg.ring{width:120px;height:120px;transform:rotate(-90deg)}
  .ring-track{fill:none;stroke:var(--dim);stroke-width:10}
  .ring-arc  {fill:none;stroke:var(--green);stroke-width:10;stroke-linecap:round;
              stroke-dasharray:339.3;stroke-dashoffset:339.3;transition:stroke-dashoffset .6s ease,stroke .4s}
  .ring-num{
    position:absolute;inset:0;display:flex;flex-direction:column;
    align-items:center;justify-content:center;font-weight:900;
    font-size:2.4rem;line-height:1;color:var(--green);transition:color .4s;
  }
  .ring-num small{font-size:.7rem;color:var(--gray);font-weight:400}
  .score-meta{flex:1}
  .score-label{font-size:.65rem;font-family:'JetBrains Mono',monospace;color:var(--gray);
               letter-spacing:.12em;text-transform:uppercase;margin-bottom:6px}
  /* Grid cards */
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .mini-card{background:#0F1420;border-radius:10px;padding:12px 14px;border:1px solid var(--dim)}
  .mini-label{font-size:.62rem;font-family:'JetBrains Mono',monospace;color:var(--gray);
              letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px}
  .mini-val{font-size:1.35rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--white)}
  /* Circles */
  .circles-row{display:flex;align-items:center;gap:14px}
  .dot{width:22px;height:22px;border-radius:50%;background:var(--dim);transition:all .4s}
  .dot.active-GREEN{background:var(--green);box-shadow:0 0 12px rgba(34,197,94,.6)}
  .dot.active-AMBER{background:var(--amber);box-shadow:0 0 12px rgba(245,158,11,.6)}
  .dot.active-RED  {background:var(--red);  box-shadow:0 0 12px rgba(239,68,68,.6)}
  .dot-lbl{font-size:.62rem;font-family:'JetBrains Mono',monospace;color:var(--gray)}
  /* Buttons */
  .btn-row{display:flex;gap:10px;flex-wrap:wrap}
  .btn{
    padding:10px 20px;border-radius:9px;font-size:.85rem;font-weight:700;
    cursor:pointer;border:none;transition:all .2s;font-family:'Inter',sans-serif;
  }
  .btn-primary{background:var(--cyan);color:#000}
  .btn-primary:hover{filter:brightness(1.15);transform:translateY(-1px)}
  .btn-primary:disabled{opacity:.4;cursor:not-allowed;transform:none}
  .btn-secondary{background:var(--panel);color:var(--gray);border:1px solid var(--dim)}
  .btn-secondary:hover{border-color:var(--gray)}
  .btn-secondary:disabled{opacity:.4;cursor:not-allowed}
  .btn-outline{background:transparent;color:var(--cyan);border:1px solid var(--cyan)}
  .btn-outline:hover{background:rgba(0,212,255,.08)}
  /* Calib bar */
  .calib-bar{font-size:.75rem;font-family:'JetBrains Mono',monospace;color:var(--amber);
             padding:8px 14px;background:#0F1420;border-radius:8px;border:1px solid var(--dim)}
  /* Summary toast */
  #toast{
    display:none;position:fixed;bottom:30px;right:30px;
    background:var(--card);border:1px solid var(--green);border-radius:14px;
    padding:20px 24px;z-index:999;min-width:280px;
    box-shadow:0 8px 32px rgba(0,0,0,.6);
    animation:slide-up .35s ease;
  }
  @keyframes slide-up{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
  #toast h3{color:var(--green);margin-bottom:12px;font-size:1rem}
  #toast .row{display:flex;justify-content:space-between;padding:3px 0;font-size:.82rem;border-bottom:1px solid var(--dim)}
  #toast .row:last-of-type{border:none}
  #toast .lbl{color:var(--gray)}
  #toast .val{color:var(--white);font-weight:600;font-family:'JetBrains Mono',monospace}
  #toast-close{position:absolute;top:10px;right:14px;background:none;border:none;color:var(--gray);cursor:pointer;font-size:1.1rem}
  /* Responsive */
  @media(max-width:900px){.main{grid-template-columns:1fr}}
</style>
</head>
<body>

<header>
  <div class="logo">DDDS <span>Driver Safety Monitor v2.0</span></div>
  <div id="status-pill" class="pill pill-green">● NORMAL</div>
</header>

<div class="main">
  <!-- Camera -->
  <div class="cam-wrap">
    <div class="cam-label">LIVE CAMERA FEED</div>
    <div class="fps-badge" id="fps-badge">FPS: --</div>
    <img id="cam" src="/video_feed" alt="live feed" loading="lazy">
  </div>

  <!-- Stats panel -->
  <div class="stats">

    <!-- Score ring card -->
    <div class="card">
      <div class="score-label">FATIGUE SCORE</div>
      <div class="score-row">
        <div class="ring-wrap">
          <svg class="ring" viewBox="0 0 120 120">
            <circle class="ring-track" cx="60" cy="60" r="54"/>
            <circle class="ring-arc" id="ring-arc" cx="60" cy="60" r="54"/>
          </svg>
          <div class="ring-num" id="ring-num">0<small>/100</small></div>
        </div>
        <div class="score-meta">
          <div class="score-label" style="margin-bottom:10px">ALERT LEVEL</div>
          <div class="circles-row">
            <div class="dot" id="dot-GREEN"></div><div class="dot-lbl">GREEN</div>
            <div class="dot" id="dot-AMBER"></div><div class="dot-lbl">AMBER</div>
            <div class="dot" id="dot-RED"></div>  <div class="dot-lbl">RED</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Stats grid -->
    <div class="card">
      <div class="score-label" style="margin-bottom:12px">LIVE METRICS</div>
      <div class="grid">
        <div class="mini-card"><div class="mini-label">EAR</div><div class="mini-val" id="val-ear">0.000</div></div>
        <div class="mini-card"><div class="mini-label">MAR</div><div class="mini-val" id="val-mar">0.000</div></div>
        <div class="mini-card"><div class="mini-label">BLINKS</div><div class="mini-val" id="val-blinks">0</div></div>
        <div class="mini-card"><div class="mini-label">SESSION</div><div class="mini-val" id="val-time">00:00</div></div>
      </div>
    </div>

    <!-- Calibration status -->
    <div class="calib-bar" id="calib-bar">Press Start to begin</div>

    <!-- Buttons -->
    <div class="card">
      <div class="btn-row">
        <button class="btn btn-primary" id="btn-start" onclick="startSession()">▶  Start Session</button>
        <button class="btn btn-secondary" id="btn-stop" onclick="stopSession()" disabled>⏹  Stop &amp; Save</button>
        <button class="btn btn-outline" onclick="recalibrate()">⟳  Recalibrate</button>
      </div>
    </div>

  </div>
</div>

<!-- Session summary toast -->
<div id="toast">
  <button id="toast-close" onclick="document.getElementById('toast').style.display='none'">✕</button>
  <h3>Session Complete</h3>
  <div class="row"><span class="lbl">Duration</span><span class="val" id="t-dur">--</span></div>
  <div class="row"><span class="lbl">Avg Fatigue</span><span class="val" id="t-avg">--</span></div>
  <div class="row"><span class="lbl">Max Fatigue</span><span class="val" id="t-max">--</span></div>
  <div class="row"><span class="lbl">Total Alerts</span><span class="val" id="t-alerts">--</span></div>
  <div class="row"><span class="lbl">Log file</span><span class="val">data/sessions.csv</span></div>
</div>

<script>
const CIRC = 2 * Math.PI * 54;   // circumference of r=54 circle
let _wasRunning = false;

const COLORS = {GREEN:"#22C55E", AMBER:"#F59E0B", RED:"#EF4444"};

async function poll(){
  try{
    const r = await fetch("/api/stats");
    const d = await r.json();

    // Score ring
    const pct = Math.min(d.score/100, 1);
    const offset = CIRC * (1 - pct);
    const arc = document.getElementById("ring-arc");
    arc.style.strokeDashoffset = offset;
    arc.style.stroke = COLORS[d.level] || "#22C55E";

    const numEl = document.getElementById("ring-num");
    numEl.innerHTML = `${Math.round(d.score)}<small>/100</small>`;
    numEl.style.color = COLORS[d.level];

    // Status pill
    const pill = document.getElementById("status-pill");
    const LABELS = {GREEN:"● NORMAL", AMBER:"⚠ DROWSY", RED:"🚨 CRITICAL"};
    pill.textContent = LABELS[d.level] || "● NORMAL";
    pill.className = "pill pill-" + d.level;

    // Alert dots
    ["GREEN","AMBER","RED"].forEach(n=>{
      const el = document.getElementById("dot-"+n);
      el.className = "dot" + (n===d.level?" active-"+n:"");
    });

    // Metrics
    document.getElementById("val-ear").textContent   = d.ear.toFixed(3);
    document.getElementById("val-mar").textContent   = d.mar.toFixed(3);
    document.getElementById("val-blinks").textContent= d.blinks;
    document.getElementById("val-time").textContent  = d.time;
    
    if (d.running) {
      document.getElementById("fps-badge").textContent = "FPS: "+d.fps.toFixed(0);
      document.getElementById("btn-start").disabled = true;
      document.getElementById("btn-stop").disabled = false;
    } else {
      document.getElementById("fps-badge").textContent = "FPS: --";
      document.getElementById("btn-start").disabled = false;
      document.getElementById("btn-stop").disabled = true;
    }

    if (d.error) {
      document.getElementById("calib-bar").textContent = "ERROR: " + d.error;
      document.getElementById("calib-bar").style.color = COLORS.RED;
    } else {
      document.getElementById("calib-bar").textContent = d.calib;
      document.getElementById("calib-bar").style.color = COLORS.AMBER;
    }

    // Session ended → show summary toast
    if(_wasRunning && !d.running && d.summary && d.summary.duration){
      showToast(d.summary);
    }
    _wasRunning = d.running;

  }catch(e){ /* network error */ }
  setTimeout(poll, 500);
}

function showToast(s){
  document.getElementById("t-dur").textContent    = s.duration    || "--";
  document.getElementById("t-avg").textContent    = s.avg_fatigue ?? "--";
  document.getElementById("t-max").textContent    = s.max_fatigue ?? "--";
  document.getElementById("t-alerts").textContent = s.total_alerts ?? "--";
  document.getElementById("toast").style.display  = "block";
}

async function startSession(){
  await fetch("/api/start");
  document.getElementById("btn-start").disabled = true;
  document.getElementById("btn-stop").disabled = false;
}
async function stopSession(){
  await fetch("/api/stop");
  document.getElementById("btn-start").disabled = false;
  document.getElementById("btn-stop").disabled = true;
}
async function recalibrate(){ await fetch("/api/recalibrate"); }

poll();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    from main import ensure_alert_wav
    ensure_alert_wav()
    print("[WEB] DDDS v2.0 web dashboard starting...")
    print("[WEB] Open browser at  http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
