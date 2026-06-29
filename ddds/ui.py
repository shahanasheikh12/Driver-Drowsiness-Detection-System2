# ui.py — Dark-themed Tkinter dashboard for DDDS v2.0
# Phase 5 | DDDS v2.0 — Driver Drowsiness Detection System

import tkinter as tk
from tkinter import messagebox
import threading
from typing import Callable
from PIL import Image, ImageTk
import numpy as np

# ── Colour palette ────────────────────────────────────────────────────
BG_MAIN  = "#0A0E1A"
BG_CARD  = "#141824"
BG_PANEL = "#1A2035"
CYAN     = "#00D4FF"
GREEN    = "#22C55E"
AMBER    = "#F59E0B"
RED      = "#EF4444"
WHITE    = "#F0F4FF"
GRAY     = "#7A8AAA"
GRAY_DIM = "#2A3550"

LEVEL_HEX = {"GREEN": GREEN, "AMBER": AMBER, "RED": RED}
LEVEL_BG  = {"GREEN": "#0D2B0D", "AMBER": "#2B1E0A", "RED": "#2B0A0A"}


class DDDSDashboard:
    """
    1200×700 dark dashboard window.
    Left 500 px: live webcam canvas.
    Right panel: score arc, stats grid, alert circles, control buttons.
    UI refreshes every 100 ms via root.after(); detection runs in a background thread.
    """

    def __init__(
        self,
        shared_state: dict,
        stop_event:   threading.Event,
        recal_event:  threading.Event,
        detection_fn: Callable,
    ) -> None:
        """
        Builds the full window layout.
        detection_fn is the run_detection() function from main.py — passed in
        to avoid circular imports (main → ui → main).
        """
        self.state        = shared_state
        self.stop_event   = stop_event
        self.recal_event  = recal_event
        self.detection_fn = detection_fn
        self._running     = False
        self._thread: threading.Thread | None = None
        self._frame_ref   = None   # prevent GC of ImageTk image
        self._splash_chars = ["|", "/", "-", "\\"]  # spinner frames
        self._splash_idx   = 0

        self.root = tk.Tk()
        self.root.title("DDDS v2.0 — Driver Safety Monitor")
        self.root.configure(bg=BG_MAIN)
        self.root.geometry("1200x700")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_left()
        self._build_right()
        # Splash runs after layout so widgets exist
        self.root.after(50, self._show_splash)

    # ── Layout builders ───────────────────────────────────────────────

    def _build_left(self) -> None:
        """500-px left pane: camera title strip, webcam canvas, calib label."""
        pane = tk.Frame(self.root, bg=BG_MAIN, width=500)
        pane.pack(side=tk.LEFT, fill=tk.Y, padx=(16, 8), pady=16)
        pane.pack_propagate(False)

        tk.Label(pane, text="LIVE  CAMERA  FEED", bg=BG_MAIN, fg=GRAY,
                 font=("Consolas", 9, "bold")).pack(anchor="w", pady=(0, 6))

        self.cam_canvas = tk.Canvas(pane, width=480, height=400, bg="#000000",
                                    highlightthickness=1,
                                    highlightbackground=GRAY_DIM)
        self.cam_canvas.pack()

        self.calib_lbl = tk.Label(pane, text="Press  Start  to  begin",
                                  bg=BG_MAIN, fg=AMBER,
                                  font=("Consolas", 10), wraplength=460)
        self.calib_lbl.pack(pady=(12, 0))

    def _build_right(self) -> None:
        """Right stats panel: header → score → badge → grid → circles → buttons."""
        pane = tk.Frame(self.root, bg=BG_MAIN)
        pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 16), pady=16)
        self._header(pane)
        self._score_section(pane)
        self._badge_section(pane)
        self._stats_grid(pane)
        self._alert_circles(pane)
        self._buttons(pane)

    def _header(self, p: tk.Frame) -> None:
        """DDDS title in cyan + subtitle + divider."""
        tk.Label(p, text="DDDS", bg=BG_MAIN, fg=CYAN,
                 font=("Arial", 34, "bold")).pack(anchor="w")
        tk.Label(p, text="Driver Safety Monitor", bg=BG_MAIN, fg=GRAY,
                 font=("Arial", 11)).pack(anchor="w")
        tk.Frame(p, bg=GRAY_DIM, height=1).pack(fill=tk.X, pady=(8, 10))

    def _score_section(self, p: tk.Frame) -> None:
        """Score arc canvas (left) + large score number (right)."""
        tk.Label(p, text="FATIGUE  SCORE", bg=BG_MAIN, fg=GRAY,
                 font=("Consolas", 9, "bold")).pack(anchor="w")

        row = tk.Frame(p, bg=BG_MAIN)
        row.pack(fill=tk.X, pady=(4, 0))

        # Arc canvas
        self.arc_cv = tk.Canvas(row, width=130, height=130, bg=BG_MAIN,
                                highlightthickness=0)
        self.arc_cv.pack(side=tk.LEFT)
        # Background ring
        self.arc_cv.create_arc(8, 8, 122, 122, start=90, extent=-360,
                               style=tk.ARC, outline=GRAY_DIM, width=11)
        # Filled arc (score indicator)
        self._arc = self.arc_cv.create_arc(8, 8, 122, 122, start=90, extent=0,
                                           style=tk.ARC, outline=GREEN, width=11)

        # Score number
        num_frame = tk.Frame(row, bg=BG_MAIN)
        num_frame.pack(side=tk.LEFT, padx=14)
        self.score_lbl = tk.Label(num_frame, text="0", bg=BG_MAIN, fg=GREEN,
                                  font=("Arial", 72, "bold"))
        self.score_lbl.pack()
        tk.Label(num_frame, text="/ 100", bg=BG_MAIN, fg=GRAY,
                 font=("Arial", 12)).pack()

    def _badge_section(self, p: tk.Frame) -> None:
        """Pill-shaped coloured status badge."""
        f = tk.Frame(p, bg=BG_MAIN)
        f.pack(anchor="w", pady=(8, 4))
        self.badge_lbl = tk.Label(f, text="● NORMAL", bg=LEVEL_BG["GREEN"],
                                  fg=GREEN, font=("Arial", 12, "bold"),
                                  padx=20, pady=6)
        self.badge_lbl.pack()

    def _stats_grid(self, p: tk.Frame) -> None:
        """2×2 cards: EAR | MAR / BLINKS | SESSION TIME."""
        grid = tk.Frame(p, bg=BG_MAIN)
        grid.pack(fill=tk.X, pady=(6, 4))

        specs = [
            ("EAR",          "ear_v",   "0.000"),
            ("MAR",          "mar_v",   "0.000"),
            ("BLINKS",       "blink_v", "0"),
            ("SESSION TIME", "time_v",  "00:00"),
        ]
        for i, (label, attr, default) in enumerate(specs):
            card = tk.Frame(grid, bg=BG_CARD, padx=14, pady=10)
            card.grid(row=i // 2, column=i % 2, padx=5, pady=4, sticky="nsew")
            tk.Label(card, text=label, bg=BG_CARD, fg=GRAY,
                     font=("Consolas", 8, "bold")).pack(anchor="w")
            lbl = tk.Label(card, text=default, bg=BG_CARD, fg=WHITE,
                           font=("Consolas", 20, "bold"))
            lbl.pack(anchor="w")
            setattr(self, attr, lbl)

        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

    def _alert_circles(self, p: tk.Frame) -> None:
        """Three indicator dots — active level glows, others dim."""
        row = tk.Frame(p, bg=BG_MAIN)
        row.pack(anchor="w", pady=(4, 2))
        tk.Label(row, text="ALERT LEVEL", bg=BG_MAIN, fg=GRAY,
                 font=("Consolas", 8, "bold")).pack(side=tk.LEFT, padx=(0, 14))

        self._dot_cv = tk.Canvas(row, width=120, height=22,
                                 bg=BG_MAIN, highlightthickness=0)
        self._dot_cv.pack(side=tk.LEFT)

        self._dots: dict[str, int] = {}
        for name, color, cx in [("GREEN", GREEN, 12), ("AMBER", AMBER, 48), ("RED", RED, 84)]:
            did = self._dot_cv.create_oval(cx - 10, 1, cx + 10, 21,
                                           fill=GRAY_DIM, outline="")
            self._dots[name] = did

    def _buttons(self, p: tk.Frame) -> None:
        """Start Session | Stop & Save | Recalibrate."""
        tk.Frame(p, bg=GRAY_DIM, height=1).pack(fill=tk.X, pady=(8, 10))
        row = tk.Frame(p, bg=BG_MAIN)
        row.pack(anchor="w")

        btn_cfg = dict(font=("Arial", 11, "bold"), bd=0, cursor="hand2",
                       padx=16, pady=8)

        self.btn_start = tk.Button(row, text="▶  Start Session",
                                   bg=CYAN, fg="#000000",
                                   command=self._on_start, **btn_cfg)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_stop = tk.Button(row, text="⏹  Stop & Save",
                                  bg=BG_CARD, fg=GRAY,
                                  command=self._on_stop, **btn_cfg)
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_recal = tk.Button(row, text="⟳  Recalibrate",
                                   bg=BG_MAIN, fg=CYAN, bd=1,
                                   relief=tk.SOLID, cursor="hand2",
                                   font=("Arial", 11, "bold"),
                                   padx=16, pady=8,
                                   command=self._on_recalibrate)
        self.btn_recal.pack(side=tk.LEFT)

    # ── Splash screen (2-second animated overlay) ──────────────────

    def _show_splash(self) -> None:
        """
        Draws a full-window overlay frame containing the DDDS logo and an
        animated ASCII spinner. Removes itself after 2 seconds via root.after().
        """
        self._splash = tk.Frame(self.root, bg=BG_MAIN)
        self._splash.place(x=0, y=0, relwidth=1, relheight=1)

        tk.Label(self._splash, text="DDDS v2.0", bg=BG_MAIN, fg=CYAN,
                 font=("Arial", 52, "bold")).pack(expand=True, pady=(140, 4))
        tk.Label(self._splash, text="Driver Drowsiness Detection System",
                 bg=BG_MAIN, fg=GRAY, font=("Arial", 14)).pack()
        tk.Label(self._splash, text="Initializing camera...",
                 bg=BG_MAIN, fg=GRAY, font=("Consolas", 10)).pack(pady=(6, 0))

        self._spin_lbl = tk.Label(self._splash, text="|", bg=BG_MAIN, fg=CYAN,
                                  font=("Consolas", 22, "bold"))
        self._spin_lbl.pack(pady=(10, 0))

        # Start spinner ticks (every 120 ms)
        self._splash_tick_id = self.root.after(120, self._tick_splash)
        # Dismiss after 2 000 ms
        self.root.after(2000, self._dismiss_splash)

    def _tick_splash(self) -> None:
        """Advances the ASCII spinner one frame and reschedules itself."""
        self._splash_idx = (self._splash_idx + 1) % len(self._splash_chars)
        try:
            self._spin_lbl.config(text=self._splash_chars[self._splash_idx])
            self._splash_tick_id = self.root.after(120, self._tick_splash)
        except tk.TclError:
            pass   # splash already destroyed

    def _dismiss_splash(self) -> None:
        """Removes the splash overlay and starts the live UI refresh loop."""
        try:
            self.root.after_cancel(self._splash_tick_id)
            self._splash.destroy()
        except Exception:
            pass
        self.update_ui()   # begin 100-ms refresh now

    # ── Live UI refresh (100 ms) ──────────────────────────────────────

    def update_ui(self) -> None:
        """
        Reads shared_state written by the detection thread.
        Updates every widget: score arc, badge, stat cards, alert circles,
        calib label (with FPS + error), and webcam frame.
        Reschedules itself every 100 ms.
        """
        s     = self.state
        level = s.get("level", "GREEN")
        score = float(s.get("score", 0))
        color = LEVEL_HEX.get(level, GREEN)

        # Score number + arc
        self.score_lbl.config(text=str(int(score)), fg=color)
        self.arc_cv.itemconfig(self._arc,
                               extent=-(score / 100) * 360, outline=color)

        # Status badge
        badge_txt = {"GREEN": "● NORMAL", "AMBER": "⚠  DROWSY",
                     "RED": "!! CRITICAL"}.get(level, "● NORMAL")
        self.badge_lbl.config(text=badge_txt, fg=color,
                              bg=LEVEL_BG.get(level, LEVEL_BG["GREEN"]))

        # Stat cards
        self.ear_v.config(text=f"{s.get('ear', 0.0):.3f}")
        self.mar_v.config(text=f"{s.get('mar', 0.0):.3f}")
        self.blink_v.config(text=str(s.get("blinks", 0)))
        self.time_v.config(text=s.get("session_time", "00:00"))

        # Alert circles — active glows, others dim
        for name, dot_id in self._dots.items():
            self._dot_cv.itemconfig(dot_id,
                                    fill=LEVEL_HEX[name] if name == level else GRAY_DIM)

        # Calibration / status label (show error if present)
        error = s.get("error", "")
        if error:
            self.calib_lbl.config(text=f"ERROR: {error}", fg=RED)
        else:
            calib_txt = s.get("calibration_status", "")
            fps_txt   = f"  |  FPS: {s.get('fps', 0.0):.0f}" if self._running else ""
            self.calib_lbl.config(text=calib_txt + fps_txt, fg=AMBER)

        # Webcam frame
        frame = s.get("frame")
        if frame is not None:
            try:
                rgb  = frame[:, :, ::-1]   # BGR → RGB in-place view
                img  = Image.fromarray(rgb).resize((480, 400), Image.LANCZOS)
                self._frame_ref = ImageTk.PhotoImage(img)
                self.cam_canvas.create_image(0, 0, anchor=tk.NW,
                                             image=self._frame_ref)
            except Exception:
                pass

        self.root.after(100, self.update_ui)

    # ── Button callbacks ──────────────────────────────────────────────

    def _on_start(self) -> None:
        """Starts the detection background thread (no-op if already running)."""
        if self._running:
            return
        self.stop_event.clear()
        self._running  = True
        self._thread   = threading.Thread(
            target=self.detection_fn,
            args=(self.state, self.stop_event, self.recal_event),
            daemon=True,
        )
        self._thread.start()
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(fg=WHITE)
        print("[UI] Detection thread started")

    def _on_stop(self) -> None:
        """Signals the detection thread to stop cleanly, then shows summary popup."""
        if not self._running:
            return
        self.stop_event.set()
        self._running = False
        self.state["calibration_status"] = "Session stopped"
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(fg=GRAY)
        print("[UI] Detection thread stopped")
        # Give thread a moment to write last_summary, then show popup
        self.root.after(800, self._show_summary_popup)

    def _show_summary_popup(self) -> None:
        """
        Reads last_summary from shared_state (written by logger.save_session)
        and shows a messagebox with session statistics.
        Does nothing if no summary data is available.
        """
        summary = self.state.get("last_summary", {})
        if not summary:
            return
        msg = (
            f"Session Complete!\n\n"
            f"Session ID   : {summary.get('session_id', 'N/A')}\n"
            f"Duration     : {summary.get('duration', '00:00')}\n"
            f"Avg Fatigue  : {summary.get('avg_fatigue', 0)}\n"
            f"Max Fatigue  : {summary.get('max_fatigue', 0)}\n"
            f"Total Alerts : {summary.get('total_alerts', 0)}\n"
            f"Rows logged  : {summary.get('rows_logged', 0)}\n\n"
            f"Log saved to data/sessions.csv"
        )
        messagebox.showinfo("DDDS — Session Summary", msg)
        # Clear so popup doesn't re-show on next stop
        self.state["last_summary"] = {}

    def _on_recalibrate(self) -> None:
        """Sets recal_event so the detection thread resets its calibrator."""
        if self._running:
            self.recal_event.set()
            print("[UI] Recalibrate signal sent")

    def _on_close(self) -> None:
        """Safe shutdown: stop detection thread, then destroy the window."""
        self._on_stop()
        self.root.destroy()

    # ── Entry point ────────────────────────────────────────────────

    def run(self) -> None:
        """
        Enters Tkinter mainloop (blocks). The splash screen schedules
        update_ui() to start after 2 s, so do NOT call update_ui() here.
        """
        self.root.mainloop()
