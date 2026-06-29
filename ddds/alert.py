# alert.py — Multi-level alert manager with sound + visual feedback
# Phase 3 | DDDS v2.0 — Driver Drowsiness Detection System

import threading
import time
import os

# ── Optional playsound import (graceful fallback if not installed) ────
try:
    from playsound import playsound  # type: ignore
    PLAYSOUND_AVAILABLE = True
    print("[ALERT] playsound loaded successfully")
except ImportError:
    PLAYSOUND_AVAILABLE = False
    print("[ALERT] playsound not available - will use console beep fallback")


class AlertManager:
    """
    Manages AMBER / RED alert triggering with a cooldown so alarms
    don't fire on every single frame.

    Alert levels:
        GREEN  score < 40   — no action
        AMBER  40 <= s < 65 — console warning, increment count
        RED    score >= 65  — console warning + audio alarm
    """

    def __init__(self) -> None:
        """
        Sets up cooldown timer, alert counter, and initial level.
        cooldown_seconds: minimum gap between consecutive alert fires.
        """
        self.cooldown_seconds: float = 8.0
        self.last_alert_time:  float = 0.0
        self.current_level:    str   = "GREEN"
        self.alert_count:      int   = 0
        print("[ALERT] AlertManager initialised  (cooldown=8s)")

    # ── Level classification ──────────────────────────────────────────

    def get_level(self, score: float) -> str:
        """
        Maps a numeric fatigue score (0-100) to a named alert level.
        GREEN  < 40  : safe, no action needed
        AMBER  40-64 : mild drowsiness, visual + console warning
        RED   >= 65  : critical, audio alarm fires
        """
        if score >= 65:
            return "RED"
        if score >= 40:
            return "AMBER"
        return "GREEN"

    def get_level_color(self, level: str) -> tuple:
        """
        Returns the BGR colour tuple for a given alert level.
        Used by main.py to colour borders and badge text.
        GREEN -> (0, 200, 0)
        AMBER -> (0, 165, 255)
        RED   -> (0, 0, 220)
        """
        return {
            "GREEN": (0, 200,   0),
            "AMBER": (0, 165, 255),
            "RED":   (0,   0, 220),
        }.get(level, (255, 255, 255))

    # ── Sound playback ────────────────────────────────────────────────

    def _play_sound_async(self, filepath: str) -> None:
        """
        Plays an audio file in a background daemon thread so it never
        blocks the webcam loop. Falls back to a console BEEP print if
        playsound is unavailable or the file does not exist.
        """
        def _run() -> None:
            if PLAYSOUND_AVAILABLE and os.path.exists(filepath):
                try:
                    playsound(filepath, block=True)
                    return
                except Exception as exc:
                    print(f"[ALERT] playsound error: {exc}")
            # Fallback — winsound beep on Windows, else text
            try:
                import winsound
                winsound.Beep(1000, 800)
            except Exception:
                print("[ALERT] BEEP")  # last-resort text beep

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    # ── Main trigger ──────────────────────────────────────────────────

    def trigger_if_needed(self, score: float, frame_count: int) -> str:
        """
        Evaluates the current fatigue score and fires the appropriate alert
        if the cooldown has elapsed.

        GREEN  : resets active state silently, returns immediately.
        AMBER  : prints a console warning every cooldown_seconds.
        RED    : prints a critical warning AND plays the alarm sound.

        Returns the current level string ("GREEN" / "AMBER" / "RED").
        """
        level = self.get_level(score)
        self.current_level = level

        # No action needed for GREEN
        if level == "GREEN":
            return level

        # Cooldown check — suppress repeat alerts
        now = time.time()
        if (now - self.last_alert_time) < self.cooldown_seconds:
            return level

        # Cooldown passed — fire alert
        self.last_alert_time = now
        self.alert_count    += 1

        if level == "AMBER":
            print(f"[ALERT] WARNING  DROWSY ALERT  -- Score: {score:.1f}  "
                  f"(alert #{self.alert_count})")

        elif level == "RED":
            print(f"[ALERT] CRITICAL ALERT         -- Score: {score:.1f}  "
                  f"(alert #{self.alert_count})")
            # Play bundled WAV; path is relative to this file's directory
            sound_path = os.path.join(os.path.dirname(__file__), "assets", "alert.wav")
            self._play_sound_async(sound_path)

        return level
