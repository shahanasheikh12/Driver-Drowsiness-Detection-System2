# logger.py — CSV session logging for DDDS v2.0
# Phase 5 | DDDS v2.0 — Driver Drowsiness Detection System

import os
import pandas as pd
from datetime import datetime

CSV_COLUMNS = [
    "session_id", "timestamp", "ear", "mar",
    "fatigue_score", "alert_level", "blink_count", "elapsed_seconds",
]


class SessionLogger:
    """
    Buffers per-frame data (every 30th frame = ~1 row/sec) and flushes
    to a persistent CSV on save_session(). Prints a summary table to
    console and returns a summary dict for the UI popup.
    """

    def __init__(self) -> None:
        """
        Creates data/ directory and initialises the CSV with headers if
        it does not already exist. Prepares an empty row buffer.
        """
        self.log_dir   = "data"
        self.log_path  = os.path.join(self.log_dir, "sessions.csv")
        os.makedirs(self.log_dir, exist_ok=True)

        if not os.path.exists(self.log_path):
            pd.DataFrame(columns=CSV_COLUMNS).to_csv(self.log_path, index=False)
            print(f"[LOG] Created sessions CSV -> {self.log_path}")
        else:
            print(f"[LOG] Appending to existing CSV -> {self.log_path}")

        self.session_id:    str | None = None
        self.rows:          list[dict] = []
        self.frame_counter: int        = 0

    # ── Session lifecycle ─────────────────────────────────────────────

    def start_session(self) -> str:
        """
        Generates a unique session ID and resets the row buffer.
        Call this when the user clicks Start Session.
        Returns the new session_id string.
        """
        self.session_id    = datetime.now().strftime("S_%Y%m%d_%H%M%S")
        self.rows          = []
        self.frame_counter = 0
        print(f"[LOG] Session started: {self.session_id}")
        return self.session_id

    # ── Frame-level logging ───────────────────────────────────────────

    def log_frame(
        self,
        ear:     float,
        mar:     float,
        score:   float,
        level:   str,
        blinks:  int,
        elapsed: float,
    ) -> None:
        """
        Appends one row to the in-memory buffer every 30 frames (~1/sec).
        Skips logging if no session is active or if face was absent
        (caller should pass level="GREEN" with score=0 during no-face periods,
        but can simply not call this method for cleaner logs).
        """
        if not self.session_id:
            return

        self.frame_counter += 1
        if self.frame_counter % 30 != 0:
            return

        self.rows.append({
            "session_id":      self.session_id,
            "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ear":             round(ear, 4),
            "mar":             round(mar, 4),
            "fatigue_score":   round(score, 2),
            "alert_level":     level,
            "blink_count":     blinks,
            "elapsed_seconds": round(elapsed, 1),
        })

    # ── Save + summary ────────────────────────────────────────────────

    def save_session(self) -> dict:
        """
        Flushes the row buffer to CSV (append mode), computes a session
        summary, prints a formatted table to console, clears the buffer,
        and returns the summary dict for display in the UI popup.
        Returns {} if no data was logged.
        """
        if not self.rows:
            print("[LOG] No data to save for this session.")
            return {}

        df = pd.DataFrame(self.rows, columns=CSV_COLUMNS)
        df.to_csv(self.log_path, mode="a", header=False, index=False)

        # Summary calculations
        total_alerts = int((df["alert_level"] != "GREEN").sum())
        duration_sec = float(df["elapsed_seconds"].max())
        mins, secs   = divmod(int(duration_sec), 60)
        summary = {
            "session_id":   self.session_id,
            "duration":     f"{mins:02d}:{secs:02d}",
            "avg_fatigue":  round(float(df["fatigue_score"].mean()), 1),
            "max_fatigue":  round(float(df["fatigue_score"].max()), 1),
            "total_alerts": total_alerts,
            "rows_logged":  len(self.rows),
            "csv_path":     self.log_path,
        }

        # Console summary table
        print("\n" + "=" * 48)
        print(f"  SESSION SUMMARY  —  {self.session_id}")
        print("=" * 48)
        for k, v in summary.items():
            print(f"  {k:<16}: {v}")
        print("=" * 48 + "\n")

        self.rows = []
        return summary
