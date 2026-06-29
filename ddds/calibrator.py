# calibrator.py — Adaptive per-driver EAR baseline calibration
# Phase 2 | DDDS v2.0 — Driver Drowsiness Detection System

import numpy as np


class BaselineCalibrator:
    """
    Collects raw EAR readings for the first N frames and computes
    a personal baseline for the current driver session.

    90 frames at ~30 fps = ~3 seconds of open-eye sampling.
    After calibration the drowsy threshold = baseline * 0.75.
    """

    def __init__(self, calibration_duration_frames: int = 90) -> None:
        """
        Initialises the calibrator with an empty sample buffer.
        calibration_duration_frames: how many frames to collect before locking
        the baseline. Default 90 (~3 s at 30 fps).
        """
        self.frames_needed: int   = calibration_duration_frames
        self.ear_samples:   list  = []
        self.calibrated:    bool  = False
        self.baseline_ear:  float = 0.27   # safe default if camera is very poor

        print(f"[CALIB] Calibrator ready — collecting {self.frames_needed} frames")

    def update(self, ear: float) -> bool:
        """
        Feed one frame's EAR value into the calibrator.
        Appends the sample when not yet calibrated and buffer is not full.
        Once the buffer is full, computes the mean and locks the baseline.
        Returns True once calibration is complete, False while still sampling.
        """
        if self.calibrated:
            return True

        if len(self.ear_samples) < self.frames_needed:
            self.ear_samples.append(ear)

        if len(self.ear_samples) == self.frames_needed:
            self.baseline_ear = float(np.mean(self.ear_samples))
            self.calibrated   = True
            print(
                f"[CALIB] Calibration complete — "
                f"baseline EAR = {self.baseline_ear:.4f}  |  "
                f"threshold = {self.get_threshold():.4f}"
            )

        return self.calibrated

    def get_threshold(self) -> float:
        """
        Returns the EAR value below which the driver is considered drowsy.
        Uses 75% of the personal baseline EAR (tighter for glasses wearers:
        lower to 0.70 if EAR is chronically under-detected).
        """
        return self.baseline_ear * 0.75

    def get_status_text(self) -> str:
        """
        Returns a short human-readable string for the on-screen overlay.
        Shows collection progress while calibrating; shows result when done.
        """
        if not self.calibrated:
            return f"CALIBRATING... {len(self.ear_samples)}/{self.frames_needed}"
        return f"CALIBRATED (baseline: {self.baseline_ear:.3f})"
