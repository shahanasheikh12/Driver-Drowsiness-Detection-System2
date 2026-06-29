# This script generates a simple 880Hz sine-wave alert tone (alert.wav)
# Run once: python generate_alert.py
# Requires: numpy, scipy

import numpy as np
from scipy.io.wavfile import write as wav_write
import os

def generate_alert_wav(path: str, freq: float = 880.0,
                        duration: float = 1.2, sample_rate: int = 44100) -> None:
    """Generates a sine-wave beep WAV file at the given frequency and duration."""
    t      = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Fade in / out envelope to avoid clicks
    env    = np.ones_like(t)
    fade   = int(sample_rate * 0.05)
    env[:fade]  = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    wave   = (np.sin(2 * np.pi * freq * t) * env * 32767).astype(np.int16)
    wav_write(path, sample_rate, wave)
    print(f"[SETUP] alert.wav generated → {path}")

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "assets", "alert.wav")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    generate_alert_wav(out)
