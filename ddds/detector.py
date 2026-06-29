# detector.py — EAR / MAR calculations + fatigue scoring + blink tracking
# Phase 1 + 2 | DDDS v2.0 — Driver Drowsiness Detection System

import numpy as np

# ── MediaPipe landmark index constants ────────────────────────────────
LEFT_EYE_INDICES  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33,  160, 158, 133, 153, 144]
MOUTH_INDICES     = [61, 291, 39, 181, 0, 17, 269, 405]


def _euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    """Returns the Euclidean distance between two 2-D points (x, y only)."""
    return float(np.linalg.norm(p1 - p2))


def calculate_ear(landmarks, eye_indices: list[int]) -> float:
    """
    Computes the Eye Aspect Ratio (EAR) for one eye.

    Formula: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    where p1..p6 are the 6 landmark points of one eye in order:
      p1 = outer corner, p2 = upper-outer, p3 = upper-inner,
      p4 = inner corner, p5 = lower-inner, p6 = lower-outer.

    A normal open eye has EAR ≈ 0.25–0.30.
    A closed eye has EAR < 0.20.

    Args:
        landmarks: MediaPipe NormalizedLandmarkList (.landmark list)
        eye_indices: list of exactly 6 integer landmark indices

    Returns:
        float EAR value (higher = more open)
    """
    # Extract (x, y) coordinates for the 6 landmark points
    pts = np.array(
        [[landmarks[i].x, landmarks[i].y] for i in eye_indices],
        dtype=np.float32,
    )

    # Vertical distances
    v1 = _euclidean(pts[1], pts[5])   # ||p2 - p6||
    v2 = _euclidean(pts[2], pts[4])   # ||p3 - p5||

    # Horizontal distance
    h  = _euclidean(pts[0], pts[3])   # ||p1 - p4||

    ear = (v1 + v2) / (2.0 * h + 1e-6)   # avoid division by zero
    return float(ear)


def calculate_mar(landmarks, mouth_indices: list[int]) -> float:
    """
    Computes the Mouth Aspect Ratio (MAR) to detect yawning.

    Formula mirrors EAR but uses 8 mouth landmarks:
      p1 = left corner, p2 = upper-left, p3 = upper-mid-left,
      p4 = upper-mid-right, p5 = right corner, p6 = lower-mid-right,
      p7 = lower-mid-left, p8 = lower-left.

    MAR > 0.6 typically indicates a yawn.

    Args:
        landmarks: MediaPipe NormalizedLandmarkList (.landmark list)
        mouth_indices: list of exactly 8 integer landmark indices

    Returns:
        float MAR value (higher = more open mouth)
    """
    pts = np.array(
        [[landmarks[i].x, landmarks[i].y] for i in mouth_indices],
        dtype=np.float32,
    )

    # Three vertical distances across the mouth opening
    v1 = _euclidean(pts[1], pts[7])   # ||p2 - p8||
    v2 = _euclidean(pts[2], pts[6])   # ||p3 - p7||
    v3 = _euclidean(pts[3], pts[5])   # ||p4 - p6||

    # Horizontal distance (mouth width)
    h  = _euclidean(pts[0], pts[4])   # ||p1 - p5||

    mar = (v1 + v2 + v3) / (2.0 * h + 1e-6)
    return float(mar)


def get_ear_mar(face_landmarks) -> tuple[float, float]:
    """
    Convenience wrapper: computes average EAR across both eyes and MAR.

    Calls calculate_ear() for the left eye and right eye separately,
    then averages them to get a single robust EAR reading.
    Calls calculate_mar() for the mouth.

    Args:
        face_landmarks: single element from results.multi_face_landmarks
                        (i.e. face_landmarks.landmark is the landmark list)

    Returns:
        (avg_ear, mar) — both floats, both ≥ 0
    """
    lm = face_landmarks.landmark   # shorthand

    left_ear  = calculate_ear(lm, LEFT_EYE_INDICES)
    right_ear = calculate_ear(lm, RIGHT_EYE_INDICES)
    avg_ear   = (left_ear + right_ear) / 2.0

    mar = calculate_mar(lm, MOUTH_INDICES)
    return float(avg_ear), float(mar)


# ── Phase 2 additions ─────────────────────────────────────────────────

def calculate_fatigue_score(
    ear: float,
    mar: float,
    baseline_ear: float,
    blink_count: int,
    session_seconds: float,
) -> dict:
    """
    Combines EAR, MAR, and blink-rate sub-scores into a single weighted
    fatigue score (0–100) and returns a breakdown dict.

    Weights:
        EAR  60% — primary drowsiness signal
        MAR  25% — yawn detection
        Blink 15% — abnormal blink rate (too slow or too fast)

    Returns:
        dict with keys: score, ear_score, mar_score, blink_score
    """
    # ── EAR sub-score (how far below personal baseline) ───────────────
    if ear < baseline_ear * 0.70:
        ear_score = 100
    elif ear < baseline_ear * 0.80:
        ear_score = 70
    elif ear < baseline_ear * 0.90:
        ear_score = 40
    else:
        ear_score = 0

    # ── MAR sub-score (yawn detection) ───────────────────────────────
    if mar > 0.65:
        mar_score = 90
    elif mar > 0.55:
        mar_score = 50
    else:
        mar_score = 0

    # ── Blink-rate sub-score ─────────────────────────────────────────
    blinks_per_min = (blink_count / max(session_seconds, 1)) * 60
    if blinks_per_min < 8:       # too few blinks → possible micro-sleep
        blink_score = 70
    elif blinks_per_min > 30:    # rapid blinking → struggling to stay awake
        blink_score = 50
    else:
        blink_score = 0

    # ── Weighted total ────────────────────────────────────────────────
    final_score = (ear_score * 0.60) + (mar_score * 0.25) + (blink_score * 0.15)

    return {
        "score":       round(final_score, 1),
        "ear_score":   ear_score,
        "mar_score":   mar_score,
        "blink_score": blink_score,
    }


def track_blink(ear: float, threshold: float, state: dict) -> tuple[int, dict]:
    """
    Stateless blink counter: detects one complete eye-open → closed → open cycle.

    Call once per frame. state dict persists across calls in main.py:
        state = {"eye_closed": False, "blink_count": 0}

    Logic:
        - Eye goes below threshold  → mark as closed (start of blink)
        - Eye returns above threshold while was closed → count one blink

    Args:
        ear:       current frame EAR value
        threshold: drowsy threshold from calibrator.get_threshold()
        state:     mutable dict with keys 'eye_closed' and 'blink_count'

    Returns:
        (blink_count, updated_state)
    """
    if ear < threshold and not state["eye_closed"]:
        # Eye just closed — start tracking
        state["eye_closed"] = True

    elif ear >= threshold and state["eye_closed"]:
        # Eye just reopened — complete blink recorded
        state["eye_closed"]  = False
        state["blink_count"] += 1
        print(f"[BLINK] #{state['blink_count']} detected (EAR={ear:.3f})")

    return state["blink_count"], state
