"""Justice Dance analysis layer aligned with the AIHuMo feature conventions.

Input tables contain AIHuMo-style lm0_x ... lm32_z columns and timestamp_s.
Each player's table remains separate so a two-player session can be scored
independently against the same reference stream.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd

LANDMARKS = 33
ANGLE_TARGETS = {
    "left_elbow": (11, 13, 15), "right_elbow": (12, 14, 16),
    "left_shoulder": (13, 11, 23), "right_shoulder": (14, 12, 24),
    "left_hip": (11, 23, 25), "right_hip": (12, 24, 26),
    "left_knee": (23, 25, 27), "right_knee": (24, 26, 28),
}
DISTANCE_TARGETS = {
    "shoulders": (11, 12), "left_hand_right_hand": (15, 16),
    "left_foot_right_foot": (27, 28), "left_hip_left_knee": (23, 25),
    "right_hip_right_knee": (24, 26),
}


@dataclass(frozen=True)
class DanceScore:
    total: float
    landmark_error: float
    angle_error: float
    velocity_error: float
    frames_compared: int


def _points(df: pd.DataFrame, prefix: str = "lm") -> np.ndarray:
    return np.stack([
        df[[f"{prefix}{i}_{axis}" for axis in "xyz"]].to_numpy(float)
        for i in range(LANDMARKS)
    ], axis=1)


def normalize_landmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Center landmarks at the shoulder midpoint and scale by shoulder width."""
    result = df.copy()
    points = _points(result)
    center = (points[:, 11] + points[:, 12]) / 2.0
    scale = np.linalg.norm(points[:, 11] - points[:, 12], axis=1)
    valid = np.isfinite(center).all(1) & np.isfinite(scale) & (scale > 1e-8)
    normalized = np.full_like(points, np.nan)
    normalized[valid] = (points[valid] - center[valid, None, :]) / scale[valid, None, None]
    for i in range(LANDMARKS):
        for axis_index, axis in enumerate("xyz"):
            result[f"n{i}_{axis}"] = normalized[:, i, axis_index]
    return result


def _angles(points: np.ndarray) -> dict[str, np.ndarray]:
    values = {}
    for name, (a, b, c) in ANGLE_TARGETS.items():
        ba, bc = points[:, a] - points[:, b], points[:, c] - points[:, b]
        denom = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
        out = np.full(len(points), np.nan)
        good = np.isfinite(denom) & (denom > 1e-8)
        out[good] = np.degrees(np.arccos(np.clip(np.sum(ba[good] * bc[good], 1) / denom[good], -1, 1)))
        values[f"angle_{name}"] = out
    return values


def build_features(landmarks: pd.DataFrame) -> pd.DataFrame:
    """Build normalized XYZ, angles, distances, velocity, and acceleration."""
    df = normalize_landmarks(landmarks)
    points = _points(df, "n")
    additions = {**_angles(points)}
    for name, (a, b) in DISTANCE_TARGETS.items():
        additions[f"distance_{name}"] = np.linalg.norm(points[:, a] - points[:, b], axis=1)
    time = pd.to_numeric(df["timestamp_s"], errors="coerce").to_numpy(float)
    dt = np.diff(time, prepend=np.nan)
    for i in range(LANDMARKS):
        for axis in "xyz":
            col = f"n{i}_{axis}"
            delta = np.diff(df[col].to_numpy(float), prepend=np.nan)
            additions[f"n{i}_v{axis}"] = np.divide(delta, dt, out=np.full(len(df), np.nan), where=dt > 0)
    for i in range(LANDMARKS):
        speed_cols = [f"n{i}_v{axis}" for axis in "xyz"]
        additions[f"n{i}_speed"] = np.sqrt(sum(np.square(additions[c]) for c in speed_cols))
    for col in [c for c in additions if "_v" in c and c[-1] in "xyz"]:
        additions[col.replace("_v", "_a")] = np.divide(np.diff(additions[col], prepend=np.nan), dt, out=np.full(len(df), np.nan), where=dt > 0)
    df = pd.concat([df, pd.DataFrame(additions, index=df.index)], axis=1)
    return df


def _resample(df: pd.DataFrame, columns: list[str], count: int) -> np.ndarray:
    source = np.linspace(0, 1, len(df)); target = np.linspace(0, 1, count)
    return np.column_stack([np.interp(target, source, pd.to_numeric(df[c], errors="coerce").interpolate().bfill().ffill()) for c in columns])


def score(reference_features: pd.DataFrame, player_features: pd.DataFrame) -> DanceScore:
    """Compare one player's normalized feature stream to the reference."""
    landmark_cols = [f"n{i}_{axis}" for i in range(LANDMARKS) for axis in "xyz"]
    angle_cols = [f"angle_{name}" for name in ANGLE_TARGETS]
    velocity_cols = [c for c in reference_features.columns if c.endswith("_speed")]
    count = max(2, min(len(reference_features), len(player_features)))
    a, b = _resample(reference_features, landmark_cols, count), _resample(player_features, landmark_cols, count)
    angles_a, angles_b = _resample(reference_features, angle_cols, count), _resample(player_features, angle_cols, count)
    vel_a, vel_b = _resample(reference_features, velocity_cols, count), _resample(player_features, velocity_cols, count)
    landmark_error = float(np.nanmean(np.abs(a - b)))
    angle_error = float(np.nanmean(np.abs(angles_a - angles_b)) / 180.0)
    velocity_error = float(np.nanmean(np.abs(vel_a - vel_b)))
    combined = .55 * landmark_error + .30 * angle_error + .15 * velocity_error
    return DanceScore(float(np.clip(100 * np.exp(-combined), 0, 100)), landmark_error, angle_error, velocity_error, count)


def write_player_parquet(features: pd.DataFrame, output_path: str | Path, player_id: str) -> Path:
    """Write one independently identifiable player's feature dataset."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    features = features.copy()
    features.insert(0, "player_id", player_id)
    features.to_parquet(output, index=False)
    return output


def write_score(score_result: DanceScore, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.write_text(json.dumps(score_result.__dict__, indent=2), encoding="utf-8")
    return output
