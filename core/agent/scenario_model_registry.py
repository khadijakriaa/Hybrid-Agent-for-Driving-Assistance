"""
Scenario model registry for training, saving, loading, and inference.
Each model predicts safety for one situation:
- intersection (target: SafeUnsafeCrossing)
- overtaking (target: SafeUnsafeOvertaking)
- danger_zone (target: SafeDangerZone)

Convention in your datasets:
- 0 = unsafe
- 1 = safe
This registry returns `risk_score` as P(unsafe) = P(class 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ScenarioTrainingConfig:
    name: str
    csv_path: Path
    target_col: str
    drop_cols: List[str]
    filter_col: str | None = None
    filter_value: Any = None


class ScenarioModelRegistry:
    def __init__(self, base_dir: str | Path | None = None):
        if base_dir is None:
            repo_root = Path(__file__).resolve().parents[2]
            base_dir = repo_root / "saved_models"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, Pipeline] = {}
        self.features: Dict[str, List[str]] = {}

    def training_plan(self, data_dir: str | Path | None = None) -> List[ScenarioTrainingConfig]:
        if data_dir is None:
            repo_root = Path(__file__).resolve().parents[2]
            data_dir = repo_root / "data"
        data_dir = Path(data_dir)

        return [
            ScenarioTrainingConfig(
                name="danger_zone",
                csv_path=data_dir / "dataset_apres_smote_DZ.csv",
                target_col="SafeDangerZone",
                drop_cols=[],
            ),
            ScenarioTrainingConfig(
                name="overtaking",
                csv_path=data_dir / "dataset_apres_smote_overtaking.csv",
                target_col="SafeUnsafeOvertaking",
                drop_cols=[],
            ),
            ScenarioTrainingConfig(
                name="intersection",
                csv_path=data_dir / "dataset_apres_smote_intersection.csv",
                target_col="SafeUnsafeCrossing",
                drop_cols=[],
            ),
        ]

    def train_and_save_all(self, data_dir: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
        reports: Dict[str, Dict[str, Any]] = {}
        for cfg in self.training_plan(data_dir):
            reports[cfg.name] = self._train_one(cfg)
        return reports

    def _train_one(self, cfg: ScenarioTrainingConfig) -> Dict[str, Any]:
        if not cfg.csv_path.exists():
            raise FileNotFoundError(f"Missing dataset: {cfg.csv_path}")

        df = pd.read_csv(cfg.csv_path)

        if cfg.filter_col is not None:
            df = df[df[cfg.filter_col] == cfg.filter_value].copy()

        if cfg.target_col not in df.columns:
            raise ValueError(f"Target column '{cfg.target_col}' not found in {cfg.csv_path.name}")

        y = pd.to_numeric(df[cfg.target_col], errors="coerce")
        keep = y.notna()
        y = y[keep].astype(int)
        X = df.loc[keep].drop(columns=[cfg.target_col], errors="ignore")

        if cfg.drop_cols:
            X = X.drop(columns=cfg.drop_cols, errors="ignore")

        # Keep numeric-only model inputs for robust runtime usage.
        X = X.select_dtypes(include=[np.number]).copy()

        if X.empty:
            raise ValueError(f"No numeric features left for model '{cfg.name}'")

        # Fill missing values with median to keep model robust.
        X = X.fillna(X.median(numeric_only=True)).fillna(0)

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(X, y)

        feature_names = list(X.columns)
        payload = {
            "name": cfg.name,
            "target": cfg.target_col,
            "feature_names": feature_names,
            "model": model,
            "label_convention": {"unsafe": 0, "safe": 1},
        }

        model_path = self.base_dir / f"{cfg.name}_model.joblib"
        joblib.dump(payload, model_path)

        self.models[cfg.name] = model
        self.features[cfg.name] = feature_names

        return {
            "model": cfg.name,
            "path": str(model_path),
            "samples": int(len(X)),
            "features": int(len(feature_names)),
        }

    def load_available_models(self) -> List[str]:
        loaded = []
        for model_file in self.base_dir.glob("*_model.joblib"):
            payload = joblib.load(model_file)
            name = payload["name"]
            self.models[name] = payload["model"]
            self.features[name] = payload["feature_names"]
            loaded.append(name)
        return loaded

    def predict_risk(self, scenario: str, vehicle_state: Dict[str, Any]) -> Dict[str, Any]:
        if scenario not in self.models:
            raise ValueError(f"Model '{scenario}' not loaded")

        feature_names = self.features[scenario]
        row = self._build_feature_row(feature_names, vehicle_state)
        X = pd.DataFrame([row], columns=feature_names)

        model = self.models[scenario]
        pred = int(model.predict(X)[0])
        probs = model.predict_proba(X)[0]
        classes = list(model.named_steps["rf"].classes_)

        unsafe_idx = classes.index(0) if 0 in classes else None
        safe_idx = classes.index(1) if 1 in classes else None
        risk_score = float(probs[unsafe_idx]) if unsafe_idx is not None else float(1.0 - probs[safe_idx])

        return {
            "scenario": scenario,
            "prediction": pred,
            "is_unsafe": pred == 0,
            "risk_score": risk_score,
            "proba": {str(c): float(p) for c, p in zip(classes, probs)},
        }

    def _build_feature_row(self, feature_names: List[str], state: Dict[str, Any]) -> Dict[str, float]:
        lower_state = {str(k).lower(): v for k, v in state.items()}

        def f(name: str, default: float = 0.0) -> float:
            if name in state:
                return self._to_float(state.get(name), default)
            if name.lower() in lower_state:
                return self._to_float(lower_state.get(name.lower()), default)
            return default

        lead = state.get("leading_vehicle") or {}
        lead_lower = {str(k).lower(): v for k, v in lead.items()}

        speed = f("Speed", f("speed", 0.0))
        leader_gap = f("LeaderGap", self._to_float(lead.get("distance", lead_lower.get("distance", np.inf)), np.inf))
        leader_speed = f("LeaderSpeed", self._to_float(lead.get("speed", lead_lower.get("speed", 0.0)), 0.0))
        accel = f("Acceleration", f("acceleration", 0.0))

        relative_speed = f("RelativeSpeed", speed - leader_speed)
        ttc = f("TTC", 999.0)
        if ttc == 999.0 and np.isfinite(leader_gap):
            rel_mps = max((relative_speed / 3.6), 0.0)
            ttc = 999.0 if rel_mps <= 0 else float(leader_gap / rel_mps)

        decel = f("Deceleration", -accel if accel < 0 else 0.0)
        safe_distance = f("SafeDistance", (speed / 10.0) * 3.0)

        base = {
            "Time": f("Time", f("time", 0.0)),
            "VehicleId": f("VehicleId", 0.0),
            "Pos_x": f("Pos_x", f("pos_x", 0.0)),
            "Pos_y": f("Pos_y", f("pos_y", 0.0)),
            "Speed": speed,
            "Lane": f("Lane", f("lane", 0.0)),
            "LeaderID": f("LeaderID", f("leader_id", -1.0)),
            "LeaderSpeed": leader_speed,
            "LeaderGap": leader_gap if np.isfinite(leader_gap) else 999.0,
            "Acceleration": accel,
            "Deceleration": decel,
            "RelativeSpeed": relative_speed,
            "TTC": ttc,
            "SafeDistance": safe_distance,
            "LanePrev": f("LanePrev", 999.0),
            "lane_opposite_change": f("lane_opposite_change", 0.0),
            "hard_brake": f("hard_brake", 0.0),
            "danger_accel_toward_leader": f("danger_accel_toward_leader", 0.0),
            "leader_stopped": f("leader_stopped", 0.0),
            "no_reaction_to_stopped_leader": f("no_reaction_to_stopped_leader", 0.0),
        }

        # Return only features expected by the selected model.
        return {name: self._to_float(base.get(name, 0.0), 0.0) for name in feature_names}

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default
