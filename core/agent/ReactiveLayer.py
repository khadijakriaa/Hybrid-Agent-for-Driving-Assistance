"""
Reactive Layer: Fast, rule-based emergency response
"""
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from config.settings import AGENT_CONFIG, AgentConfig
from core.agent.scenario_model_registry import ScenarioModelRegistry


@dataclass
class CriticalAlert:
    """Represents a critical alert from reactive layer"""
    type: str
    severity: str  # "warning", "critical", "emergency"
    message: str
    timestamp: float
    reaction_time_ms: float
    parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp,
            "reaction_time_ms": self.reaction_time_ms,
            "parameters": self.parameters
        }


class ReactiveLayer:
    """
    Ultra-fast reactive safety layer.
    Implements reflex-like responses to immediate dangers.
    """

    def __init__(self, config: AgentConfig = None, confidence_threshold: float = 0.75):
        self.config = config or AGENT_CONFIG
        self.confidence_threshold = confidence_threshold
        self.alert_history = []
        self.model_registry = ScenarioModelRegistry()
        self.loaded_scenario_models = self.model_registry.load_available_models()
        self.performance_stats = {
            "total_checks": 0,
            "alerts_generated": 0,
            "avg_response_time": 0.0,
            "max_response_time": 0.0
        }
        if self.loaded_scenario_models:
            print(f"✅ Reactive Layer initialized (models: {', '.join(self.loaded_scenario_models)})")
            print(f"   Confidence threshold: {self.confidence_threshold:.0%}")
        else:
            print("✅ Reactive Layer initialized (no saved scenario models found yet)")

    # ===== RULE-BASED SCENARIO DETECTION =====

    def _detect_scenario(self, vehicle_state: Dict) -> Optional[str]:
        """
        Detect which scenario using rules.
        Priority: SAFETY FIRST - danger_zone > overtaking > intersection
        """
        # Priority 1: Danger zone (safety critical - ALWAYS CHECK FIRST)
        if self._is_danger_zone_scenario(vehicle_state):
            return "danger_zone"

        # Priority 2: Overtaking
        if self._is_overtaking_scenario(vehicle_state):
            return "overtaking"

        # Priority 3: Intersection
        if self._is_intersection_scenario(vehicle_state):
            return "intersection"

        return None

    def _is_danger_zone_scenario(self, state: Dict) -> bool:
        """Detect danger_zone using dynamic + rule-based indicators"""
        # EMERGENCY FLAGS (immediate danger - highest priority)
        if state.get('hard_brake', 0) == 1 or state.get('hard_brake', 0) is True:
            return True
        if state.get('leader_stopped', 0) == 1 or state.get('leader_stopped', 0) is True:
            return True
        if state.get('danger_accel_toward_leader', 0) == 1:
            return True

        # TTC check (only if valid and not 999)
        ttc = state.get('TTC', float('inf'))
        if isinstance(ttc, (int, float)) and 0 < ttc < 2.5:
            return True

        # Critical leader gap with stopped leader
        leader_gap = state.get('LeaderGap', float('inf'))
        if isinstance(leader_gap, (int, float)) and 0 < leader_gap < 8.0:
            if state.get('leader_stopped', 0) == 1:
                return True

        # Emergency deceleration
        deceleration = state.get('Deceleration', 0)
        if isinstance(deceleration, (int, float)) and deceleration > 3.0:
            return True

        return False

    def _is_overtaking_scenario(self, state: Dict) -> bool:
        """
        Detect overtaking scenario.
        Conditions:
        - Ego faster than leader (RelativeSpeed > 3 m/s)
        - Safe TTC (> 2.5s)
        - NOT in danger zone
        """
        rel_speed = state.get('RelativeSpeed', 0)
        ttc = state.get('TTC', 999)
        speed = state.get('Speed', 0)

        # Minimum speed required for overtaking (can't overtake at very low speeds)
        if speed < 15:  # ~54 km/h minimum
            return False

        # Overtaking: faster than leader AND safe following distance
        if rel_speed > 3.0 and ttc > 2.5:
            return True

        # Alternative: Large speed differential with safe gap
        leader_speed = state.get('LeaderSpeed', 0)
        leader_gap = state.get('LeaderGap', float('inf'))
        if speed - leader_speed > 8.0 and leader_gap > 20:
            return True

        return False

    def _is_intersection_scenario(self, state: Dict) -> bool:
        """
        Detect intersection approach.
        Conditions:
        - Decelerating
        - Moderate speed
        - No leader (or far leader)
        """
        deceleration = state.get('Deceleration', 0)
        speed = state.get('Speed', 0)
        leader_gap = state.get('LeaderGap', float('inf'))

        # Intersection approach: slowing down from moderate speed
        if speed < 60 and deceleration > 0.5:
            return True

        # No leader (open road) and slowing down
        if leader_gap == -1 and deceleration > 0.3 and speed < 50:
            return True

        return False

    def check_immediate_danger(self, vehicle_state: Dict) -> Optional[CriticalAlert]:
        """
        Ultra-fast danger check (<5ms target).
        Returns CriticalAlert if danger detected, None otherwise.
        """
        start_time = time.perf_counter()
        self.performance_stats["total_checks"] += 1

        # Step 1: Rule-based danger zone check (FAST PATH - no ML)
        if self._is_danger_zone_scenario(vehicle_state):
            alert = CriticalAlert(
                type="danger_zone",
                severity="emergency",
                message="EMERGENCY! Brake immediately! Collision risk!",
                timestamp=time.time(),
                reaction_time_ms=0.0,
                parameters={"ttc": vehicle_state.get('TTC', 999)}
            )

            reaction_time = (time.perf_counter() - start_time) * 1000
            alert.reaction_time_ms = reaction_time

            self.alert_history.append(alert)
            self.performance_stats["alerts_generated"] += 1
            self._update_performance_stats(reaction_time)

            return alert

        # Step 2: Detect scenario for ML validation (if needed)
        detected_scenario = self._detect_scenario(vehicle_state)

        # Step 3: ML validation for non-emergency scenarios
        if detected_scenario and detected_scenario in self.loaded_scenario_models:
            ml_result = self.evaluate_scenario_model(detected_scenario, vehicle_state)
            if "error" not in ml_result:
                risk = float(ml_result.get("risk_score", 0.0))

                if risk >= self.confidence_threshold:
                    severity = "emergency" if risk >= 0.90 else ("critical" if risk >= 0.80 else "warning")

                    alert = CriticalAlert(
                        type=f"{detected_scenario}_risk",
                        severity=severity,
                        message=self._generate_alert_message(detected_scenario, risk),
                        timestamp=time.time(),
                        reaction_time_ms=0.0,
                        parameters=ml_result,
                    )

                    reaction_time = (time.perf_counter() - start_time) * 1000
                    alert.reaction_time_ms = reaction_time

                    self.alert_history.append(alert)
                    self.performance_stats["alerts_generated"] += 1
                    self._update_performance_stats(reaction_time)

                    return alert

        # No alert
        reaction_time = (time.perf_counter() - start_time) * 1000
        self._update_performance_stats(reaction_time)
        return None

    def _generate_alert_message(self, scenario: str, risk: float) -> str:
        """Generate appropriate alert message"""
        messages = {
            "danger_zone": f"CRITICAL: Danger zone detected (risk: {risk:.0%})",
            "overtaking": f"Warning: Overtaking risk detected (risk: {risk:.0%})",
            "intersection": f"Warning: Intersection hazard (risk: {risk:.0%})"
        }
        return messages.get(scenario, f"Alert: {scenario} (risk: {risk:.0%})")

    def _update_performance_stats(self, reaction_time: float):
        """Update performance statistics"""
        self.performance_stats["avg_response_time"] = (
            self.performance_stats["avg_response_time"] * 0.9 + reaction_time * 0.1
        )
        self.performance_stats["max_response_time"] = max(
            self.performance_stats["max_response_time"], reaction_time
        )

    def evaluate_scenario_model(self, scenario: str, vehicle_state: Dict) -> Dict[str, Any]:
        """Run saved model inference for a specific scenario."""
        if not self.loaded_scenario_models:
            return {"error": "no_models_loaded"}
        if scenario not in self.loaded_scenario_models:
            return {"error": f"model_not_loaded_for_{scenario}"}
        try:
            return self.model_registry.predict_risk(scenario, vehicle_state)
        except Exception as exc:
            return {"error": str(exc)}

    def evaluate_all_scenarios(self, vehicle_state: Dict) -> Dict[str, Dict[str, Any]]:
        """Run inference against all loaded scenario models."""
        results = {}
        for scenario in self.loaded_scenario_models:
            results[scenario] = self.evaluate_scenario_model(scenario, vehicle_state)
        return results

    def get_performance_report(self) -> Dict:
        """Get performance statistics report"""
        return {
            **self.performance_stats,
            "meets_sla": self.performance_stats.get("max_response_time", 0) < self.config.MAX_REACTIVE_TIME_MS,
        }

    def reset_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            "total_checks": 0,
            "alerts_generated": 0,
            "avg_response_time": 0.0,
            "max_response_time": 0.0
        }
        self.alert_history = []

    def set_confidence_threshold(self, threshold: float) -> None:
        """Adjust the confidence threshold for alert generation."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        self.confidence_threshold = threshold
        print(f"✅ Confidence threshold updated to {threshold:.0%}")