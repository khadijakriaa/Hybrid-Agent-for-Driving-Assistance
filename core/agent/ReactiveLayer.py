"""
Reactive Layer: Fast, rule-based emergency response
"""
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

from config.settings import AGENT_CONFIG, VEHICLE_CONFIG, AgentConfig
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
        Detect which scenario (intersection, overtaking, danger_zone) using rules.
        Priority: SAFETY FIRST - danger_zone > overtaking > intersection
        
        Returns: "danger_zone", "overtaking", "intersection", or None
        """
        # Priority 1: Check for danger_zone (SAFETY CRITICAL - check first!)
        # Hard brake, leader stopped, low TTC, close gap
        if self._is_danger_zone_scenario(vehicle_state):
            return "danger_zone"
        
        # Priority 2: Check for overtaking (lane change, speed differential)
        if self._is_overtaking_scenario(vehicle_state):
            return "overtaking"
        
        # Priority 3: Check for intersection (traffic signal, angle changes)
        if self._is_intersection_scenario(vehicle_state):
            return "intersection"
        
        # Fallback: Return most probable scenario if none detected clearly
        return self._fallback_scenario_detection(vehicle_state)

    def _is_danger_zone_scenario(self, state: Dict) -> bool:
        """
        Detect danger_zone using dynamic + rule-based indicators:
        - Hard brake flag (explicit emergency)
        - Leader stopped (explicit emergency)
        - Danger acceleration toward leader
        - No reaction to stopped leader
        - Low TTC (< 3.0 seconds = time to collision)
        - Very close gap (< safe distance * 0.5)
        - High deceleration (> 2.0 m/s²)
        
        Priority: Safety first! Be more sensitive to danger_zone
        """
        # ===== EXPLICIT EMERGENCY FLAGS =====
        
        # Hard brake detected
        if state.get('hard_brake', 0) == 1:
            return True
        
        # Leader stopped
        if state.get('leader_stopped', 0) == 1:
            return True
        
        # Danger acceleration toward leader
        if state.get('danger_accel_toward_leader', 0) == 1:
            return True
        
        # No reaction to stopped leader
        if state.get('no_reaction_to_stopped_leader', 0) == 1:
            return True
        
        # ===== DYNAMIC DANGER INDICATORS =====
        
        # TTC (Time To Collision) < 3.0 seconds = immediate danger
        ttc = state.get('TTC', float('inf'))
        if isinstance(ttc, (int, float)) and ttc < 3.0:
            return True
        
        # Very close gap (too close for safety)
        leader_gap = state.get('LeaderGap', float('inf'))
        if isinstance(leader_gap, (int, float)) and leader_gap < 5.0:
            # Less than 5m is dangerous
            leader_stopped = state.get('leader_stopped', 0)
            if leader_stopped == 1:
                return True
        
        # Safe distance check: gap < safe_distance * 0.5
        safe_distance = state.get('SafeDistance', None)
        if isinstance(safe_distance, (int, float)) and safe_distance > 0:
            if isinstance(leader_gap, (int, float)) and leader_gap < safe_distance * 0.5:
                # Less than 50% of safe distance = danger zone
                return True
        
        # High deceleration (emergency braking)
        deceleration = state.get('Deceleration', 0)
        if isinstance(deceleration, (int, float)) and deceleration > 2.0:
            # Deceleration > 2.0 m/s² = emergency braking
            return True
        
        # High negative acceleration (same as deceleration)
        acceleration = state.get('Acceleration', 0)
        if isinstance(acceleration, (int, float)) and acceleration < -2.0:
            return True
        
        return False

    def _is_overtaking_scenario(self, state: Dict) -> bool:
        """
        Detect overtaking using rules:
        - Lane change indicators
        - SafeDistance flags/metrics
        - Positive speed differential (ego faster than leader)
        - Lateral position changes
        - Lane opposite direction indicators
        """
        # Lane change in progress
        if state.get('lane_change_in_progress', 0) == 1:
            return True
        
        # Lane change maneuver
        if state.get('LaneChange', 0) == 1:
            return True
        
        # Lane opposite (attempting to change to opposite lane)
        if state.get('lane_opposite_change', 0) == 1:
            return True
        
        # Safe distance metric (indicates overtaking assessment)
        safe_distance = state.get('SafeDistance', None)
        if isinstance(safe_distance, (int, float)) and safe_distance >= 0:
            # Safe distance exists -> being evaluated for overtaking
            leader_gap = state.get('LeaderGap', float('inf'))
            if isinstance(leader_gap, (int, float)) and leader_gap > safe_distance:
                # Gap > safe distance -> could be overtaking
                ego_speed = state.get('speed', 0)
                leader_speed = state.get('LeaderSpeed', 0)
                if ego_speed > leader_speed + 5:  # Going faster than leader
                    return True
        
        # Speed differential (ego faster than leader by significant margin)
        ego_speed = state.get('speed', 0)
        leader_speed = state.get('LeaderSpeed', state.get('speed', 0))
        if ego_speed - leader_speed > 10:  # At least 10 km/h faster
            # Also check if gap is increasing (safe overtaking) or exists (preparing to overtake)
            leader_gap = state.get('LeaderGap', float('inf'))
            if isinstance(leader_gap, (int, float)) and leader_gap > 5:
                return True
        
        # Relative position changing (lateral position change)
        if state.get('relative_x_position', None) is not None:
            lateral_change = abs(state.get('relative_x_position', 0))
            if lateral_change > 1.5:  # Significant lateral movement
                return True
        
        return False

    def _is_intersection_scenario(self, state: Dict) -> bool:
        """
        Detect intersection using specific rules:
        - Direct intersection flag (from dataset)
        - Traffic signal/light presence (explicit)
        - Scenario column == "intersection" (if available)
        - Multiple vehicles at different angles (high confidence)

        NOTE: Lane == 0 and Angle != 0 removed (too permissive)
              These create false positives on highways
        """
        # Direct intersection flag (most reliable)
        if state.get('Intersection', 0) == 1:
            return True

        # Scenario label from dataset (if available)
        if state.get('Scenario', None) == 'intersection':
            return True
        
        # Traffic signal/light presence (explicit indicator)
        if state.get('traffic_signal', 0) == 1:
            return True
        
        if state.get('traffic_light', 0) == 1:
            return True

        # Multiple vehicles in area (potential intersection)
        # BUT: Require strong evidence (> 3 vehicles)
        num_vehicles = state.get('num_vehicles_nearby', 0)
        if num_vehicles > 3:
            # Multiple vehicles AND no clear leader = likely intersection
            leader_gap = state.get('LeaderGap', float('inf'))
            if (isinstance(leader_gap, (int, float)) and leader_gap == float('inf')) or leader_gap > 100:
                return True

        return False

    def _fallback_scenario_detection(self, state: Dict) -> Optional[str]:
        """
        Fallback scenario detection based on available features.
        Used when no clear scenario is detected.
        
        Priority: danger_zone > overtaking > intersection
        """
        # ===== CHECK FOR DANGER_ZONE FIRST =====
        # Even if explicit flags missed, check dynamic danger
        ttc = state.get('TTC', float('inf'))
        leader_gap = state.get('LeaderGap', float('inf'))
        
        # Low TTC = danger zone
        if isinstance(ttc, (int, float)) and ttc < 3.0:
            return "danger_zone"
        
        # Very close gap = danger zone
        if isinstance(leader_gap, (int, float)) and leader_gap < 5.0:
            return "danger_zone"
        
        # ===== CHECK FOR OVERTAKING =====
        # Has leader + gap > 5m + speed differential
        if isinstance(leader_gap, (int, float)) and leader_gap > 5.0 and leader_gap < 100:
            speed = state.get('speed', 0)
            leader_speed = state.get('LeaderSpeed', speed)
            
            # Ego faster than leader by 5+ km/h = overtaking
            if speed - leader_speed > 5:
                return "overtaking"
        
        # ===== CHECK FOR INTERSECTION =====
        # High speed + no leader = could be intersection
        # But be conservative (speed > 30, not 40)
        speed = state.get('speed', 0)
        if speed > 30 and (isinstance(leader_gap, (int, float)) and leader_gap > 100):
            return "intersection"
        
        return None

    def check_immediate_danger(self, vehicle_state: Dict) -> Optional[CriticalAlert]:
        """
        Rule-based scenario detection + ML algorithm inference (must complete in < 50ms)
        1. Detect scenario using rules (intersection, overtaking, danger_zone)
        2. Call appropriate ML model for detected scenario
        Returns CriticalAlert if danger detected, None otherwise.
        """
        start_time = time.perf_counter()
        self.performance_stats["total_checks"] += 1

        alert = None

        # Step 1: Detect scenario using rules
        detected_scenario = self._detect_scenario(vehicle_state)
        
        # Step 2: Call appropriate ML model if models are available
        if self.loaded_scenario_models and detected_scenario and detected_scenario in self.loaded_scenario_models:
            ml_result = self.evaluate_scenario_model(detected_scenario, vehicle_state)
            if "error" not in ml_result:
                risk = float(ml_result.get("risk_score", 0.0))  # P(unsafe) from model
                unsafe = bool(ml_result.get("is_unsafe", False))

                # Only generate alert if risk exceeds confidence threshold
                if unsafe and risk >= self.confidence_threshold:
                    # Severity based on how much risk exceeds threshold
                    confidence_margin = risk - self.confidence_threshold
                    
                    if risk >= 0.90:
                        severity = "emergency"
                    elif risk >= 0.80:
                        severity = "critical"
                    elif risk >= self.confidence_threshold:
                        severity = "warning"
                    else:
                        severity = None

                    if severity is not None:
                        alert = CriticalAlert(
                            type=f"{detected_scenario}_ml_risk",
                            severity=severity,
                            message=f"ML {severity.upper()}: {detected_scenario} risk score={risk:.2f}",
                            timestamp=time.time(),
                            reaction_time_ms=0.0,
                            parameters=ml_result,
                        )

        # Calcul du temps de réaction
        if alert:
            reaction_time = (time.perf_counter() - start_time) * 1000
            alert.reaction_time_ms = reaction_time

            self.alert_history.append(alert)
            self.performance_stats["alerts_generated"] += 1
            self.performance_stats["avg_response_time"] = (
                    self.performance_stats["avg_response_time"] * 0.9 + reaction_time * 0.1
            )
            self.performance_stats["max_response_time"] = max(
                self.performance_stats["max_response_time"], reaction_time
            )

        return alert

    def evaluate_scenario_model(self, scenario: str, vehicle_state: Dict) -> Dict[str, Any]:
        """Run saved model inference for a specific scenario.

        Returns a dict with:
        - prediction (0 unsafe, 1 safe)
        - is_unsafe
        - risk_score (probability of unsafe)
        """
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
        results: Dict[str, Dict[str, Any]] = {}
        for scenario in self.loaded_scenario_models:
            results[scenario] = self.evaluate_scenario_model(scenario, vehicle_state)
        return results
    # ===== NOUVELLES FONCTIONS POUR ZONE DANGEREUSE =====

    def _check_danger_zone(self, state: Dict) -> bool:
        """
        Vérifie si le véhicule se trouve dans une zone dangereuse
        """
        # Indicateurs directs de danger
        hard_brake = state.get('hard_brake', 0)
        leader_stopped = state.get('leader_stopped', 0)
        danger_accel = state.get('danger_accel_toward_leader', 0)
        no_reaction = state.get('no_reaction_to_stopped_leader', 0)

        direct_danger = [
            hard_brake == 1,
            leader_stopped == 1,
            danger_accel == 1,
            no_reaction == 1
        ]

        # Si un indicateur direct est présent, c'est une zone dangereuse
        if any(direct_danger):
            if hard_brake == 1:
                print(f"⚠️ Danger zone: hard_brake detected")
            return True

        # Indicateurs indirects
        ttc = state.get('TTC', float('inf'))
        leader_gap = state.get('LeaderGap', float('inf'))

        if ttc < 5.0 and ttc > 2.0:
            print(f"⚠️ Danger zone: TTC={ttc:.2f}s")
            return True

        if leader_gap != float('inf'):
            safe_distance = self._calculate_safe_distance(state)
            if leader_gap < safe_distance * 0.7:
                print(f"⚠️ Danger zone: gap={leader_gap:.1f}m < safe_dist*0.7")
                return True

        return False
    def _create_danger_zone_alert(self, state: Dict) -> CriticalAlert:
        """Crée une alerte pour zone dangereuse"""
        # Identifier la raison du danger
        reasons = []

        if state.get('hard_brake', 0) == 1:
            reasons.append("freinage d'urgence détecté")
        if state.get('leader_stopped', 0) == 1:
            reasons.append("véhicule arrêté devant")
        if state.get('danger_accel_toward_leader', 0) == 1:
            reasons.append("approche dangereuse")
        if state.get('TTC', float('inf')) < 5.0:
            reasons.append(f"TTC dangereux: {state.get('TTC', 0):.1f}s")

        reason_text = ", ".join(reasons) if reasons else "conditions dangereuses"

        return CriticalAlert(
            type="danger_zone",
            severity="critical",
            message=f"⚠️ ZONE DANGEREUSE: {reason_text}",
            timestamp=time.time(),
            reaction_time_ms=0.0,
            parameters={
                "ttc": state.get('TTC', None),
                "speed": state.get('speed', 0),
                "hard_brake": state.get('hard_brake', 0),
                "leader_stopped": state.get('leader_stopped', 0),
                "danger_accel": state.get('danger_accel_toward_leader', 0)
            }
        )

    # ===== FONCTIONS EXISTANTES =====

    def _check_emergency_brake(self, state: Dict) -> bool:
        """Check if emergency braking is required (TTC < threshold)"""
        ttc = self._calculate_ttc(state)
        if ttc is None or ttc > self.config.CRITICAL_TTC:
            return False

        required_decel = self._calculate_required_decel(state)
        return required_decel > VEHICLE_CONFIG.MAX_DECELERATION * 0.8

    def _check_critical_speed(self, state: Dict) -> bool:
        """Check if speed is critically high"""
        current_speed = state.get('speed', 0)
        speed_limit = state.get('speed_limit', 90)

        return (current_speed > self.config.CRITICAL_SPEED or
                current_speed > speed_limit * 1.1)

    def _check_unsafe_following(self, state: Dict) -> bool:
        if 'leading_vehicle' not in state or not state['leading_vehicle']:
            return False

        distance = state['leading_vehicle'].get('distance', float('inf'))
        if distance == float('inf'):
            return False

        safe_distance = self._calculate_safe_distance(state)
        return distance < safe_distance * 0.5

    def _check_sudden_obstacle(self, state: Dict) -> bool:
        """Check for sudden obstacles (rapid deceleration ahead)"""
        if 'leading_vehicle' not in state:
            return False

        lead_accel = state['leading_vehicle'].get('acceleration', 0)
        return lead_accel < -self.config.EMERGENCY_DECEL_THRESHOLD

    def _calculate_ttc(self, state: Dict) -> Optional[float]:
        """Calculate Time To Collision with leading vehicle"""
        if ('leading_vehicle' not in state or
                not state['leading_vehicle'] or
                'distance' not in state['leading_vehicle']):
            return None

        distance = state['leading_vehicle']['distance']
        ego_speed = state.get('speed', 0) / 3.6
        lead_speed = state['leading_vehicle'].get('speed', 0) / 3.6

        relative_speed = ego_speed - lead_speed

        if relative_speed <= 0:
            return float('inf')

        return distance / relative_speed

    def _calculate_required_decel(self, state: Dict) -> float:
        """Calculate required deceleration to avoid collision"""
        ttc = self._calculate_ttc(state)
        if ttc is None or ttc == float('inf'):
            return 0.0

        ego_speed = state.get('speed', 0) / 3.6
        return ego_speed / ttc if ttc > 0 else float('inf')

    def _calculate_safe_distance(self, state: Dict) -> float:
        """Calculate safe following distance based on speed"""
        speed_mps = state.get('speed', 0) / 3.6

        return max(
            self.config.MIN_SAFE_DISTANCE,
            speed_mps * 2 + VEHICLE_CONFIG.VEHICLE_LENGTH
        )

    def get_performance_report(self) -> Dict:
        """Get performance statistics report"""
        return {
            **self.performance_stats,
            "meets_sla": self.performance_stats.get("max_response_time", 0) < self.config.MAX_REACTIVE_TIME_MS,
            "alert_types": {
                alert.type: len([a for a in self.alert_history if a.type == alert.type])
                for alert in self.alert_history[:10]
            }
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
        """
        Adjust the confidence threshold for alert generation.
        
        Args:
            threshold: Risk score threshold (0.0-1.0)
                - 0.50: Very sensitive (detect 100% but many false alerts)
                - 0.70: Balanced (good default for safety-critical)
                - 0.80: More selective (fewer false alerts)
                - 0.90: Very conservative (may miss some dangers)
        
        Example:
            reactive.set_confidence_threshold(0.80)  # More selective
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        self.confidence_threshold = threshold
        print(f"✅ Confidence threshold updated to {threshold:.0%}")


# Test function for PyCharm
def test_reactive_layer():
    """Test the reactive layer implementation"""
    print("\n🧪 Testing Reactive Layer...")

    reactive = ReactiveLayer()

    # Test 1: Emergency brake scenario
    print("\nTest 1: Emergency Brake")
    test_state = {
        "speed": 80,
        "leading_vehicle": {
            "distance": 10,
            "speed": 0,
            "acceleration": 0
        }
    }

    alert = reactive.check_immediate_danger(test_state)
    if alert:
        print(f"✅ Alert generated: {alert.type}")
        print(f"   Message: {alert.message}")
        print(f"   Reaction time: {alert.reaction_time_ms:.1f}ms")
    else:
        print("❌ No alert (should have detected emergency)")

    # Test 2: Danger zone with hard brake
    print("\nTest 2: Danger Zone (hard brake)")
    test_state = {
        "speed": 60,
        "hard_brake": 1,
        "TTC": 3.5,
        "LeaderGap": 15
    }

    alert = reactive.check_immediate_danger(test_state)
    if alert and alert.type == "danger_zone":
        print(f"✅ Danger zone detected: {alert.message}")
    else:
        print("❌ Failed to detect danger zone")

    # Test 3: Critical speed
    print("\nTest 3: Critical Speed")
    test_state = {
        "speed": 95,
        "speed_limit": 90
    }

    alert = reactive.check_immediate_danger(test_state)
    if alert and alert.type == "critical_speed":
        print(f"✅ Speed alert correctly detected")
    else:
        print("❌ Failed to detect speed violation")

    # Performance report
    print("\n📊 Performance Report:")
    report = reactive.get_performance_report()
    for key, value in report.items():
        if key != "alert_types":
            print(f"  {key}: {value}")

    print("\n✅ Reactive Layer Tests Complete!")


if __name__ == "__main__":
    test_reactive_layer()