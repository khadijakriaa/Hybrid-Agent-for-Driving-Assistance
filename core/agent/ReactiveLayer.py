"""
Reactive Layer: Fast, rule-based emergency response
"""
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

from config.settings import AGENT_CONFIG, VEHICLE_CONFIG, AgentConfig


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

    def __init__(self, config: AgentConfig = None):
        self.config = config or AGENT_CONFIG
        self.alert_history = []
        self.performance_stats = {
            "total_checks": 0,
            "alerts_generated": 0,
            "avg_response_time": 0.0,
            "max_response_time": 0.0
        }
        print("✅ Reactive Layer initialized")

    def check_immediate_danger(self, vehicle_state: Dict) -> Optional[CriticalAlert]:
        """
        Check for immediate dangers (must complete in < 50ms)
        Returns CriticalAlert if danger detected, None otherwise
        """
        start_time = time.perf_counter()
        self.performance_stats["total_checks"] += 1

        alert = None

        # PRIORITÉ 1: Collision imminente (le plus dangereux)
        if self._check_emergency_brake(vehicle_state):
            ttc = self._calculate_ttc(vehicle_state)
            alert = CriticalAlert(
                type="emergency_brake",
                severity="emergency",
                message="EMERGENCY: Collision imminent! Brake hard now!",
                timestamp=time.time(),
                reaction_time_ms=0.0,
                parameters={"ttc": ttc, "required_deceleration": self._calculate_required_decel(vehicle_state)}
            )

        # PRIORITÉ 2: Zone dangereuse (basée sur les indicateurs du dataset)
        elif self._check_danger_zone(vehicle_state):
            alert = self._create_danger_zone_alert(vehicle_state)
            print(f"🔴 DANGER ZONE DETECTED!")  # Pour déboguer

        # PRIORITÉ 3: Vitesse critique
        elif self._check_critical_speed(vehicle_state):
            alert = CriticalAlert(
                type="critical_speed",
                severity="critical",
                message=f"CRITICAL: Speed {vehicle_state.get('speed', 0):.0f} km/h exceeds safety limit!",
                timestamp=time.time(),
                reaction_time_ms=0.0,
                parameters={"current_speed": vehicle_state.get('speed', 0),
                            "speed_limit": vehicle_state.get('speed_limit', 90)}
            )

        # PRIORITÉ 4: Freinage soudain du véhicule devant
        elif self._check_sudden_obstacle(vehicle_state):
            alert = CriticalAlert(
                type="sudden_obstacle",
                severity="critical",
                message="CRITICAL: Sudden obstacle detected! Take evasive action!",
                timestamp=time.time(),
                reaction_time_ms=0.0
            )

        # PRIORITÉ 5: Distance de sécurité insuffisante
        elif self._check_unsafe_following(vehicle_state):
            distance = vehicle_state.get('leading_vehicle', {}).get('distance', 0)
            alert = CriticalAlert(
                type="unsafe_following",
                severity="warning",
                message=f"WARNING: Following too close! Distance: {distance:.1f}m",
                timestamp=time.time(),
                reaction_time_ms=0.0,
                parameters={"following_distance": distance,
                            "min_safe_distance": self._calculate_safe_distance(vehicle_state)}
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