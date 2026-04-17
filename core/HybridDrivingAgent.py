"""
Hybrid Driving Agent: Integrates Reactive Layer and BDI Layer
"""
import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from collections import deque

from core.agent.ReactiveLayer import ReactiveLayer, CriticalAlert
from core.agent.bdi_layer import BDILayer, Intention
from core.agent.scenario_model_registry import ScenarioModelRegistry
from config.settings import AGENT_CONFIG, AgentConfig


@dataclass
class VehicleState:
    """Represents current vehicle state from VEINS data"""
    speed: float = 0.0
    acceleration: float = 0.0
    deceleration: float = 0.0
    ttc: float = 999.0
    leader_gap: float = -1.0
    leader_speed: float = 0.0
    relative_speed: float = 0.0
    hard_brake: bool = False
    leader_stopped: bool = False
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Speed": self.speed,
            "Acceleration": self.acceleration,
            "Deceleration": self.deceleration,
            "TTC": self.ttc,
            "LeaderGap": self.leader_gap,
            "LeaderSpeed": self.leader_speed,
            "RelativeSpeed": self.relative_speed,
            "hard_brake": 1 if self.hard_brake else 0,
            "leader_stopped": 1 if self.leader_stopped else 0,
        }


@dataclass
class HybridAgentResult:
    """Result from hybrid agent processing"""
    alert_type: str  # "reactive", "contextual", "none"
    message: Optional[str] = None
    situation: Optional[str] = None
    reaction_time_ms: float = 0.0
    ml_predictions: Dict[str, float] = field(default_factory=dict)


class HybridDrivingAgent:
    """
    Hybrid reactive + BDI agent for driving assistance.
    Uses rule-based detection for speed, ML for confidence scoring.
    """

    def __init__(self, config: AgentConfig = None, confidence_threshold: float = 0.70):
        self.config = config or AGENT_CONFIG
        self.confidence_threshold = confidence_threshold

        self.vehicle_state = VehicleState()
        self.alert_history = deque(maxlen=20)

        # Initialize components
        self.model_registry = ScenarioModelRegistry()
        self.loaded_models = self.model_registry.load_available_models()
        self.reactive_layer = ReactiveLayer(config=self.config, confidence_threshold=confidence_threshold)
        self.bdi_layer = BDILayer(config=self.config)

        self.performance_stats = {
            "total_frames": 0,
            "reactive_alerts": 0,
            "contextual_alerts": 0,
            "avg_reaction_time_ms": 0.0,
            "max_reaction_time_ms": 0.0,
            "scenarios_detected": {"danger_zone": 0, "overtaking": 0, "intersection": 0}
        }

        print("✅ Hybrid Driving Agent initialized")
        print(f"   Models available: {', '.join(self.loaded_models) if self.loaded_models else 'None (rule-based only)'}")
        print(f"   Confidence Threshold: {confidence_threshold:.0%}")

    def process_frame(self, frame_data: Dict) -> HybridAgentResult:
        """Main processing loop - FAST PATH (<10ms target)"""
        start_time = time.perf_counter()
        self.performance_stats["total_frames"] += 1

        # 1. Update state
        self._update_state(frame_data)
        vehicle_dict = self.vehicle_state.to_dict()

        # 2. FAST PATH: Rule-based scenario detection (<1ms)
        scenario = self._detect_scenario_fast(vehicle_dict)
        confidence = self._calculate_confidence(scenario, vehicle_dict)

        # 3. Generate alert if confidence exceeds threshold
        alert_type = "none"
        message = None

        if scenario and confidence >= self.confidence_threshold:
            if scenario == "danger_zone":
                alert_type = "reactive"
                message = self._danger_zone_message(vehicle_dict)
                self.performance_stats["reactive_alerts"] += 1
            else:
                alert_type = "contextual"
                message = self._contextual_message(scenario)
                self.performance_stats["contextual_alerts"] += 1

            self.performance_stats["scenarios_detected"][scenario] += 1
            self._record_alert(scenario)

        # 4. Track performance
        reaction_time = (time.perf_counter() - start_time) * 1000
        self._update_performance_stats(reaction_time)

        # 5. Optional: Get ML predictions for logging (async - doesn't block)
        ml_predictions = {"danger_zone": confidence if scenario == "danger_zone" else 0.0,
                          "overtaking": confidence if scenario == "overtaking" else 0.0,
                          "intersection": confidence if scenario == "intersection" else 0.0}

        return HybridAgentResult(
            alert_type=alert_type,
            message=message,
            situation=scenario,
            reaction_time_ms=reaction_time,
            ml_predictions=ml_predictions
        )

    def _update_state(self, frame_data: Dict):
        """Update vehicle state from frame data"""
        self.vehicle_state.speed = float(frame_data.get("Speed", 0.0))
        self.vehicle_state.acceleration = float(frame_data.get("Acceleration", 0.0))
        self.vehicle_state.deceleration = float(frame_data.get("Deceleration", 0.0))
        self.vehicle_state.ttc = float(frame_data.get("TTC", 999.0))
        self.vehicle_state.leader_gap = float(frame_data.get("LeaderGap", -1.0))
        self.vehicle_state.leader_speed = float(frame_data.get("LeaderSpeed", 0.0))
        self.vehicle_state.relative_speed = float(frame_data.get("RelativeSpeed", 0.0))
        self.vehicle_state.hard_brake = bool(frame_data.get("hard_brake", 0))
        self.vehicle_state.leader_stopped = bool(frame_data.get("leader_stopped", 0))
        self.vehicle_state.timestamp = time.time()

    def _detect_scenario_fast(self, state: Dict) -> Optional[str]:
        """Ultra-fast rule-based scenario detection"""
        ttc = state.get("TTC", 999)
        hard_brake = state.get("hard_brake", 0)
        leader_stopped = state.get("leader_stopped", 0)
        rel_speed = state.get("RelativeSpeed", 0)
        speed = state.get("Speed", 0)
        deceleration = state.get("Deceleration", 0)

        # PRIORITY 1: Danger zone (safety critical)
        if hard_brake or leader_stopped or (0 < ttc < 2.5):
            return "danger_zone"

        # PRIORITY 2: Overtaking
        if speed > 15 and rel_speed > 3.0 and ttc > 2.5:
            return "overtaking"

        # PRIORITY 3: Intersection
        if speed < 60 and deceleration > 0.5:
            return "intersection"

        return None

    def _calculate_confidence(self, scenario: Optional[str], state: Dict) -> float:
        """Calculate confidence based on scenario indicators"""
        if scenario is None:
            return 0.0

        if scenario == "danger_zone":
            ttc = state.get("TTC", 999)
            if state.get("hard_brake", 0) or state.get("leader_stopped", 0):
                return 0.99
            elif ttc < 1.5:
                return 0.99
            elif ttc < 2.0:
                return 0.90
            elif ttc < 2.5:
                return 0.75
            return 0.60

        elif scenario == "overtaking":
            rel_speed = state.get("RelativeSpeed", 0)
            if rel_speed > 8.0:
                return 0.95
            elif rel_speed > 5.0:
                return 0.85
            elif rel_speed > 3.0:
                return 0.70
            return 0.55

        elif scenario == "intersection":
            decel = state.get("Deceleration", 0)
            speed = state.get("Speed", 0)
            if decel > 1.0 and speed < 40:
                return 0.85
            elif decel > 0.5:
                return 0.70
            return 0.55

        return 0.0

    def _danger_zone_message(self, state: Dict) -> str:
        """Generate danger zone alert message"""
        ttc = state.get("TTC", 999)
        if state.get("hard_brake", 0) or state.get("leader_stopped", 0) or ttc < 1.5:
            return "EMERGENCY! Brake immediately! Collision imminent!"
        elif ttc < 2.0:
            return "CRITICAL: High collision risk! Brake now!"
        else:
            return "Warning: Dangerous zone. Increase following distance."

    def _contextual_message(self, scenario: str) -> str:
        """Generate contextual alert message"""
        messages = {
            "overtaking": "Attention: Vehicle overtaking detected. Maintain lane.",
            "intersection": "Approaching intersection. Reduce speed."
        }
        return messages.get(scenario, "Caution advised.")

    def _is_alert_fatigued(self, scenario: str, cooldown: float = 3.0) -> bool:
        """Check if same alert was sent recently"""
        current = time.time()
        for alert in self.alert_history:
            if alert["scenario"] == scenario:
                if current - alert["timestamp"] < cooldown:
                    return True
        return False

    def _record_alert(self, scenario: str):
        """Record alert in history"""
        self.alert_history.append({"scenario": scenario, "timestamp": time.time()})

    def _update_performance_stats(self, reaction_time_ms: float):
        """Update performance statistics"""
        self.performance_stats["avg_reaction_time_ms"] = (
            self.performance_stats["avg_reaction_time_ms"] * 0.9 + reaction_time_ms * 0.1
        )
        self.performance_stats["max_reaction_time_ms"] = max(
            self.performance_stats["max_reaction_time_ms"], reaction_time_ms
        )

    def get_performance_report(self) -> Dict:
        """Get performance report"""
        return {
            **self.performance_stats,
            "meets_sla": self.performance_stats["max_reaction_time_ms"] < 50.0
        }

    def reset_stats(self):
        """Reset statistics"""
        self.performance_stats = {
            "total_frames": 0,
            "reactive_alerts": 0,
            "contextual_alerts": 0,
            "avg_reaction_time_ms": 0.0,
            "max_reaction_time_ms": 0.0,
            "scenarios_detected": {"danger_zone": 0, "overtaking": 0, "intersection": 0}
        }
        self.alert_history.clear()


# ===== TEST FUNCTION =====

def test_hybrid_agent():
    """Test the hybrid agent"""
    print("\n" + "="*70)
    print("🧪 TESTING HYBRID DRIVING AGENT")
    print("="*70)

    agent = HybridDrivingAgent(confidence_threshold=0.65)

    tests = [
        ("Danger Zone", {"Speed": 80, "TTC": 1.5, "hard_brake": 1, "leader_stopped": 1}),
        ("Overtaking", {"Speed": 95, "RelativeSpeed": 12, "LeaderGap": 35, "TTC": 4.5}),
        ("Intersection", {"Speed": 45, "Deceleration": 1.0, "TTC": 6.0}),
        ("Normal", {"Speed": 75, "TTC": 8.0, "LeaderGap": 50})
    ]

    for name, frame in tests:
        print(f"\n📊 {name} Frame")
        print("-"*50)
        result = agent.process_frame(frame)
        print(f"   Alert: {result.alert_type}")
        print(f"   Situation: {result.situation}")
        print(f"   Message: {result.message}")
        print(f"   Time: {result.reaction_time_ms:.2f} ms")

    print("\n" + "="*70)
    print("📊 PERFORMANCE REPORT")
    print("="*70)
    report = agent.get_performance_report()
    print(f"Total Frames: {report['total_frames']}")
    print(f"Reactive Alerts: {report['reactive_alerts']}")
    print(f"Contextual Alerts: {report['contextual_alerts']}")
    print(f"Avg Time: {report['avg_reaction_time_ms']:.3f} ms")
    print(f"Max Time: {report['max_reaction_time_ms']:.3f} ms")
    print(f"Meets SLA (<50ms): {report['meets_sla']}")
    print(f"Scenarios: {report['scenarios_detected']}")
    print("\n✅ Tests Complete!")


if __name__ == "__main__":
    test_hybrid_agent()