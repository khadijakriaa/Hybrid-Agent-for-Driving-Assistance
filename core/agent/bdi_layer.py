"""
BDI Layer: Belief-Desire-Intention reasoning for contextual decisions
"""
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from collections import deque
import numpy as np

from config.settings import AGENT_CONFIG, AgentConfig


@dataclass
class Belief:
    """Agent's beliefs about the current driving context"""
    timestamp: float
    vehicle_state: Dict[str, Any]
    context_probs: Dict[str, float]
    surroundings: Dict[str, Any]
    road_conditions: Dict[str, Any]

    def get_primary_context(self, threshold: float = 0.5) -> str:
        if not self.context_probs:
            return "unknown"
        sorted_contexts = sorted(
            self.context_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        primary, confidence = sorted_contexts[0]
        return primary if confidence > threshold else "uncertain"

    def get_context_confidence(self, context: str) -> float:
        return self.context_probs.get(context, 0.0)


@dataclass
class Desire:
    """Agent's desires with weights"""
    safety: float = 0.0
    information: float = 0.0
    comfort: float = 0.0
    efficiency: float = 0.0

    def weighted_sum(self, weights: Dict[str, float]) -> float:
        return (
                self.safety * weights.get("safety", 0) +
                self.information * weights.get("information", 0) +
                self.comfort * weights.get("comfort", 0) +
                self.efficiency * weights.get("efficiency", 0)
        )


@dataclass
class Intention:
    """Agent's intention/planned action"""
    action_type: str
    utility: float
    parameters: Dict[str, Any]
    priority: int
    requires_alert: bool = True

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.utility > other.utility


class BDILayer:
    """
    Belief-Desire-Intention reasoning layer.
    Handles contextual analysis and strategic decision making.
    """

    def __init__(self, config: AgentConfig = None):
        self.config = config or AGENT_CONFIG
        self.beliefs_history = deque(maxlen=50)
        self.intentions_history = deque(maxlen=20)
        self.current_belief: Optional[Belief] = None
        self.current_desire: Optional[Desire] = None
        self.current_intentions: List[Intention] = []

        self.performance_stats = {
            "total_cycles": 0,
            "avg_cycle_time_ms": 0.0,
            "max_cycle_time_ms": 0.0,
            "intentions_generated": 0
        }

        print("✅ BDI Layer initialized")

    def process_context(self,
                        vehicle_state: Dict,
                        ml_predictions: Dict) -> List[Intention]:
        """Main BDI processing cycle"""
        start_time = time.perf_counter()
        self.performance_stats["total_cycles"] += 1

        self.current_belief = self._update_beliefs(vehicle_state, ml_predictions)
        self.current_desire = self._evaluate_desires(self.current_belief)
        self.current_intentions = self._form_intentions(
            self.current_belief,
            self.current_desire
        )

        selected_intentions = [
            intention for intention in self.current_intentions
            if intention.utility >= self.config.UTILITY_THRESHOLD
        ]

        selected_intentions.sort()
        self.intentions_history.extend(selected_intentions)
        self.performance_stats["intentions_generated"] += len(selected_intentions)

        cycle_time = (time.perf_counter() - start_time) * 1000
        self._update_performance_stats(cycle_time)

        return selected_intentions

    def get_intentions_for_reactive(self, reactive_alert: Optional[Dict] = None) -> List[Intention]:
        """Get intentions, suppressing redundant alerts if reactive layer handled emergency"""
        if reactive_alert and reactive_alert.get("severity") == "emergency":
            filtered = [
                i for i in self.current_intentions
                if i.action_type != "alert_danger_zone"
            ]
            return filtered
        return self.current_intentions

    def _update_beliefs(self, vehicle_state: Dict, ml_predictions: Dict) -> Belief:
        surroundings = self._analyze_surroundings(vehicle_state)
        road_conditions = self._analyze_road_conditions(vehicle_state)

        belief = Belief(
            timestamp=time.time(),
            vehicle_state=vehicle_state,
            context_probs=ml_predictions,
            surroundings=surroundings,
            road_conditions=road_conditions
        )

        self.beliefs_history.append(belief)
        return belief

    def _evaluate_desires(self, belief: Belief) -> Desire:
        desire = Desire()
        desire.safety = self._calculate_safety_desire(belief)
        desire.information = self._calculate_information_desire(belief)
        desire.comfort = self._calculate_comfort_desire(belief)
        desire.efficiency = self._calculate_efficiency_desire(belief)
        return desire

    def _form_intentions(self, belief: Belief, desire: Desire) -> List[Intention]:
        intentions = []
        possible_actions = self._generate_possible_actions(belief)

        for action in possible_actions:
            utility = self._calculate_action_utility(action, belief, desire)
            priority = self._determine_action_priority(action["type"])

            intention = Intention(
                action_type=action["type"],
                utility=utility,
                parameters=action.get("parameters", {}),
                priority=priority,
                requires_alert=action.get("requires_alert", True)
            )
            intentions.append(intention)

        return intentions

    def _analyze_surroundings(self, state: Dict) -> Dict:
        return {
            "traffic_density": self._calculate_traffic_density(state),
            "lane_position": self._determine_lane_position(state),
            "intersection_proximity": state.get("distance_to_intersection", 1000),
            "overtaking_opportunity": self._check_overtaking_opportunity(state),
            "blind_spots": self._check_blind_spots(state)
        }

    def _analyze_road_conditions(self, state: Dict) -> Dict:
        return {
            "road_type": state.get("road_type", "unknown"),
            "visibility": state.get("visibility", "good"),
            "surface_condition": state.get("surface_condition", "dry"),
            "curvature": state.get("road_curvature", 0)
        }

    def _calculate_safety_desire(self, belief: Belief) -> float:
        risk_score = 0.0
        speed = belief.vehicle_state.get("Speed", 0)
        speed_limit = belief.vehicle_state.get("speed_limit", 90)
        if speed > speed_limit:
            risk_score += 0.3

        ttc = belief.vehicle_state.get("TTC", 999)
        if ttc < 2.0:
            risk_score += 0.5
        elif ttc < 3.0:
            risk_score += 0.3

        leader_gap = belief.vehicle_state.get("LeaderGap", 999)
        if leader_gap != -1 and leader_gap < 10:
            risk_score += 0.3

        if belief.vehicle_state.get("hard_brake", 0) == 1:
            risk_score += 0.4
        if belief.vehicle_state.get("leader_stopped", 0) == 1:
            risk_score += 0.4

        if belief.get_primary_context() == "danger_zone":
            risk_score += 0.3

        return min(1.0, risk_score)

    def _calculate_information_desire(self, belief: Belief) -> float:
        if belief.get_primary_context() == "uncertain":
            return 0.8
        context_probs = belief.context_probs
        if context_probs.get("overtaking", 0) > 0.6:
            return 0.7
        if context_probs.get("intersection", 0) > 0.6:
            return 0.7
        return 0.4

    def _calculate_comfort_desire(self, belief: Belief) -> float:
        acceleration = abs(belief.vehicle_state.get("Acceleration", 0))
        deceleration = abs(belief.vehicle_state.get("Deceleration", 0))
        if acceleration > 3.0 or deceleration > 3.0:
            return 0.2
        elif acceleration > 2.0 or deceleration > 2.0:
            return 0.4
        return 0.8

    def _calculate_efficiency_desire(self, belief: Belief) -> float:
        speed = belief.vehicle_state.get("Speed", 0)
        optimal_speed = 80
        if speed < 20:
            return 0.3
        elif speed > 120:
            return 0.4
        speed_diff = abs(speed - optimal_speed)
        if speed_diff < 10:
            return 0.9
        elif speed_diff < 20:
            return 0.6
        return 0.4

    def _generate_possible_actions(self, belief: Belief) -> List[Dict]:
        actions = []
        primary_context = belief.get_primary_context()

        if primary_context == "danger_zone":
            ttc = belief.vehicle_state.get("TTC", 999)
            hard_brake = belief.vehicle_state.get("hard_brake", 0)
            leader_stopped = belief.vehicle_state.get("leader_stopped", 0)

            if ttc < 2.0 or hard_brake or leader_stopped:
                severity = "emergency"
                recommended = "Brake immediately!"
            elif ttc < 3.0:
                severity = "critical"
                recommended = "Prepare to brake. High collision risk."
            else:
                severity = "warning"
                recommended = "Increase following distance."

            actions.append({
                "type": "alert_danger_zone",
                "parameters": {
                    "confidence": belief.get_context_confidence("danger_zone"),
                    "ttc": ttc,
                    "hard_brake_detected": bool(hard_brake),
                    "leader_stopped": bool(leader_stopped),
                    "severity": severity,
                    "recommended_action": recommended
                },
                "requires_alert": True
            })

        elif primary_context == "overtaking":
            rel_speed = belief.vehicle_state.get("RelativeSpeed", 0)
            actions.append({
                "type": "alert_overtaking",
                "parameters": {
                    "confidence": belief.get_context_confidence("overtaking"),
                    "relative_speed": rel_speed,
                    "recommended_action": "Vehicle overtaking detected. Maintain lane and speed."
                },
                "requires_alert": True
            })

        elif primary_context == "intersection":
            actions.append({
                "type": "alert_intersection",
                "parameters": {
                    "confidence": belief.get_context_confidence("intersection"),
                    "recommended_action": "Approaching intersection. Reduce speed and check crossing traffic."
                },
                "requires_alert": True
            })

        else:
            actions.append({
                "type": "provide_situational_awareness",
                "parameters": {
                    "dominant_context": primary_context,
                    "confidence": belief.get_context_confidence(primary_context) if primary_context != "unknown" else 0.0,
                    "speed": belief.vehicle_state.get("Speed", 0)
                },
                "requires_alert": False
            })

        return actions

    def _calculate_action_utility(self, action: Dict, belief: Belief, desire: Desire) -> float:
        """Calculate utility of an action"""
        action_type = action["type"]

        if action_type == "alert_danger_zone":
            base_utility = desire.safety * 0.8 + desire.information * 0.2
            severity = action.get("parameters", {}).get("severity", "warning")
            if severity == "emergency":
                return 1.0
            elif severity == "critical":
                base_utility = min(1.0, base_utility * 1.3)
            return min(1.0, base_utility)

        elif action_type == "alert_overtaking":
            confidence = belief.get_context_confidence("overtaking")
            base_utility = (
                    desire.information * 0.5 +
                    desire.safety * 0.3 +
                    confidence * 0.2
            )
            # Ensure high-confidence overtaking generates an alert
            if confidence > 0.7:
                base_utility = max(base_utility, 0.65)
            return min(1.0, base_utility)

        elif action_type == "alert_intersection":
            confidence = belief.get_context_confidence("intersection")
            base_utility = (
                    desire.information * 0.4 +
                    desire.safety * 0.4 +
                    confidence * 0.2
            )
            # Ensure high-confidence intersection generates an alert
            if confidence > 0.7:
                base_utility = max(base_utility, 0.65)
            return min(1.0, base_utility)

        elif action_type == "provide_situational_awareness":
            base_utility = desire.information * 0.5 + desire.comfort * 0.5
            return min(1.0, base_utility)

        else:
            return 0.3
    def _determine_action_priority(self, action_type: str) -> int:
        priority_map = {
            "alert_danger_zone": 1,
            "alert_intersection": 2,
            "alert_overtaking": 2,
            "provide_situational_awareness": 3
        }
        return priority_map.get(action_type, 3)

    def _calculate_traffic_density(self, state: Dict) -> float:
        nearby = state.get("nearby_vehicles", [])
        return min(1.0, len(nearby) / 10.0)

    def _determine_lane_position(self, state: Dict) -> str:
        lane = str(state.get("Lane", ""))
        if "left" in lane.lower():
            return "left_lane"
        elif "right" in lane.lower():
            return "right_lane"
        return "center_lane"

    def _check_overtaking_opportunity(self, state: Dict) -> bool:
        rel_speed = state.get("RelativeSpeed", 0)
        leader_gap = state.get("LeaderGap", 999)
        return rel_speed > 3.0 and leader_gap > 20

    def _check_blind_spots(self, state: Dict) -> List[str]:
        blind_spots = []
        nearby = state.get("nearby_vehicles", [])
        for vehicle in nearby:
            rel_position = vehicle.get("relative_position", "")
            if "blind_spot" in rel_position.lower():
                blind_spots.append(rel_position)
        return blind_spots

    def _update_performance_stats(self, cycle_time: float):
        self.performance_stats["avg_cycle_time_ms"] = (
                self.performance_stats["avg_cycle_time_ms"] * 0.9 + cycle_time * 0.1
        )
        self.performance_stats["max_cycle_time_ms"] = max(
            self.performance_stats["max_cycle_time_ms"], cycle_time
        )
        if cycle_time > self.config.MAX_BDI_CYCLE_MS:
            print(f"⚠️  Warning: BDI cycle took {cycle_time:.1f}ms (> {self.config.MAX_BDI_CYCLE_MS}ms)")

    def get_current_state(self) -> Dict:
        return {
            "belief": asdict(self.current_belief) if self.current_belief else None,
            "desire": asdict(self.current_desire) if self.current_desire else None,
            "intentions": [asdict(i) for i in self.current_intentions],
            "performance": self.performance_stats
        }

    def get_performance_report(self) -> Dict:
        safety_values = [d.safety for d in [self.current_desire] if self.current_desire]
        return {
            **self.performance_stats,
            "meets_sla": self.performance_stats.get("max_cycle_time_ms", 0) < self.config.MAX_BDI_CYCLE_MS,
            "current_safety_desire": safety_values[0] if safety_values else 0,
            "recent_intentions": [
                {"type": i.action_type, "utility": i.utility}
                for i in list(self.intentions_history)[-5:]
            ]
        }

    def reset_stats(self):
        self.performance_stats = {
            "total_cycles": 0,
            "avg_cycle_time_ms": 0.0,
            "max_cycle_time_ms": 0.0,
            "intentions_generated": 0
        }
        self.beliefs_history.clear()
        self.intentions_history.clear()


def test_bdi_layer():
    """Test the BDI layer implementation"""
    print("\n🧪 Testing BDI Layer...")
    bdi = BDILayer()

    # Test 1: Danger Zone
    print("\n" + "="*60)
    print("Test 1: Danger Zone Scenario")
    print("="*60)
    vehicle_state = {
        "Speed": 80, "TTC": 1.8, "LeaderGap": 12,
        "hard_brake": 1, "leader_stopped": 1,
        "Acceleration": -3.5, "Deceleration": 3.5
    }
    ml_predictions = {"danger_zone": 0.94, "overtaking": 0.03, "intersection": 0.03}
    intentions = bdi.process_context(vehicle_state, ml_predictions)
    print(f"✅ Generated {len(intentions)} intentions:")
    for i, intention in enumerate(intentions, 1):
        print(f"  {i}. {intention.action_type}")
        print(f"     Utility: {intention.utility:.2f}, Priority: {intention.priority}")

    # Test 2: Overtaking
    print("\n" + "="*60)
    print("Test 2: Overtaking Scenario")
    print("="*60)
    vehicle_state = {
        "Speed": 95, "RelativeSpeed": 12, "LeaderGap": 35,
        "Lane": "highway_left_0", "TTC": 4.5, "Acceleration": 1.2
    }
    ml_predictions = {"overtaking": 0.87, "danger_zone": 0.08, "intersection": 0.05}
    intentions = bdi.process_context(vehicle_state, ml_predictions)
    print(f"✅ Generated {len(intentions)} intentions:")
    for i, intention in enumerate(intentions, 1):
        print(f"  {i}. {intention.action_type} (utility: {intention.utility:.2f})")

    # Test 3: Intersection
    print("\n" + "="*60)
    print("Test 3: Intersection Scenario")
    print("="*60)
    vehicle_state = {
        "Speed": 45, "TTC": 6.0, "distance_to_intersection": 60,
        "Acceleration": -1.0
    }
    ml_predictions = {"intersection": 0.91, "danger_zone": 0.05, "overtaking": 0.04}
    intentions = bdi.process_context(vehicle_state, ml_predictions)
    print(f"✅ Generated {len(intentions)} intentions:")
    for i, intention in enumerate(intentions, 1):
        print(f"  {i}. {intention.action_type} (utility: {intention.utility:.2f})")

    # Performance report
    print("\n" + "="*60)
    print("📊 BDI Performance Report")
    print("="*60)
    report = bdi.get_performance_report()
    for key, value in report.items():
        if key != "recent_intentions":
            print(f"  {key}: {value}")

    print("\n✅ BDI Layer Tests Complete!")


if __name__ == "__main__":
    test_bdi_layer()