# Rule-Based Scenario Detection + ML Algorithm Pipeline

## Overview
The Reactive Layer now implements a two-stage safety assessment system:
1. **Stage 1: Rule-Based Scenario Detection** - Detects which scenario (intersection, overtaking, danger_zone) applies
2. **Stage 2: ML Algorithm Inference** - Calls the appropriate trained ML model for that scenario

## Workflow

```
vehicle_state
    ↓
Rule-Based Scenario Detection
    ├→ Check Intersection Rules → Lane undefined? Angle present? Traffic signal?
    ├→ Check Overtaking Rules → Lane change? Speed differential? SafeDistance metric?
    └→ Check Danger Zone Rules → Hard brake? Leader stopped? Collision flags?
    ↓ (detected_scenario)
Load Appropriate ML Model
    ├→ danger_zone_model.joblib
    ├→ overtaking_model.joblib
    └→ intersection_model.joblib
    ↓
ML Inference
    ├→ Build feature vector from vehicle_state
    ├→ Predict safety: 0=unsafe, 1=safe
    └→ Return risk_score (0.0-1.0)
    ↓
Map Risk Score to Severity
    ├→ emergency: risk_score ≥ 0.85
    ├→ critical: risk_score ≥ 0.70
    ├→ warning: risk_score ≥ 0.55
    └→ none: risk_score < 0.55
    ↓
CriticalAlert (or None)
```

## Rule-Based Scenario Detection Details

### Intersection Detection Rules
Detects scenarios involving traffic intersections or unclear lane definitions:
- **Lane undefined** (Lane=0, Lane=-1, or None)
- **Angle change** present (heading/bearing change)
- **Traffic signal/light** flags set
- **Multiple vehicles** in different directions (num_vehicles_nearby > 2)
- **No leading vehicle** context (LeaderGap > 100m)

### Overtaking Detection Rules
Detects lane-changing and passing maneuvers:
- **Lane change in progress** flag set
- **Lane opposite change** indicator (attempting different lane)
- **SafeDistance metric** available + leader gap exceeds safe distance
- **Speed differential** (ego speed > leader_speed + 10 km/h) + leader gap > 5m
- **Lateral position change** (significant lateral movement ≥1.5m)

### Danger Zone Detection Rules
Detects emergency/critical situations requiring immediate response:
- **Hard brake** flag set (sudden emergency braking)
- **Leader stopped** flag set (vehicle ahead stopped)
- **Danger acceleration** toward leader (unsafe approach)
- **No reaction to stopped leader** flag set

**Note:** Danger zone rules use explicit flags only - no TTC/gap thresholds to avoid false positives in normal traffic.

## Implementation Example

### Input (Frame without Scenario Label)
```python
vehicle_state = {
    "speed": 78.0,
    "acceleration": 0.8,
    "LeaderGap": 15.0,
    "LeaderSpeed": 60.0,
    "Lane": 21,
    "LanePrev": 20,
    "lane_opposite_change": 1,      # ← Overtaking indicator
    "SafeDistance": 12.0,            # ← Overtaking indicator
    "TTC": 2.5,
    "leading_vehicle": {"distance": 15.0, "speed": 60.0},
}
```

### Processing
```
1. Rule Detection:
   - Check intersection rules: False (Lane defined, no angle)
   - Check overtaking rules: True (lane_opposite_change=1, SafeDistance present)
   → Detected scenario: "overtaking"

2. ML Inference:
   - Load overtaking_model.joblib
   - Extract 15 features from vehicle_state
   - Run RandomForest prediction
   - Risk score: 0.945

3. Severity Mapping:
   - risk_score=0.945 ≥ 0.85 and is_unsafe=True
   → severity: "emergency"

4. Return CriticalAlert:
   {
      "type": "overtaking_ml_risk",
      "severity": "emergency",
      "message": "ML EMERGENCY: overtaking risk score=0.95",
      "parameters": {
         "risk_score": 0.945,
         "is_unsafe": True,
         "prediction": 0,
         "probabilities": {...}
      }
   }
```

## ML Models Used

| Scenario | Model File | Samples | Features | Algorithm |
|----------|-----------|---------|----------|-----------|
| **intersection** | intersection_model.joblib | 44,176 | 15 | RandomForest (200 trees) |
| **overtaking** | overtaking_model.joblib | 33,778 | 15 | RandomForest (200 trees) |
| **danger_zone** | danger_zone_model.joblib | 25,689 | 14 | RandomForest (200 trees) |

All models use StandardScaler preprocessing for feature normalization.

## Key Benefits

✅ **Interpretable Detection** - Rules based on meaningful vehicle dynamics features  
✅ **Fast Execution** - Rule-based detection is O(1), typical response time: 60-80ms  
✅ **Scenario-Specific** - Each ML model trained on scenario-relevant data  
✅ **No Manual Labeling** - Scenarios detected automatically, not passed in externally  
✅ **Fallback Support** - Fallback logic handles edge cases and unusual combinations  

## Usage in Code

```python
from core.agent.ReactiveLayer import ReactiveLayer

# Initialize reactive layer (loads all 3 models automatically)
reactive = ReactiveLayer()

# Check for danger with automatic scenario detection
vehicle_state = {...}  # No "scenario" field needed!
alert = reactive.check_immediate_danger(vehicle_state)

if alert:
    print(f"Alert: {alert.type}")
    print(f"Severity: {alert.severity}")
    print(f"Risk Score: {alert.parameters['risk_score']}")
else:
    print("No danger detected")
```

## Demo Execution

Run the demo to see rule-based detection in action:
```bash
python core/main.py
```

Expected output shows:
1. **Frame 1 (Intersection)**: Rule detects undefined Lane + Angle → calls intersection model
2. **Frame 2 (Overtaking)**: Rule detects lane_opposite_change + SafeDistance → calls overtaking model  
3. **Frame 3 (Danger Zone)**: Rule detects hard_brake + leader_stopped → calls danger_zone model

---

**Created**: 2026-03-17  
**Version**: 1.0 - Rule-Based Scenario Detection + ML Pipeline
