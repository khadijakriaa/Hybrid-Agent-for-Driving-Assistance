import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs('engineered_datasets', exist_ok=True)

print("🚀 Starting Feature Engineering Pipeline")
print("=" * 50)
# Dictionary to store all datasets
datasets = {}

# Define file paths (adjust these to your actual paths)
file_paths = {
    'danger_zone': 'dataset_apres_smote_DZ.csv',
    'overtaking_pure': 'dataset_apres_smote_overtaking.csv',
    'intersection': 'intersection.csv',
    'overtaking_raw1': 'overtaking.csv',
    'overtaking_raw2': 'overtakingsansobjet.csv',
    'danger_zone_raw': 'outputDZD.csv',
    'fusion': 'Fusion_dataset.csv'
}

# Load each dataset
for name, path in file_paths.items():
    try:
        datasets[name] = pd.read_csv(path)
        print(f"✅ Loaded {name}: {datasets[name].shape[0]} rows, {datasets[name].shape[1]} cols")
    except FileNotFoundError:
        print(f"❌ Could not find {path}")
        datasets[name] = None
print("=" * 50)

def engineer_danger_zone_features(df, dataset_name="danger_zone"):
    """Feature engineering for danger zone datasets"""
    if df is None:
        return None

    print(f"\n🔧 Engineering features for {dataset_name}...")
    features = df.copy()
    original_cols = len(features.columns)

    # 1. Time-based features
    if 'Time' in features.columns:
        features['time_normalized'] = features['Time'] / features['Time'].max()

    # 2. Speed-based features
    features['speed_squared'] = features['Speed'] ** 2  # Kinetic energy proxy
    features['speed_normalized'] = features['Speed'] / 130  # Assuming max 130 km/h

    # 3. TTC-based risk features
    features['ttc_risk_critical'] = (features['TTC'] < 2).astype(int)
    features['ttc_risk_warning'] = ((features['TTC'] >= 2) & (features['TTC'] < 4)).astype(int)
    features['ttc_risk_safe'] = (features['TTC'] >= 4).astype(int)
    features['ttc_inverse'] = 1 / (features['TTC'] + 0.01)  # Avoid division by zero

    # 4. Following distance features
    if 'LeaderGap' in features.columns and 'SafeDistance' in features.columns:
        features['gap_ratio'] = features['LeaderGap'] / (features['SafeDistance'] + 0.01)
        features['gap_deficit'] = features['SafeDistance'] - features['LeaderGap']
        features['gap_deficit_ratio'] = features['gap_deficit'] / (features['SafeDistance'] + 0.01)
        features['unsafe_following_severe'] = (features['gap_ratio'] < 0.3).astype(int)
        features['unsafe_following_moderate'] = ((features['gap_ratio'] >= 0.3) &
                                                 (features['gap_ratio'] < 0.5)).astype(int)

    # 5. Relative speed features
    if 'RelativeSpeed' in features.columns:
        features['closing_rate'] = (features['RelativeSpeed'] > 5).astype(int)
        features['separating_rate'] = (features['RelativeSpeed'] < -5).astype(int)

    # 6. Binary flag combinations
    binary_flags = ['hard_brake', 'danger_accel_toward_leader',
                    'leader_stopped', 'no_reaction_to_stopped_leader']

    # Count active flags
    active_flags = [col for col in binary_flags if col in features.columns]
    if active_flags:
        features['danger_flag_count'] = features[active_flags].sum(axis=1)

    # 7. Combined risk score (weighted)
    risk_weights = {
        'hard_brake': 0.4,
        'danger_accel_toward_leader': 0.3,
        'leader_stopped': 0.2,
        'no_reaction_to_stopped_leader': 0.3,
        'ttc_risk_critical': 0.5,
        'ttc_risk_warning': 0.2
    }

    features['combined_risk_score'] = 0.0
    for flag, weight in risk_weights.items():
        if flag in features.columns:
            features['combined_risk_score'] += features[flag] * weight
        elif flag == 'ttc_risk_critical' and 'ttc_risk_critical' in features.columns:
            features['combined_risk_score'] += features['ttc_risk_critical'] * weight
        elif flag == 'ttc_risk_warning' and 'ttc_risk_warning' in features.columns:
            features['combined_risk_score'] += features['ttc_risk_warning'] * weight

    # 8. Interaction features
    if all(col in features.columns for col in ['Speed', 'gap_ratio']):
        features['speed_x_gap'] = features['Speed'] * features['gap_ratio']

    print(f"   Added {len(features.columns) - original_cols} new features")
    print(f"   Total features: {len(features.columns)}")

    return features


# Apply to danger zone datasets
datasets['danger_zone_engineered'] = engineer_danger_zone_features(
    datasets['danger_zone'], 'danger_zone_pure'
)
if datasets['danger_zone_raw'] is not None:
    datasets['danger_zone_raw_engineered'] = engineer_danger_zone_features(
        datasets['danger_zone_raw'], 'danger_zone_raw'
    )


def engineer_overtaking_features(df, dataset_name="overtaking"):
    """Feature engineering for overtaking datasets"""
    if df is None:
        return None

    print(f"\n🔧 Engineering features for {dataset_name}...")
    features = df.copy()
    original_cols = len(features.columns)

    # 1. Speed advantage features
    if 'RelativeSpeed' in features.columns:
        features['speed_advantage'] = features['RelativeSpeed']
        features['speed_advantage_cat'] = pd.cut(
            features['RelativeSpeed'],
            bins=[-float('inf'), -5, 5, float('inf')],
            labels=['slower', 'equal', 'faster']
        )
        # One-hot encode
        features = pd.get_dummies(features, columns=['speed_advantage_cat'], prefix='speed')

    # 2. Time to reach leader
    if 'LeaderGap' in features.columns and 'RelativeSpeed' in features.columns:
        rel_speed_abs = features['RelativeSpeed'].abs()
        features['time_to_reach'] = features['LeaderGap'] / (rel_speed_abs / 3.6 + 0.1)
        features['time_to_reach_cat'] = pd.cut(
            features['time_to_reach'],
            bins=[0, 3, 6, 10, float('inf')],
            labels=['very_soon', 'soon', 'moderate', 'far']
        )
        features = pd.get_dummies(features, columns=['time_to_reach_cat'], prefix='reach')

    # 3. Lane change features
    if 'Lane' in features.columns and 'LanePrev' in features.columns:
        features['lane_changing'] = (features['Lane'] != features['LanePrev']).astype(int)

    # 4. Opposite lane risk
    if 'lane_opposite_change' in features.columns:
        features['opposite_lane_occupied'] = features['lane_opposite_change']

    # 5. Overtaking feasibility score
    feasibility_conditions = []
    if 'RelativeSpeed' in features.columns:
        features['overtaking_possible_speed'] = (features['RelativeSpeed'] > 5).astype(int)
        feasibility_conditions.append('overtaking_possible_speed')

    if 'LeaderGap' in features.columns:
        features['overtaking_possible_gap'] = (features['LeaderGap'] < 100).astype(int)
        feasibility_conditions.append('overtaking_possible_gap')

    if 'TTC' in features.columns:
        features['overtaking_safe_ttc'] = (features['TTC'] > 5).astype(int)
        feasibility_conditions.append('overtaking_safe_ttc')

    if 'lane_opposite_change' in features.columns:
        features['overtaking_lane_clear'] = (features['lane_opposite_change'] == 0).astype(int)
        feasibility_conditions.append('overtaking_lane_clear')

    # Combine all conditions
    if feasibility_conditions:
        features['overtaking_feasibility_score'] = (
                features[feasibility_conditions].sum(axis=1) / len(feasibility_conditions)
        )
        features['overtaking_recommended'] = (
                features['overtaking_feasibility_score'] > 0.7
        ).astype(int)

    # 6. TTC during overtaking
    if 'TTC' in features.columns:
        features['ttc_overtaking_risk'] = (features['TTC'] < 3).astype(int)

    # 7. Slope impact (if available)
    if 'Slope' in features.columns:
        features['steep_slope'] = (features['Slope'].abs() > 5).astype(int)

    print(f"   Added {len(features.columns) - original_cols} new features")
    print(f"   Total features: {len(features.columns)}")

    return features


# Apply to overtaking datasets
datasets['overtaking_pure_engineered'] = engineer_overtaking_features(
    datasets['overtaking_pure'], 'overtaking_pure'
)

for raw_name in ['overtaking_raw1', 'overtaking_raw2']:
    if datasets[raw_name] is not None:
        datasets[f'{raw_name}_engineered'] = engineer_overtaking_features(
            datasets[raw_name], raw_name
        )


def engineer_intersection_features(df, dataset_name="intersection"):
    """Feature engineering for intersection datasets"""
    if df is None:
        return None

    print(f"\n🔧 Engineering features for {dataset_name}...")
    features = df.copy()
    original_cols = len(features.columns)

    # 1. Approach speed features
    if 'Speed' in features.columns:
        features['approaching_slow'] = (features['Speed'] < 30).astype(int)
        features['approaching_moderate'] = ((features['Speed'] >= 30) &
                                            (features['Speed'] < 50)).astype(int)
        features['approaching_fast'] = (features['Speed'] >= 50).astype(int)

    # 2. Deceleration pattern (indicating stopping)
    if 'Acceleration' in features.columns:
        features['decelerating'] = (features['Acceleration'] < -1).astype(int)
        features['hard_deceleration'] = (features['Acceleration'] < -3).astype(int)

    # 3. Following behavior at intersection
    if 'LeaderGap' in features.columns:
        features['close_to_leader'] = (features['LeaderGap'] < 20).astype(int)

    # 4. Type encoding (if 'Type' column exists)
    if 'Type' in features.columns:
        # One-hot encode intersection type
        features = pd.get_dummies(features, columns=['Type'], prefix='intersection_type')

    # 5. Slope impact
    if 'Slope' in features.columns:
        features['intersection_slope'] = features['Slope']

    # 6. Combined approach score (proxy for intersection likelihood)
    approach_score = 0
    if 'approaching_slow' in features.columns:
        approach_score += features['approaching_slow'] * 0.4
    if 'decelerating' in features.columns:
        approach_score += features['decelerating'] * 0.4
    if 'close_to_leader' in features.columns:
        approach_score += features['close_to_leader'] * 0.2

    if isinstance(approach_score, int) == False:  # If we added any components
        features['intersection_likelihood'] = approach_score

    print(f"   Added {len(features.columns) - original_cols} new features")
    print(f"   Total features: {len(features.columns)}")

    return features


# Apply to intersection dataset
datasets['intersection_engineered'] = engineer_intersection_features(
    datasets['intersection'], 'intersection'
)


def engineer_fusion_features(df, dataset_name="fusion"):
    """Complete feature engineering for the fusion dataset (Dataset 4)"""
    if df is None:
        return None

    print(f"\n🔧🔧🔧 Engineering features for {dataset_name} (YOUR MAIN DATASET)...")
    features = df.copy()
    original_cols = len(features.columns)

    # ===== 1. UNIVERSAL FEATURES (apply to all situations) =====

    # Time-based
    if 'Time' in features.columns:
        features['time_normalized'] = features['Time'] / features['Time'].max()

    # Speed-based
    features['speed_squared'] = features['Speed'] ** 2
    features['speed_category'] = pd.cut(
        features['Speed'],
        bins=[0, 30, 60, 90, 200],
        labels=['slow', 'moderate', 'fast', 'very_fast']
    )
    features = pd.get_dummies(features, columns=['speed_category'], prefix='speed')

    # TTC-based
    features['ttc_emergency'] = (features['TTC'] < 2).astype(int)
    features['ttc_warning'] = ((features['TTC'] >= 2) & (features['TTC'] < 4)).astype(int)
    features['ttc_safe'] = (features['TTC'] >= 4).astype(int)
    features['ttc_normalized'] = features['TTC'] / 10  # Normalize to 0-1 range
    features['ttc_inverse'] = 1 / (features['TTC'] + 0.01)

    # Gap features
    features['gap_ratio'] = features['LeaderGap'] / (features['SafeDistance'] + 0.01)
    features['gap_deficit'] = features['SafeDistance'] - features['LeaderGap']
    features['gap_deficit_normalized'] = features['gap_deficit'] / (features['SafeDistance'] + 0.01)
    features['dangerous_gap'] = (features['gap_ratio'] < 0.5).astype(int)
    features['critical_gap'] = (features['gap_ratio'] < 0.3).astype(int)

    # Relative speed
    features['closing_fast'] = (features['RelativeSpeed'] > 10).astype(int)
    features['closing_moderate'] = ((features['RelativeSpeed'] > 3) &
                                    (features['RelativeSpeed'] <= 10)).astype(int)
    features['separating'] = (features['RelativeSpeed'] < -3).astype(int)

    # Acceleration features
    features['hard_accel'] = (features['Acceleration'] > 2).astype(int)
    features['hard_decel'] = (features['Deceleration'] > 2).astype(int)
    features['smooth_driving'] = ((features['Acceleration'].abs() < 1) &
                                  (features['Deceleration'] < 1)).astype(int)

    # ===== 2. OVERTAKING-SPECIFIC FEATURES =====

    features['overtaking_speed_advantage'] = features['RelativeSpeed']
    features['overtaking_time_to_reach'] = features['LeaderGap'] / (features['RelativeSpeed'].abs() / 3.6 + 0.1)

    # Lane change indicators
    if 'LanePrev' in features.columns:
        features['lane_change_active'] = (features['Lane'] != features['LanePrev']).astype(int)

    if 'lane_opposite_change' in features.columns:
        features['opposite_lane_occupied'] = features['lane_opposite_change']

    # Overtaking opportunity score
    overtaking_conditions = [
        (features['RelativeSpeed'] > 5),
        (features['LeaderGap'] < 100),
        (features['TTC'] > 5),
        (features['gap_ratio'] > 0.3)
    ]

    if 'lane_opposite_change' in features.columns:
        overtaking_conditions.append((features['lane_opposite_change'] == 0))

    features['overtaking_opportunity_score'] = (
            sum([cond.astype(int) for cond in overtaking_conditions]) / len(overtaking_conditions)
    )

    # ===== 3. DANGER ZONE-SPECIFIC FEATURES =====

    # Binary flag features
    danger_flags = ['hard_brake', 'danger_accel_toward_leader',
                    'leader_stopped', 'no_reaction_to_stopped_leader']

    existing_flags = [f for f in danger_flags if f in features.columns]
    if existing_flags:
        features['danger_flag_count'] = features[existing_flags].sum(axis=1)
        features['any_danger_flag'] = (features['danger_flag_count'] > 0).astype(int)

    # Risk score calculation
    features['danger_risk_score'] = (
            features.get('hard_brake', 0) * 0.4 +
            features.get('ttc_emergency', 0) * 0.4 +
            features.get('danger_accel_toward_leader', 0) * 0.2 +
            features.get('leader_stopped', 0) * 0.2 -
            features.get('ttc_safe', 0) * 0.1
    )
    features['danger_risk_score'] = features['danger_risk_score'].clip(0, 1)

    # ===== 4. INTERSECTION-SPECIFIC FEATURES =====

    # Approach behavior
    features['intersection_approach_slow'] = (features['Speed'] < 30).astype(int)
    features['intersection_decelerating'] = (features['Deceleration'] > 1).astype(int)

    # Combined intersection likelihood
    features['intersection_likelihood'] = (
            features['intersection_approach_slow'] * 0.3 +
            features['intersection_decelerating'] * 0.3 +
            ((features['gap_ratio'] > 0.8).astype(int)) * 0.2 +  # Not following closely
            features['smooth_driving'] * 0.2
    )

    # ===== 5. SCENARIO-BASED FEATURES =====

    if 'Scenario' in features.columns:
        features = pd.get_dummies(features, columns=['Scenario'], prefix='scenario')

    if 'AccidentTypes' in features.columns:
        features['has_accident'] = (features['AccidentTypes'].notna() &
                                    (features['AccidentTypes'] != '')).astype(int)

    # ===== 6. HIGH-LEVEL COMPOSITE FEATURES =====

    # Overall situation assessment
    features['likely_overtaking'] = (
            (features['overtaking_opportunity_score'] > 0.6) &
            (features['SafeUnsafeOvertaking'].notna() if 'SafeUnsafeOvertaking' in features.columns else True)
    ).astype(int)

    features['likely_danger_zone'] = (
        (features['danger_risk_score'] > 0.5)
    ).astype(int)

    features['likely_intersection'] = (
        (features['intersection_likelihood'] > 0.5)
    ).astype(int)

    # Multi-hot encoding for final classification
    if all(col in features.columns for col in
           ['SafeUnsafeOvertaking', 'SafeUnsafeCrossing', 'SafeDangerZone']):
        features['situation_overtaking'] = features['SafeUnsafeOvertaking'].fillna(0)
        features['situation_intersection'] = features['SafeUnsafeCrossing'].fillna(0)
        features['situation_dangerzone'] = features['SafeDangerZone'].fillna(0)

        # Multi-class label (mutually exclusive - choose primary situation)
        conditions = [
            (features['situation_dangerzone'] == 1),
            (features['situation_overtaking'] == 1),
            (features['situation_intersection'] == 1)
        ]
        choices = [2, 1, 3]  # 2=danger, 1=overtaking, 3=intersection
        features['situation_label'] = np.select(conditions, choices, default=0)

    print(f"   Added {len(features.columns) - original_cols} new features")
    print(f"   Total features: {len(features.columns)}")

    return features


# Apply to fusion dataset (THIS IS YOUR MAIN TARGET)
if datasets['fusion'] is not None:
    datasets['fusion_engineered'] = engineer_fusion_features(datasets['fusion'], 'fusion')
print("\n" + "=" * 50)
print("💾 Saving engineered datasets...")

for name, df in datasets.items():
    if df is not None and '_engineered' in name:
        output_path = f"engineered_datasets/{name}.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Saved {name}: {df.shape[0]} rows, {df.shape[1]} cols")

print("=" * 50)
print("🎉 Feature engineering pipeline complete!")
print("\n📊 FEATURE ENGINEERING SUMMARY")
print("=" * 50)

for name, df in datasets.items():
    if df is not None and '_engineered' in name:
        print(f"\n{name.upper()}:")
        print(f"  - Rows: {df.shape[0]}")
        print(f"  - Columns: {df.shape[1]}")

        # Show feature types
        feature_types = df.dtypes.value_counts()
        print(f"  - Data types: {dict(feature_types)}")

        # Show label columns if they exist
        label_cols = [col for col in df.columns if 'label' in col or 'Safe' in col or 'situation' in col]
        if label_cols:
            print(f"  - Label columns: {label_cols[:5]}...")

print("\n" + "=" * 50)
print("✅ Pipeline complete! Check the 'engineered_datasets' folder.")