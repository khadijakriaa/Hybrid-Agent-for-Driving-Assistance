"""Train and persist one model per driving situation.

Usage from project root:
    python core/train_scenario_models.py
"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.agent.scenario_model_registry import ScenarioModelRegistry


def main() -> None:
    data_dir = REPO_ROOT / "data"

    registry = ScenarioModelRegistry()
    report = registry.train_and_save_all(data_dir=data_dir)

    print("Saved scenario models:")
    for name, info in report.items():
        print(
            f"- {name}: samples={info['samples']}, features={info['features']}, path={info['path']}"
        )


if __name__ == "__main__":
    main()
