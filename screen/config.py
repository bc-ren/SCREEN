import json
from pathlib import Path

DATASETS = ("AWA2", "CUB", "SUN")
PROFILES = ("Best-H", "Best-CZSL")
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def load_config(dataset, profile):
    if dataset not in DATASETS or profile not in PROFILES:
        raise ValueError("Unknown dataset or checkpoint profile")
    values = json.loads((CONFIGS / f"{dataset}.json").read_text())
    selected = values[profile]
    if "same_as" in selected:
        selected = values[selected["same_as"]]
    return {**values["shared"], **selected}
