import argparse
import json
from pathlib import Path

from .config import DATASETS, PROFILES, load_config


def main():
    parser = argparse.ArgumentParser(prog="python -m screen")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("config", "train", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("dataset", choices=DATASETS)
        sub.add_argument("profile", choices=PROFILES)
        if name != "config":
            sub.add_argument("--data", required=True, type=Path)
            sub.add_argument("--weights", required=True, type=Path)
            sub.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.dataset, args.profile)
    if args.command == "config":
        print(json.dumps(cfg, indent=2))
        return
    from .data import configure_device
    from .inference import evaluate
    from .train import train

    configure_device()
    if args.command == "train":
        train(args.data, args.weights, cfg, args.device)
    else:
        result = evaluate(args.data, args.weights, cfg, args.device)
        print(json.dumps({key: round(100 * value, 1) for key, value in result.items()}))


if __name__ == "__main__":
    main()
