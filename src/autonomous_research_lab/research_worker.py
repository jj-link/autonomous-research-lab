"""Internal detached coordinator process; start via ResearchEngine or the CLI."""
import argparse
from pathlib import Path

from .research import ResearchEngine
from .store import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--program", required=True)
    args = parser.parse_args()
    program = ResearchEngine(Store(args.root)).work(args.program)
    return int(program["state"] in {"failed", "blocked"})


if __name__ == "__main__":
    raise SystemExit(main())
