"""Internal process entrypoint; the public interface is `arl run`/`arl recover`."""
import argparse
from pathlib import Path

from .execution import supervise
from .store import Store


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    result = supervise(Store(args.root), args.attempt, recovery=args.recover)
    return 0 if result["state"] in {"completed", "stopped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
