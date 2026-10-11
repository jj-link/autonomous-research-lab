"""Command-line entrypoint. JSON output is stable across interactive and automated use."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .models import LabError, load_manifest
from .store import Store


def main(argv=None):
    parser = argparse.ArgumentParser(prog="arl", description="Recorded computational research")
    parser.add_argument("--root", type=Path, default=Path(".arl"), help="Research state directory (default: .arl)")
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit", help="Snapshot an experiment's design, settings, and source")
    submit.add_argument("manifest", type=Path)
    listing = commands.add_parser("list", help="List saved experiments or execution attempts")
    listing.add_argument("--attempts", action="store_true")
    show = commands.add_parser("show", help="Inspect an experiment or execution attempt")
    show.add_argument("id")
    for name in ("events", "result"):
        command = commands.add_parser(name)
        command.add_argument("id")
    args = parser.parse_args(argv)
    try:
        store = Store(args.root)
        if args.command == "submit":
            value = store.submit(load_manifest(args.manifest))
        elif args.command == "list":
            value = store.list_attempts() if args.attempts else store.list_experiments()
        elif args.command == "show":
            if args.id.startswith("exp_"):
                value = store.get_experiment(args.id)
                value["attempts"] = store.list_attempts(args.id)
            else:
                value = store.get_attempt(args.id)
        elif args.command == "events":
            value = store.events(args.id)
        else:
            value = store.get_attempt(args.id)["result"]
        print(json.dumps(value, indent=2, allow_nan=False))
        return 0
    except (LabError, OSError) as exc:
        print(f"arl: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
