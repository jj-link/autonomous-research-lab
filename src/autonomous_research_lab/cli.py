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
    config = commands.add_parser("configure", help="Select an existing local execution image and allocated GPU")
    config.add_argument("--image", required=True)
    config.add_argument("--gpu")
    config.add_argument("--gpu-mode", choices=("cdi", "wsl"), default="cdi")
    commands.add_parser("doctor", help="Inspect actual worker resources without changing host settings")
    for name in ("run", "stop"):
        command = commands.add_parser(name)
        command.add_argument("id")
        command.add_argument("--wait", action="store_true")
    waiting = commands.add_parser("wait", help="Wait for a recorded attempt without re-executing it")
    waiting.add_argument("id")
    waiting.add_argument("--timeout", type=float)
    recovery = commands.add_parser("recover", help="Reconcile existing attempts, never blindly resubmit them")
    recovery.add_argument("--wait", action="store_true")
    logs = commands.add_parser("logs", help="Read the last 64 KiB of a container log")
    logs.add_argument("id")
    args = parser.parse_args(argv)
    try:
        store = Store(args.root)
        exit_code = 0
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
        elif args.command == "result":
            value = store.get_attempt(args.id)["result"]
        else:
            from . import execution
            if args.command == "configure":
                value = execution.configure(store, args.image, args.gpu, args.gpu_mode)
            elif args.command == "doctor":
                value = execution.doctor(store)
            elif args.command == "run":
                value = execution.run(store, args.id, wait=args.wait)
                exit_code = int(args.wait and value["state"] != "completed")
            elif args.command == "wait":
                if args.timeout is not None and args.timeout <= 0:
                    raise LabError("--timeout must be positive")
                value = execution.wait_for_attempt(store, args.id, timeout=args.timeout)
                exit_code = int(value["state"] != "completed")
            elif args.command == "stop":
                value = execution.stop(store, args.id, wait=args.wait)
            elif args.command == "recover":
                value = execution.recover(store, wait=args.wait)
            else:
                store.get_attempt(args.id)
                path = store.attempt_dir(args.id) / "console.log"
                if path.exists():
                    with path.open("rb") as stream:
                        size = stream.seek(0, 2)
                        stream.seek(max(0, size - 65536))
                        text = stream.read(65536).decode("utf-8", errors="replace")
                    value = {"text": text, "truncated": size > 65536}
                else:
                    value = {"text": "", "truncated": False}
        print(json.dumps(value, indent=2, allow_nan=False))
        return exit_code
    except (LabError, OSError) as exc:
        print(f"arl: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
