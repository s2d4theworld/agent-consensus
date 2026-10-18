#!/usr/bin/env python3
# agent-consensus — Proposer vs Critic debate until they reach consensus,
# judged each round by a neutral judge. Python stdlib only.
#
# The code lives in sibling modules:
#   ac_errors.py     — shared exception types
#   ac_config.py     — paths, constants, config file I/O, agent installer
#   ac_transports.py — HTTP helpers and both transports (opencode / openai)
#   ac_engine.py     — the debate engine (round loop, sessions, judge)
#   ac_ui.py         — prompts, setup wizard, menu status
#
# Usage:
#   python agent_consensus.py               # interactive menu
#   python agent_consensus.py setup         # configure only
#   python agent_consensus.py chat          # same as no arguments (menu)
#   python agent_consensus.py install-agent # install the websearch agent
#   python agent_consensus.py --version     # print the version

import argparse
import sys

# model output may contain characters the console codepage can't encode
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from ac_config import Config, ensure_agent_installed, load_config, validate_config
from ac_engine import run_discussion
from ac_errors import ConsensusError
from ac_ui import banner, print_status, setup_wizard

__version__ = "1.0.0"

def safe_run_discussion(cfg: Config, topic: str | None = None) -> bool:
    # True = run completed, False = it was aborted/interrupted (the menu
    # stays interactive either way)
    try:
        run_discussion(cfg, topic=topic)
        return True
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        return False
    except EOFError:
        print("\n  Discussion aborted: no input available (stdin closed?)")
        return False
    except ConsensusError as e:
        print(f"\n  Discussion aborted: {e}")
        return False
    except Exception as e:
        print(f"\n  Discussion aborted: {type(e).__name__}: {e}")
        return False

def run_wizard() -> None:
    # EOF/Ctrl+C mid-setup must not traceback; nothing is saved on abort
    try:
        setup_wizard()
    except (EOFError, KeyboardInterrupt):
        print("\n  Setup aborted.")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent_consensus",
        description="Proposer vs Critic debate until consensus, judged each round.")
    parser.add_argument(
        "command", nargs="?", default="chat",
        help="chat = interactive menu (default), setup = configure, "
             "install-agent = install the websearch agent")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    return parser

def menu() -> None:
    while True:
        banner()
        cfg = load_config()
        err = validate_config(cfg) if cfg else None
        print_status(cfg, err)
        print()
        print("  [1] Setup / change configuration")
        print("  [2] Start discussion")
        print("  [3] Quit")
        print()
        try:
            choice = input("  Choose (1-3): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if choice == "1":
            run_wizard()
        elif choice == "2":
            cfg = load_config()
            err = validate_config(cfg) if cfg else None
            if not cfg or err:
                print(f"  Cannot start: {err or 'no config'}. Run setup first.")
            else:
                safe_run_discussion(cfg)
        elif choice == "3":
            break
        else:
            print("  Invalid choice.")

def main() -> None:
    # lenient parsing: unknown flags/args are ignored; anything that is not
    # `setup` or `install-agent` (including `chat`) opens the interactive menu
    args, _ = build_parser().parse_known_args()
    if args.command == "setup":
        run_wizard()
    elif args.command == "install-agent":
        ensure_agent_installed()
    else:
        menu()

if __name__ == "__main__":
    main()
