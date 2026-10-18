# agent-consensus — end-to-end golden-output check.
#
# Replays a few complete scenarios (debates, menu, wizard, abort path) against
# the local fake server from tests/helpers.py and compares the full normalized
# result (stdout, exception, transcript file, recorded requests) with the
# checked-in goldens in tools/golden/.
#
#   python tools/golden_check.py            # verify against the goldens
#   python tools/golden_check.py --update   # rewrite the goldens (intentional changes)
#
# Normalized away: temp paths, timestamps, durations, ephemeral fake-server
# ports. Everything else must match byte for byte.
import contextlib
import glob
import io
import json
import os
import re
import shutil
import sys
import tempfile
from typing import Any

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
GOLDEN_DIR = os.path.join(TOOLS_DIR, "golden")

# the project imports below come after this path setup, not at the top of the
# file; importing helpers also puts ROOT on sys.path
sys.path.insert(0, os.path.join(ROOT, "tests"))
from helpers import FakeServer, oc_cfg

import ac_config
import ac_ui
import agent_consensus
from ac_engine import parse_verdict, run_discussion

SCENARIOS: dict[str, Any] = {}


def scenario(fn):
    SCENARIOS[fn.__name__] = fn
    return fn


class Ctx:
    def __init__(self, root):
        self.root = root
        self.transcripts = os.path.join(root, "transcripts")
        self.agents = os.path.join(root, "agents")
        self.config_path = os.path.join(root, "config.json")
        os.makedirs(self.transcripts, exist_ok=True)
        self.srv = FakeServer()

    def stop(self):
        self.srv.stop()
        shutil.rmtree(self.root, ignore_errors=True)


# ---------- scenarios ----------
@scenario
def s1_consensus_round3(ctx):
    ctx.srv.verdicts[:] = ["NO - positions differ.", "YES - both agree now."]
    run_discussion(oc_cfg(ctx.srv.base, max_rounds=3), topic="S1 topic")


@scenario
def s2_no_consensus(ctx):
    ctx.srv.verdicts[:] = ["NO - still far apart."]
    run_discussion(oc_cfg(ctx.srv.base, max_rounds=1), topic="S2 topic")


@scenario
def s3_missing_agent(ctx):
    ctx.srv.agent_names.clear()
    ctx.srv.verdicts[:] = ["YES - agree."]
    run_discussion(oc_cfg(ctx.srv.base, max_rounds=1), topic="S3 topic")


@scenario
def s4_unreachable(_ctx):
    agent_consensus.safe_run_discussion(
        oc_cfg("http://127.0.0.1:1"), topic="S4 topic")


@scenario
def s5_no_judge(ctx):
    ctx.srv.verdicts[:] = ["YES - agree."]
    run_discussion(oc_cfg(ctx.srv.base, with_judge=False, max_rounds=1),
                   topic="S5 topic")


@scenario
def s6_menu(ctx):
    with open(ctx.config_path, "w", encoding="utf-8") as f:
        json.dump(oc_cfg(ctx.srv.base), f)
    old_stdin, old_argv = sys.stdin, sys.argv
    sys.stdin = io.StringIO("3\n")
    sys.argv = ["agent_consensus.py"]
    try:
        agent_consensus.main()
    finally:
        sys.stdin, sys.argv = old_stdin, old_argv


@scenario
def s7_wizard_opencode(ctx):
    old_stdin = sys.stdin
    # 12 blank answers (4 prompts x 3 roles) + transport + the trailing "3"
    # which is the max_rounds answer, not a menu quit
    sys.stdin = io.StringIO("opencode\n" + "\n" * 12 + "3\n")
    try:
        ac_ui.setup_wizard()
    finally:
        sys.stdin = old_stdin
    with open(ctx.config_path, "r", encoding="utf-8") as f:
        print("CONFIG:", json.dumps(json.load(f), sort_keys=True))


@scenario
def s8_parse_verdict(_ctx):
    cases = ["YES - agree", "yes", "NO.", "  No because", "maybe yes later",
             "Absolutely not", "", "NOtable difference", "YESmen not", "Final: YES"]
    for c in cases:
        print(f"{c!r} -> {parse_verdict(c)}")


# ---------- runner ----------
def normalize(text, root):
    text = text.replace(root, "<TMP>")
    text = re.sub(r"Duration     : \d+ sec", "Duration     : <N> sec", text)
    text = re.sub(r"DURATION: \d+s", "DURATION: <N>s", text)
    text = re.sub(r"\d{8}_\d{6}", "<TS>", text)
    text = re.sub(r"START: .*", "START: <T>", text)
    text = re.sub(r"127\.0\.0\.1:\d{5,}", "127.0.0.1:<P>", text)  # ephemeral port
    return text


def run_scenario(name, fn):
    ctx = Ctx(tempfile.mkdtemp(prefix=f"golden_{name}_"))
    saved = (ac_config.TRANSCRIPT_DIR, ac_config.CONFIG_PATH, ac_config.AGENT_SRC,
             ac_config.OPENCODE_AGENTS_DIR)
    ac_config.TRANSCRIPT_DIR = ctx.transcripts
    ac_config.CONFIG_PATH = ctx.config_path
    ac_config.AGENT_SRC = os.path.join(ROOT, ".opencode", "agents", "debater.md")
    ac_config.OPENCODE_AGENTS_DIR = ctx.agents
    buf = io.StringIO()
    err = None
    try:
        with contextlib.redirect_stdout(buf):
            try:
                fn(ctx)
            except SystemExit:
                pass
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
    finally:
        (ac_config.TRANSCRIPT_DIR, ac_config.CONFIG_PATH, ac_config.AGENT_SRC,
         ac_config.OPENCODE_AGENTS_DIR) = saved
    try:
        tfiles = sorted(glob.glob(os.path.join(ctx.transcripts, "*.txt")))
        transcript = ""
        if tfiles:
            with open(tfiles[0], "r", encoding="utf-8") as f:
                transcript = f.read()
        sessions = [normalize(s.get("title") or "?", ctx.root) for s in ctx.srv.sessions]
        msgs = [(m["sid"], m["messageID"], m["agent"], m["text"]) for m in ctx.srv.msgs]
        parts = [
            "=== STDOUT ===",
            normalize(buf.getvalue(), ctx.root).rstrip("\n"),
            "=== EXC ===",
            err or "",
            "=== TRANSCRIPT ===",
            normalize(transcript, ctx.root).rstrip("\n"),
            "=== REC ===",
            f"sessions: {sessions}",
            f"deletes: {len(ctx.srv.deletes)}",
            f"msgs: {msgs}",
        ]
        return "\n".join(parts) + "\n"
    finally:
        ctx.stop()


def main() -> int:
    update = "--update" in sys.argv[1:]
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    failed = []
    for name, fn in SCENARIOS.items():
        try:
            rendered = run_scenario(name, fn)
        except Exception as e:
            print(f"FAIL {name}: harness error {type(e).__name__}: {e}")
            failed.append(name)
            continue
        path = os.path.join(GOLDEN_DIR, f"{name}.txt")
        if update:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(rendered)
            print(f"UPDATED {name}")
            continue
        if not os.path.exists(path):
            print(f"FAIL {name}: missing golden {path} (run with --update)")
            failed.append(name)
            continue
        with open(path, "r", encoding="utf-8") as f:
            expected = f.read()
        if rendered == expected:
            print(f"PASS {name}")
            continue
        failed.append(name)
        print(f"FAIL {name}: output differs from golden")
        for i, (a, b) in enumerate(zip(expected.splitlines(), rendered.splitlines())):
            if a != b:
                print(f"   line {i}: golden={a!r}")
                print(f"   line {i}: actual={b!r}")
                break
        else:
            print("   (line count differs)")
    if not update:
        print("\nALL PASS" if not failed else f"\nFAILED: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
