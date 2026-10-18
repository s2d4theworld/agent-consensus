# agent-consensus — the debate engine: availability checks, prompts,
# sessions, the round loop and the judge.
import os
import re
import socket
import time
import urllib.parse
from typing import TypedDict

import ac_config
from ac_config import (
    AgentCfg,
    Config,
    DEFAULT_MAX_ROUNDS,
    ROLES,
    ROLE_CRITIC,
    ROLE_JUDGE,
    ROLE_PROPOSER,
    SESSION_TITLE_PREFIX,
    TIMEOUT_PREFLIGHT,
    is_opencode,
    transport_of,
)
from ac_errors import ServerUnreachable
from ac_transports import (
    chat_completion,
    oc_agent_list,
    oc_base_url,
    oc_delete,
    oc_send,
    oc_session_create,
    oc_sweep,
)
# engine -> ui (no cycle: ac_ui does not import the engine); the topic prompt
# lives in ac_ui so it stays AFTER the fail-fast server checks
from ac_ui import input_required

# shared shapes
Turns = list[tuple[str, str, str]]           # (agent_key, name, text)
Available = dict[str, set[str] | None]       # base_url -> agent names (None = unreachable)
EffectiveAgents = dict[str, str | None]      # role -> restricted agent actually in use

# ---------- Consensus checking (stateless transport) ----------
JUDGE_SYSTEM = (
    "You are a neutral judge. Your job is to decide whether the TWO agents "
    "below have reached CONSENSUS on the given topic. "
    "Consensus means both agents agree on the same position/decision "
    "and explicitly state their agreement (not just a comment). "
    "Start your answer with one word: YES or NO, followed by a one-sentence reason."
)

def parse_verdict(out: str) -> bool:
    # judge is told to START with YES/NO: prefer the leading token,
    # fall back to the first YES/NO word anywhere
    t = out.strip().upper()
    m = re.match(r"[^A-Z]*(YES|NO)\b", t)
    if m:
        return m.group(1) == "YES"
    m = re.search(r"\b(YES|NO)\b", t)
    if m:
        return m.group(1) == "YES"
    return False

def check_consensus(cfg: Config, topic: str, transcript: str,
                    judge_key: str) -> tuple[bool, str]:
    judge_cfg = cfg["agents"][judge_key]
    judge_msgs = [
        {"role": "user", "content": f"TOPIC: {topic}\n\nHere is the transcript:\n\n{transcript}"}
    ]
    out = chat_completion(judge_key, judge_cfg, JUDGE_SYSTEM, judge_msgs)
    return parse_verdict(out), out

# ---------- Discussion engine ----------
def format_transcript(turns: Turns) -> str:
    # turns: list of (agent_key, name, text) -> string for judge
    return "\n\n".join(f"[{name}]:\n{text}" for _, name, text in turns)

def parse_max_rounds(cfg: Config) -> int:
    try:
        return max(1, int(cfg.get("max_rounds", DEFAULT_MAX_ROUNDS)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_ROUNDS

def check_servers(cfg: Config) -> Available:
    # sweep orphan sessions; fail fast if a stateful server is unreachable.
    # returns {base_url: set(agent names)} for the reachable opencode servers
    bases = sorted({oc_base_url(a) for a in cfg["agents"].values()
                    if is_opencode(a)})
    available: Available = {}
    for base in bases:
        oc_sweep(base)
        available[base] = oc_agent_list(base)
    down = [b for b in bases if available[b] is None]
    if down:
        raise ServerUnreachable(f"cannot reach {', '.join(down)} - is `opencode serve` running?")
    return available

def check_stateless_endpoints(cfg: Config) -> None:
    # fail fast when a stateless endpoint refuses TCP connections; the first
    # model call would otherwise burn ~2 minutes of connection retries.
    # other connection errors only warn (a proxy may still make it work)
    seen = set()
    for a in cfg["agents"].values():
        if is_opencode(a):
            continue
        url = a["api_url"]
        if url in seen:
            continue
        seen.add(url)
        parts = urllib.parse.urlsplit(url)
        if not parts.hostname:
            continue
        try:
            port = parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError:
            continue  # nonnumeric port: the request itself will complain later
        try:
            socket.create_connection((parts.hostname, port),
                                     timeout=TIMEOUT_PREFLIGHT).close()
        except OSError as e:
            if isinstance(e, ConnectionRefusedError) or getattr(e, "winerror", None) == 10061:
                raise ServerUnreachable(f"cannot reach {url} - connection refused")
            print(f"  [Warn] {url} not reachable directly ({e}) - trying anyway")

def resolve_restricted_agents(cfg: Config, available: Available) -> EffectiveAgents:
    # per-role restricted (websearch) agent to use; None where unavailable
    agent_eff: EffectiveAgents = {}
    warned = set()
    for key, a in cfg["agents"].items():
        name = a.get("agent")
        if not name or not is_opencode(a):
            agent_eff[key] = None
            continue
        base = oc_base_url(a)
        if name in (available.get(base) or set()):
            agent_eff[key] = name
            continue
        if (base, name) not in warned:
            print(f"  [Warn] agent '{name}' not on {base} - using the default "
                  "agent without tools.")
            print("  [Warn] run: python agent_consensus.py install-agent, "
                  "then restart opencode serve.")
            warned.add((base, name))
        agent_eff[key] = None
    return agent_eff

def build_prompts(topic: str, proposer: AgentCfg, critic: AgentCfg,
                  agent_eff: EffectiveAgents) -> tuple[str, str]:
    def tools_line(key):
        if agent_eff.get(key):
            return ("You may use the websearch tool to verify facts when needed, "
                    "but no other tools; otherwise reply with text only. Language: English.")
        return "Do not use any tools; reply with text only. Language: English."

    proposer_sys = (
        f"You are an agent named {proposer['name']} — the PROPOSER, tasked with "
        f"DEFENDING this topic: \"{topic}\". Open by supporting the topic and "
        "give its strongest arguments. Read the discussion history, keep or "
        "adjust your position consistently, and rebut the opponent's arguments "
        "substantively. If the opponent is convincing enough, state explicit "
        f"agreement. {tools_line(ROLE_PROPOSER)}"
    )
    critic_sys = (
        f"You are an agent named {critic['name']} — the CRITIC, tasked with "
        f"CHALLENGING this topic: \"{topic}\". Open with an OPPOSING position "
        "(reject, doubt, or lower its claims). Read the opponent's arguments, "
        "look for logic/data gaps, and rebut substantively. Only state explicit "
        "agreement if the opponent's arguments are truly unassailable after "
        "scrutiny. Don't argue just for the sake of arguing — but don't agree "
        f"without scrutiny. {tools_line(ROLE_CRITIC)}"
    )
    return proposer_sys, critic_sys

class SessionState(TypedDict):
    # OpencodeSessions bookkeeping for one session role
    id: str | None
    started: bool
    sent: int
    cfg_key: str

class OpencodeSessions:
    # one lazily-created opencode session per session role (id, started, sent,
    # cfg key). The judge uses its own session role so a judge-less config
    # (judging falls back to the proposer's agent) still gets an isolated
    # judging session.
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.state: dict[str, SessionState] = {}

    def ensure(self, sess_key: str, cfg_key: str, title: str) -> SessionState:
        st = self.state.setdefault(sess_key, {"id": None, "started": False,
                                              "sent": 0, "cfg_key": cfg_key})
        if st["id"] is None:
            st["id"] = oc_session_create(self.cfg["agents"][cfg_key], title)
        return st

    def close_all(self) -> None:
        for st in self.state.values():
            if st["id"]:
                oc_delete(self.cfg["agents"][st["cfg_key"]], st["id"])
        self.state.clear()

def _print_turn(name: str, model: str, text: str) -> None:
    print(f"\n  ===== [{name}] ({model}) =====")
    print(text)
    print()

class Discussion:
    # one debate run: owns the turns, sessions, restricted-agent resolution
    # and the transcript file. run_discussion() wires the checks and prompts
    # around it.
    def __init__(self, cfg: Config, topic: str, max_rounds: int,
                 available: Available) -> None:
        self.cfg = cfg
        self.topic = topic
        self.max_rounds = max_rounds
        self.proposer = cfg["agents"][ROLE_PROPOSER]
        self.critic = cfg["agents"][ROLE_CRITIC]
        self.judge_key = (ROLE_JUDGE if isinstance(cfg["agents"].get(ROLE_JUDGE), dict)
                          else ROLE_PROPOSER)
        self.judge = cfg["agents"][self.judge_key]
        self.agent_eff = resolve_restricted_agents(cfg, available)
        self.proposer_sys, self.critic_sys = build_prompts(
            topic, self.proposer, self.critic, self.agent_eff)
        self.sessions = OpencodeSessions(cfg)
        self.turns: Turns = []
        self.ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(ac_config.TRANSCRIPT_DIR, f"discussion_{self.ts}.txt")
        self.initial_msg = f"Discussion topic: {topic}\nStart the discussion now."

    def start(self) -> None:
        # [Model]/[Judge]/[Transport] lines + the transcript header
        self.started_at = time.time()
        print(f"\n  [Model] {self.proposer['name']} ({self.proposer['model']})  vs  "
              f"{self.critic['name']} ({self.critic['model']})")
        print(f"  [Judge] {self.judge['name']} ({self.judge['model']})")
        print(f"  [Transport] {', '.join(sorted({transport_of(a) for a in (self.proposer, self.critic, self.judge)}))}\n")
        with open(self.path, "w", encoding="utf-8") as tf:
            tf.write(f"TOPIC: {self.topic}\nSTART: {time.ctime()}\n")
            tf.write(f"JUDGE: {self.judge['name']} ({self.judge['model']})\n{'='*60}\n\n")

    def full_view(self, agent_key: str) -> list[dict[str, str]]:
        # stateless transport view: own turns = assistant, opponent = user
        msgs = [{"role": "user", "content": self.initial_msg}]
        for key, name, text in self.turns:
            if key == agent_key:
                msgs.append({"role": "assistant", "content": text})
            else:
                msgs.append({"role": "user", "content": f"[{name}]: {text}"})
        return msgs

    def respond(self, key: str, sys_prompt: str) -> str:
        a = self.cfg["agents"][key]
        if not is_opencode(a):
            return chat_completion(key, a, sys_prompt, self.full_view(key))
        # stateful: send only the turns this session hasn't seen yet
        st = self.sessions.ensure(key, key, f"{SESSION_TITLE_PREFIX}{key} {self.ts}")
        parts = []
        if not st["started"]:
            parts.append(self.initial_msg)
        for k2, n2, t2 in self.turns[st["sent"]:]:
            if k2 != key:
                parts.append(f"[{n2}]: {t2}")
        out = oc_send(a, st["id"], sys_prompt, "\n\n".join(parts),
                      agent_name=self.agent_eff.get(key), turn_index=st["sent"])
        st["started"] = True
        st["sent"] = len(self.turns)
        return out

    def judge_now(self) -> tuple[bool, str]:
        # stateless: full transcript each round. stateful: first call = full
        # transcript, later calls = only the new exchange
        if not is_opencode(self.cfg["agents"][self.judge_key]):
            return check_consensus(self.cfg, self.topic, format_transcript(self.turns),
                                   self.judge_key)
        st = self.sessions.ensure(ROLE_JUDGE, self.judge_key,
                                  f"{SESSION_TITLE_PREFIX}judge {self.ts}")
        if not st["started"]:
            text = (f"TOPIC: {self.topic}\n\nHere is the transcript:\n\n"
                    f"{format_transcript(self.turns)}")
        else:
            text = "New exchange:\n\n" + format_transcript(self.turns[st["sent"]:])
        out = oc_send(self.judge, st["id"], JUDGE_SYSTEM, text,
                      agent_name=self.agent_eff.get(self.judge_key),
                      turn_index=st["sent"])
        st["started"] = True
        st["sent"] = len(self.turns)
        return parse_verdict(out), out

    def run(self) -> tuple[bool, int, str]:
        # the round loop; returns (consensus, rounds_done, judge_note)
        consensus = False
        rounds_done = 0
        judge_note = ""
        try:
            for rnd in range(1, self.max_rounds + 1):
                rounds_done = rnd
                print(f"  [Round {rnd}] {self.proposer['name']} is thinking...", flush=True)
                a_text = self.respond(ROLE_PROPOSER, self.proposer_sys)
                self.turns.append((ROLE_PROPOSER, self.proposer["name"], a_text))
                _print_turn(self.proposer["name"], self.proposer["model"], a_text)

                print(f"  [Round {rnd}] {self.critic['name']} is evaluating...", flush=True)
                b_text = self.respond(ROLE_CRITIC, self.critic_sys)
                self.turns.append((ROLE_CRITIC, self.critic["name"], b_text))
                _print_turn(self.critic["name"], self.critic["model"], b_text)

                self._append_round(rnd, a_text, b_text)

                # check consensus (round 1: opening positions can't be consensus yet)
                if rnd == 1 and self.max_rounds > 1:
                    print("  [Judge] skipped in round 1\n")
                else:
                    try:
                        ok, note = self.judge_now()
                        judge_note = note
                        print(f"  [Judge: {self.judge['name']} "
                              f"({self.judge['model']})] {note}\n")
                        if ok:
                            consensus = True
                            break
                    except Exception as e:
                        print(f"  [Judge] failed to evaluate: {e}\n")
        finally:
            self.sessions.close_all()
        return consensus, rounds_done, judge_note

    def finish(self, consensus: bool, rounds_done: int, judge_note: str) -> None:
        # final summary on screen + transcript footer
        elapsed = time.time() - self.started_at
        status = "CONSENSUS" if consensus else "NO CONSENSUS (hit max rounds)"
        print("=" * 60)
        print(f"  Final status : {status}")
        print(f"  Duration     : {elapsed:.0f} sec, {rounds_done} rounds")
        print(f"  Transcript   : {self.path}")
        print("=" * 60)
        with open(self.path, "a", encoding="utf-8") as tf:
            tf.write(f"{'='*60}\nSTATUS: {status}\n")
            tf.write(f"JUDGE NOTE: {judge_note}\nDURATION: {elapsed:.0f}s\n")

    def _append_round(self, rnd: int, a_text: str, b_text: str) -> None:
        with open(self.path, "a", encoding="utf-8") as tf:
            tf.write(f"--- Round {rnd} ---\n")
            tf.write(f"[{self.proposer['name']}]\n{a_text}\n\n")
            tf.write(f"[{self.critic['name']}]\n{b_text}\n\n")

def run_discussion(cfg: Config, topic: str | None = None) -> None:
    os.makedirs(ac_config.TRANSCRIPT_DIR, exist_ok=True)
    available = check_servers(cfg)
    check_stateless_endpoints(cfg)

    extra_roles = sorted(k for k in cfg["agents"] if k not in ROLES)
    if extra_roles:
        print(f"  [Warn] ignoring unknown agent role(s): {', '.join(extra_roles)}")

    prompted = topic is None
    topic = topic or input_required("Discussion topic: ")
    max_rounds = parse_max_rounds(cfg)
    print(f"\n  Topic: {topic}")
    print(f"  Max rounds: {max_rounds}\n")

    d = Discussion(cfg, topic, max_rounds, available)
    d.start()
    consensus, rounds_done, judge_note = d.run()
    d.finish(consensus, rounds_done, judge_note)

    if prompted:
        try:
            input("\n  Press Enter to return to the menu...")
        except (EOFError, KeyboardInterrupt):
            # redirected stdin hit EOF or Ctrl+C at the last prompt: the run
            # already finished, so don't let it look aborted
            print()
