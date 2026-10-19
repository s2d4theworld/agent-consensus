# agent-consensus — the debate engine: availability checks, prompts,
# sessions, the round loop and the judge.
import os
import re
import socket
import time
import urllib.parse
from typing import NamedTuple, TypedDict

import ac_config
from ac_config import (
    AgentCfg,
    Config,
    INSTALL_HINT,
    ROLES,
    ROLE_CRITIC,
    ROLE_JUDGE,
    ROLE_PROPOSER,
    SESSION_TITLE_PREFIX,
    TIMEOUT_PREFLIGHT,
    TS_FORMAT,
    coerce_max_rounds,
    is_opencode,
    role_agents,
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
class Turn(NamedTuple):
    # one recorded debate turn
    key: str
    name: str
    model: str
    text: str

class JudgeRound(NamedTuple):
    # one judging attempt; skipped (round 1) has attempted == failed == False
    consensus: bool = False
    note: str | None = None
    attempted: bool = False
    failed: bool = False

class RunSummary:
    # outcome of the round loop; run() reports it whether the loop finished
    # or an exception interrupted it mid-round
    def __init__(self) -> None:
        self.consensus = False
        self.rounds_done = 0
        self.judge_note = ""
        self.judged = False
        self.judge_failed = False

    def add_round(self, rnd: int, result: JudgeRound) -> bool:
        # record one judged round; True = consensus reached, stop the loop
        self.rounds_done = rnd
        self.judged = self.judged or result.attempted
        self.judge_failed = self.judge_failed or result.failed
        if result.note is not None:
            self.judge_note = result.note
        self.consensus = self.consensus or result.consensus
        return self.consensus

    @property
    def no_verdict(self) -> bool:
        # the judge never answered once: not a genuine no-consensus verdict
        return self.judge_failed and not self.judged

Turns = list[Turn]
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
    # judge is told to START with YES/NO; otherwise use the first whole-word
    # YES/NO anywhere in the answer (\b keeps "NOtable"/"YESmen" out)
    t = out.strip().upper()
    m = re.match(r"[^A-Z]*(YES|NO)\b", t) or re.search(r"\b(YES|NO)\b", t)
    return bool(m and m.group(1) == "YES")

def _judge_prompt(topic: str, transcript: str) -> str:
    return f"TOPIC: {topic}\n\nHere is the transcript:\n\n{transcript}"

def check_consensus(cfg: Config, topic: str, transcript: str,
                    judge_key: str) -> tuple[bool, str]:
    judge_cfg = cfg["agents"][judge_key]
    judge_msgs = [{"role": "user", "content": _judge_prompt(topic, transcript)}]
    out = chat_completion(judge_key, judge_cfg, JUDGE_SYSTEM, judge_msgs,
                          role=ROLE_JUDGE)
    return parse_verdict(out), out

# ---------- Discussion engine ----------
def format_transcript(turns: Turns) -> str:
    # turns: recorded debate turns -> string for judge
    return "\n\n".join(f"[{turn.name}]:\n{turn.text}" for turn in turns)

def parse_max_rounds(cfg: Config) -> int:
    return coerce_max_rounds(cfg.get("max_rounds"))

def check_servers(cfg: Config) -> Available:
    # sweep orphan sessions; fail fast if a stateful server is unreachable.
    # returns {base_url: set(agent names)} for the reachable opencode servers
    bases = sorted({oc_base_url(a) for _, a in role_agents(cfg)
                    if is_opencode(a)})
    available: Available = {}
    for base in bases:
        oc_sweep(base)
        available[base] = oc_agent_list(base)
    down = [b for b in bases if available[b] is None]
    if down:
        raise ServerUnreachable(f"cannot reach {', '.join(down)} - is `opencode serve` running?")
    return available

def _endpoint_target(url: str) -> tuple[str, int] | None:
    try:
        parts = urllib.parse.urlsplit(url)
        hostname = parts.hostname
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return None  # malformed URL/port: the request itself will complain later
    return (hostname, port) if hostname else None

def _connection_refused(err: OSError) -> bool:
    return isinstance(err, ConnectionRefusedError) or getattr(err, "winerror", None) == 10061

def check_stateless_endpoints(cfg: Config) -> None:
    # fail fast when a stateless endpoint refuses TCP connections; the first
    # model call would otherwise burn ~2 minutes of connection retries.
    # other connection errors only warn (a proxy may still make it work)
    seen = set()
    for _, a in role_agents(cfg):
        if is_opencode(a):
            continue
        url = a["api_url"]
        if url in seen:
            continue
        seen.add(url)
        target = _endpoint_target(url)
        if target is None:
            continue
        try:
            socket.create_connection(target, timeout=TIMEOUT_PREFLIGHT).close()
        except OSError as e:
            if _connection_refused(e):
                raise ServerUnreachable(f"cannot reach {url} - connection refused") from e
            print(f"  [Warn] {url} not reachable directly ({e}) - trying anyway")

def resolve_restricted_agents(cfg: Config, available: Available) -> EffectiveAgents:
    # per-role restricted (websearch) agent to use; None where unavailable
    agent_eff: EffectiveAgents = {}
    warned = set()
    for key, a in role_agents(cfg):
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
            print(f"  [Warn] {INSTALL_HINT}.")
            warned.add((base, name))
        agent_eff[key] = None
    return agent_eff

def build_prompts(topic: str, proposer: AgentCfg, critic: AgentCfg,
                  agent_eff: EffectiveAgents) -> dict[str, str]:
    def tools_line(key: str) -> str:
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
    return {ROLE_PROPOSER: proposer_sys, ROLE_CRITIC: critic_sys}

class SessionState(TypedDict):
    # OpencodeSessions bookkeeping for one session role; id is "" until the
    # session is created. sent = number of turns (Discussion.turns) the
    # session already covers; only mark_seen() advances it.
    id: str
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

    def ensure(self, sess_key: str, title: str,
               cfg_key: str | None = None) -> SessionState:
        # cfg_key is only needed when the session role differs from the agent
        # config it runs on (the judge falling back to a debater's config)
        cfg_key = cfg_key or sess_key
        st = self.state.setdefault(sess_key, {"id": "", "started": False,
                                              "sent": 0, "cfg_key": cfg_key})
        if not st["id"]:
            st["id"] = oc_session_create(self.cfg["agents"][cfg_key], title)
        return st

    def mark_seen(self, sess_key: str, turns_seen: int) -> None:
        # the single place the session state moves: a recorded turn means the
        # session saw the exchange (started) and the cursor counts every turn,
        # including the agent's own (its session already holds it as the
        # assistant message); stateless roles have no session -> no-op
        st = self.state.get(sess_key)
        if st is not None:
            st["started"] = True
            st["sent"] = turns_seen

    def close_all(self) -> None:
        for st in self.state.values():
            if st["id"]:
                oc_delete(self.cfg["agents"][st["cfg_key"]], st["id"])
        self.state.clear()

def _print_turn(turn: Turn) -> None:
    print(f"\n  ===== [{turn.name}] ({turn.model}) =====")
    print(turn.text)
    print()

class Transcript:
    # the run transcript: unique path, header write, append-only turn text
    def __init__(self) -> None:
        self.ts = time.strftime(TS_FORMAT)
        self.path = self._reserve()

    def _reserve(self) -> str:
        # exclusive create: a second run started in the same second must not
        # overwrite the earlier transcript
        base = os.path.join(ac_config.TRANSCRIPT_DIR, f"discussion_{self.ts}")
        path, n = base + ".txt", 2
        while True:
            try:
                with open(path, "x", encoding="utf-8"):
                    return path
            except FileExistsError:
                path, n = f"{base}_{n}.txt", n + 1

    def write(self, text: str, mode: str = "a") -> None:
        # single writer: encoding and the append-only rule (a turn is durable
        # as soon as it completes) live in one place
        with open(self.path, mode, encoding="utf-8") as tf:
            tf.write(text)

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
        self.sys_prompts = build_prompts(
            topic, self.proposer, self.critic, self.agent_eff)
        self.sessions = OpencodeSessions(cfg)
        self.turns: Turns = []
        self.started_at = 0.0  # set by start()
        self.transcript = Transcript()
        self.initial_msg = f"Discussion topic: {topic}\nStart the discussion now."

    def start(self) -> None:
        # [Model]/[Judge]/[Transport] lines + the transcript header
        self.started_at = time.time()
        print(f"\n  [Model] {self.proposer['name']} ({self.proposer['model']})  vs  "
              f"{self.critic['name']} ({self.critic['model']})")
        print(f"  [Judge] {self.judge['name']} ({self.judge['model']})")
        transports = sorted({transport_of(a)
                             for a in (self.proposer, self.critic, self.judge)})
        print(f"  [Transport] {', '.join(transports)}\n")
        header = (f"TOPIC: {self.topic}\nSTART: {time.ctime()}\n"
                  f"JUDGE: {self.judge['name']} ({self.judge['model']})\n{'='*60}\n\n")
        self.transcript.write(header, mode="w")

    def full_view(self, agent_key: str) -> list[dict[str, str]]:
        # stateless transport view: own turns = assistant, opponent = user
        msgs = [{"role": "user", "content": self.initial_msg}]
        for turn in self.turns:
            if turn.key == agent_key:
                msgs.append({"role": "assistant", "content": turn.text})
            else:
                msgs.append({"role": "user", "content": f"[{turn.name}]: {turn.text}"})
        return msgs

    def _session_title(self, role: str) -> str:
        return f"{SESSION_TITLE_PREFIX}{role} {self.transcript.ts}"

    def _stateful_session(self, sess_key: str,
                          cfg_key: str | None = None) -> SessionState:
        # one session per debate role; cfg_key differs only when the judge
        # falls back to a debater's agent config
        return self.sessions.ensure(sess_key, self._session_title(sess_key),
                                    cfg_key=cfg_key)

    def _stateful_send(self, st: SessionState, system: str, text: str) -> str:
        # every session send carries the same pair: the effective restricted
        # agent and the session's current turn cursor
        return oc_send(self.cfg["agents"][st["cfg_key"]], st["id"], system, text,
                       agent_name=self.agent_eff.get(st["cfg_key"]),
                       turn_index=st["sent"])

    def _request_turn(self, key: str, sys_prompt: str) -> str:
        a = self.cfg["agents"][key]
        if not is_opencode(a):
            return chat_completion(key, a, sys_prompt, self.full_view(key))
        # stateful: send only the turns this session hasn't seen yet
        st = self._stateful_session(key)
        parts = []
        if not st["started"]:
            parts.append(self.initial_msg)
        for turn in self.turns[st["sent"]:]:
            if turn.key != key:
                parts.append(f"[{turn.name}]: {turn.text}")
        return self._stateful_send(st, sys_prompt, "\n\n".join(parts))

    def take_turn(self, key: str, sys_prompt: str) -> Turn:
        # a debater turn: send, then record it and advance the session cursor
        # together - recording first means the cursor naturally counts the
        # agent's own turn, so the next send carries exactly the new reply
        a = self.cfg["agents"][key]
        text = self._request_turn(key, sys_prompt)
        turn = Turn(key, a["name"], a["model"], text)
        self.turns.append(turn)
        self.sessions.mark_seen(key, len(self.turns))
        return turn

    def judge_now(self) -> tuple[bool, str]:
        # stateless: full transcript each round. stateful: first call = full
        # transcript, later calls = only the new exchange
        if not is_opencode(self.judge):
            return check_consensus(self.cfg, self.topic, format_transcript(self.turns),
                                   self.judge_key)
        st = self._stateful_session(ROLE_JUDGE, self.judge_key)
        if st["started"]:
            text = "New exchange:\n\n" + format_transcript(self.turns[st["sent"]:])
        else:
            text = _judge_prompt(self.topic, format_transcript(self.turns))
        out = self._stateful_send(st, JUDGE_SYSTEM, text)
        self.sessions.mark_seen(ROLE_JUDGE, len(self.turns))
        return parse_verdict(out), out

    def run(self) -> None:
        # the whole run: round loop + exactly one summary/footer, whether it
        # ends normally or aborts mid-round (completed turns are already in the
        # transcript by then). Sessions are always closed on the way out.
        summary = RunSummary()
        try:
            for rnd in range(1, self.max_rounds + 1):
                self._start_round(rnd)

                print(f"  [Round {rnd}] {self.proposer['name']} is thinking...", flush=True)
                self._debate_turn(ROLE_PROPOSER)

                print(f"  [Round {rnd}] {self.critic['name']} is evaluating...", flush=True)
                self._debate_turn(ROLE_CRITIC)

                if summary.add_round(rnd, self._judge_round(rnd)):
                    break
        except BaseException as e:
            try:
                self.finish(False, summary.rounds_done, summary.judge_note,
                            aborted=f"{type(e).__name__}: {e}")
            except Exception:
                pass  # never mask the original failure
            raise
        else:
            self.finish(summary.consensus, summary.rounds_done, summary.judge_note,
                        no_verdict=summary.no_verdict)
        finally:
            self.sessions.close_all()

    def _debate_turn(self, key: str) -> None:
        # one debater turn: send, record, show; take_turn keeps the bookkeeping
        turn = self.take_turn(key, self.sys_prompts[key])
        self._append_turn(turn)
        _print_turn(turn)

    def _judge_round(self, rnd: int) -> JudgeRound:
        # round 1 is opening positions only: there is nothing to judge yet,
        # so even a 1-round run is never judged. A skipped/failed round has no
        # note, so the previous note stays in place.
        if rnd == 1:
            if self.max_rounds == 1:
                print("  [Judge] 1 round only, no judgement will be given\n")
            else:
                print("  [Judge] skipped in round 1\n")
            return JudgeRound()
        try:
            ok, note = self.judge_now()
        except Exception as e:
            print(f"  [Judge] failed to evaluate: {e}\n")
            return JudgeRound(failed=True)
        print(f"  [Judge: {self.judge['name']} "
              f"({self.judge['model']})] {note}\n")
        return JudgeRound(ok, note, attempted=True)

    def finish(self, consensus: bool, rounds_done: int, judge_note: str,
               aborted: str | None = None, no_verdict: bool = False) -> None:
        # final summary on screen + transcript footer
        elapsed = time.time() - self.started_at
        if aborted:
            status = "ABORTED"
        elif consensus:
            status = "CONSENSUS"
        elif no_verdict:
            status = "NO VERDICT (judge unavailable)"
        else:
            status = "NO CONSENSUS (hit max rounds)"
        plural = "" if rounds_done == 1 else "s"
        print("=" * 60)
        print(f"  Final status : {status}")
        print(f"  Duration     : {elapsed:.0f} sec, {rounds_done} round{plural}")
        print(f"  Transcript   : {self.transcript.path}")
        print("=" * 60)
        footer = (f"{'='*60}\nSTATUS: {status}\n"
                  f"JUDGE NOTE: {judge_note}\nDURATION: {elapsed:.0f}s\n")
        if aborted:
            footer += f"ABORTED: {aborted}\n"
        self.transcript.write(footer)

    def _start_round(self, rnd: int) -> None:
        self.transcript.write(f"--- Round {rnd} ---\n")

    def _append_turn(self, turn: Turn) -> None:
        self.transcript.write(f"[{turn.name}]\n{turn.text}\n\n")

def run_discussion(cfg: Config, topic: str | None = None) -> None:
    os.makedirs(ac_config.TRANSCRIPT_DIR, exist_ok=True)
    extra_roles = sorted(k for k in cfg["agents"] if k not in ROLES)
    if extra_roles:
        print(f"  [Warn] ignoring unknown agent role(s): {', '.join(extra_roles)}")

    available = check_servers(cfg)
    check_stateless_endpoints(cfg)

    prompted = topic is None
    topic = topic or input_required("Discussion topic: ")
    max_rounds = parse_max_rounds(cfg)
    print(f"\n  Topic: {topic}")
    print(f"  Max rounds: {max_rounds}\n")

    d = Discussion(cfg, topic, max_rounds, available)
    d.start()
    d.run()

    if prompted:
        try:
            input("\n  Press Enter to return to the menu...")
        except (EOFError, KeyboardInterrupt):
            # redirected stdin hit EOF or Ctrl+C at the last prompt: the run
            # already finished, so don't let it look aborted
            print()
