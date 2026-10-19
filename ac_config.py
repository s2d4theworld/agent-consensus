# agent-consensus — paths, constants, config file I/O and the
# restricted-agent installer. Python stdlib only.
import json
import os
from math import isfinite
from typing import Any, NotRequired, TypedDict

class AgentCfg(TypedDict):
    # one config.json "agents" value; everything but name/api_url/model is
    # optional (defaults are filled in by the wizard or at call time)
    name: str
    api_url: str
    model: str
    transport: NotRequired[str]
    api_key: NotRequired[str]
    agent: NotRequired[str | None]
    temperature: NotRequired[float]

class Config(TypedDict):
    # the whole config.json
    agents: dict[str, AgentCfg]
    max_rounds: NotRequired[int]

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
TRANSCRIPT_DIR = os.path.join(HERE, "transcripts")
AGENT_SRC = os.path.join(HERE, ".opencode", "agents", "debater.md")
OPENCODE_AGENTS_DIR = os.path.expanduser(
    os.path.join("~", ".config", "opencode", "agents")
)

TRANSPORT_OPENCODE = "opencode"
TRANSPORT_OPENAI = "openai"
ROLES = ("proposer", "critic", "judge")
ROLE_PROPOSER, ROLE_CRITIC, ROLE_JUDGE = ROLES
DEFAULT_MAX_ROUNDS = 10
# debaters get some variety, the judge stays deterministic; one source for
# the wizard suggestions and the transport fallbacks
TEMPERATURE_MIN, TEMPERATURE_MAX = 0.0, 2.0
DEFAULT_TEMPERATURE_JUDGE = 0.0
DEFAULT_TEMPERATURE_DEBATER = 0.7

# timeouts (seconds)
TIMEOUT_DEFAULT = 180    # generic HTTP requests
TIMEOUT_SESSION = 60     # opencode session creation
TIMEOUT_MESSAGE = 300    # opencode message turns (models can think for long)
TIMEOUT_LIST = 15        # session list / delete / sweep
TIMEOUT_MENU = 4         # agent list from the interactive menu (stay snappy)
TIMEOUT_PREFLIGHT = 3    # TCP preflight per stateless endpoint
AGENT_LIST_TTL = 5       # menu may reuse a recent agent-list lookup (seconds)
# orphan sweep: skip sessions with activity newer than this (seconds); a
# crashed run goes quiet, a parallel run keeps refreshing time.updated
SWEEP_MIN_AGE = 300
SESSION_TITLE_PREFIX = "agent-consensus "  # the sweep recognizes titles by it
TS_FORMAT = "%Y%m%d_%H%M%S"  # run-start stamp appended to session titles
# deterministic message id: {sid} = session id without the "ses_" prefix, cut
# to 12 chars; {turn} = index of the first new turn; {digest} = sha256 of the
# prompt text, first 8 hex chars. An identical retry after a lost response
# reuses the id and the server dedupes it; different content always gets a
# fresh id, so id reuse can never drop a turn
MESSAGE_ID_TPL = "msg_{sid}_t{turn}_{digest}"
# advice shown when a configured restricted (websearch) agent is missing
INSTALL_HINT = "run: python agent_consensus.py install-agent, then restart opencode serve"

# post_json retry policy
RETRY_HTTP_CODES = (429, 502, 503)   # transient upstream errors
RETRY_HTTP_WAITS = (45, 90, 135)     # waits before retry 1/2/3 (seconds)
RETRY_500_WAITS = (15,)              # a 500 is often deterministic: retry once, briefly
RETRY_CONN_WAITS = (20, 40, 60)      # waits after connection failures
RETRY_AFTER_MAX = 300                # cap for a server-provided Retry-After

def load_config() -> Config | None:
    if not os.path.exists(CONFIG_PATH):
        return None
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:  # ValueError: JSONDecodeError, bad encoding
        print(f"  [config] cannot read {CONFIG_PATH}: {e}")
        return None

def validate_config(cfg: object) -> str | None:
    agents = cfg.get("agents") if isinstance(cfg, dict) else None
    if not isinstance(agents, dict):
        return "missing 'agents' object"
    for key in (ROLE_PROPOSER, ROLE_CRITIC):
        if not isinstance(agents.get(key), dict):
            return f"missing agent '{key}' (outdated config? re-run setup)"
    for key, a in agents.items():
        err = _agent_error(key, a)
        if err:
            return err
    return None

def _agent_error(key: str, agent: Any) -> str | None:
    if not isinstance(agent, dict):
        return f"agent '{key}' is not an object"
    for field in ("name", "api_url", "model"):
        if not agent.get(field):
            return f"agent '{key}' is missing '{field}'"
    if "://" not in agent["api_url"]:
        return f"agent '{key}' field 'api_url' must be a full URL (http://...)"
    if agent.get("transport", TRANSPORT_OPENAI) not in (TRANSPORT_OPENCODE, TRANSPORT_OPENAI):
        return (f"agent '{key}' field 'transport' must be "
                f"'{TRANSPORT_OPENCODE}' or '{TRANSPORT_OPENAI}'")
    if agent.get("agent") is not None and not isinstance(agent.get("agent"), str):
        return f"agent '{key}' field 'agent' must be a string"
    if "temperature" in agent and not valid_temperature(agent["temperature"]):
        return (f"agent '{key}' field 'temperature' must be a number "
                f"between {TEMPERATURE_MIN} and {TEMPERATURE_MAX}")
    return None

def valid_temperature(value: object) -> bool:
    # bool is an int in Python but never a sensible temperature
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return isfinite(value) and TEMPERATURE_MIN <= value <= TEMPERATURE_MAX

def save_config(cfg: Config) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"  Config saved: {CONFIG_PATH}")

def coerce_max_rounds(value: Any) -> int:
    # one rule for config files and wizard input: malformed values fall back
    # to the default, a valid value is never below 1 round
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return DEFAULT_MAX_ROUNDS

def transport_of(agent_cfg: AgentCfg) -> str:
    return agent_cfg.get("transport", TRANSPORT_OPENAI)

def api_key_env(agent_key: str) -> str:
    # the wizard prints this name and the transport reads it: keep one source
    return f"CONSENSUS_API_KEY_{agent_key.upper()}"

def is_opencode(agent_cfg: AgentCfg) -> bool:
    return transport_of(agent_cfg) == TRANSPORT_OPENCODE

def ensure_agent_installed() -> bool:
    # copy the repo's restricted websearch agent into opencode's global
    # agents dir (~/.config/opencode/agents) so any machine can use it
    if not os.path.exists(AGENT_SRC):
        print(f"  Agent template missing: {AGENT_SRC}")
        return False
    try:
        with open(AGENT_SRC, "r", encoding="utf-8") as f:
            content = f.read()
        os.makedirs(OPENCODE_AGENTS_DIR, exist_ok=True)
        dest = os.path.join(OPENCODE_AGENTS_DIR, "debater.md")
        if os.path.exists(dest):
            with open(dest, "r", encoding="utf-8") as f:
                if f.read() == content:
                    print(f"  Websearch agent already installed: {dest}")
                    return True
        with open(dest, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Websearch agent installed: {dest}")
        print("  (restart opencode serve to load it)")
        return True
    except OSError as e:
        print(f"  Could not install websearch agent: {e}")
        return False
