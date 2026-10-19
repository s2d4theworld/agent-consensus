# agent-consensus — interactive UI: prompts, setup wizard and menu status.
from ac_config import (
    AgentCfg,
    Config,
    DEFAULT_MAX_ROUNDS,
    INSTALL_HINT,
    MAX_ROUNDS_LIMIT,
    ROLES,
    ROLE_CRITIC,
    ROLE_JUDGE,
    ROLE_PROPOSER,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    TIMEOUT_MENU,
    TRANSPORT_OPENCODE,
    TRANSPORT_OPENAI,
    api_key_env,
    coerce_max_rounds,
    default_temperature,
    ensure_agent_installed,
    is_opencode,
    parse_temperature,
    role_agents,
    save_config,
)
from ac_transports import oc_agent_list_cached, oc_base_url

ROLE_DESCRIPTIONS: dict[str, str] = {
    ROLE_PROPOSER: "Proposer (put forward positions & arguments)",
    ROLE_CRITIC: "Critic (evaluate, find gaps, agree/disagree)",
    ROLE_JUDGE: "Judge (neutral — score consensus from round 2)",
}
DEFAULT_NAMES: dict[str, str] = {
    ROLE_PROPOSER: "Proposer",
    ROLE_CRITIC: "Critic",
    ROLE_JUDGE: "Judge",
}
DEFAULT_MODELS: dict[str, dict[str, str]] = {
    TRANSPORT_OPENCODE: {
        ROLE_PROPOSER: "opencode/longcat-2.5-preview-free",
        ROLE_CRITIC: "opencode/beam-free",
        ROLE_JUDGE: "opencode/deepseek-v4.1-flash-free",
    },
    TRANSPORT_OPENAI: {
        ROLE_PROPOSER: "meituan/longcat-2.5-preview:free",
        ROLE_CRITIC: "reflection/beam-1.0-501b-a23b:free",
        ROLE_JUDGE: "poolside/laguna-s-2.1:free",
    },
}

def banner() -> None:
    print("=" * 60)
    print("  AGENT CONSENSUS — Proposer vs Critic")
    print("=" * 60)

def input_required(prompt: str) -> str:
    while True:
        v = input(prompt).strip()
        if v:
            return v
        print("  (required)")

def input_optional(prompt: str, default: str = "") -> str:
    v = input(prompt).strip()
    return v if v else default

def _prompt_temperature(key: str) -> float:
    default = default_temperature(key)
    raw = input_optional(
        f"    Temperature ({TEMPERATURE_MIN}-{TEMPERATURE_MAX})", str(default))
    temp = parse_temperature(raw)
    if temp is None:
        print(f"    (invalid temperature, using {default})")
        return default
    return temp

def _prompt_opencode_agent(key: str) -> AgentCfg:
    name = input_optional("    Display name", DEFAULT_NAMES[key])
    url = input_optional("    Server URL", "http://127.0.0.1:4096")
    agent_name = input_optional(
        "    Restricted agent ('-' = none; 'debater' = websearch only)", "debater"
    )
    agent: AgentCfg = {
        "name": name,
        "api_url": url,
        "model": input_optional("    Model", DEFAULT_MODELS[TRANSPORT_OPENCODE][key]),
        "transport": TRANSPORT_OPENCODE,
    }
    if agent_name != "-":
        agent["agent"] = agent_name
    return agent

def _prompt_openai_agent(key: str) -> AgentCfg:
    name = input_optional("    Display name", DEFAULT_NAMES[key])
    url = input_optional(
        "    API URL (chat completions)",
        "http://127.0.0.1:8645/v1/chat/completions")
    raw = input_optional(
        f"    API key (Enter = 'proxy'; type '-' to use env {api_key_env(key)}): ",
        "proxy"
    )
    return {
        "name": name,
        "api_url": url,
        "model": input_optional("    Model", DEFAULT_MODELS[TRANSPORT_OPENAI][key]),
        "transport": TRANSPORT_OPENAI,
        "api_key": "" if raw == "-" else raw,
        "temperature": _prompt_temperature(key),
    }

def _prompt_agent(key: str, transport: str) -> AgentCfg:
    print(f"  --- Agent: {ROLE_DESCRIPTIONS[key]} ---")
    agent = (_prompt_opencode_agent(key) if transport == TRANSPORT_OPENCODE
             else _prompt_openai_agent(key))
    print()
    return agent

def setup_wizard() -> bool:
    banner()
    print("  Transports:")
    print("    opencode = local `opencode serve` sessions (stateful, no api_key)")
    print("    openai   = any OpenAI-compatible chat completions endpoint")
    print()
    transport = input_optional("  Transport (opencode/openai)", TRANSPORT_OPENCODE).lower()
    if transport not in (TRANSPORT_OPENCODE, TRANSPORT_OPENAI):
        transport = TRANSPORT_OPENCODE
    print()
    if transport == TRANSPORT_OPENCODE:
        print("  Default: local opencode server (no credentials needed).")
        print("  Run first: opencode serve --port 4096")
        print("  Model format: provider/model (e.g. opencode/beam-free)")
    else:
        print("  Default: local Hermes proxy (any api_key works).")
        print("  Run first: hermes proxy start --provider nous")
        print("  Empty api_key = env var CONSENSUS_API_KEY_<ROLE>")
    print("  Tip: pick a judge model DIFFERENT from both debaters for neutrality.")
    print()
    cfg: Config = {"agents": {key: _prompt_agent(key, transport) for key in ROLES}}
    cfg["max_rounds"] = coerce_max_rounds(
        input_optional(f"    Max discussion rounds (1-{MAX_ROUNDS_LIMIT})",
                       str(DEFAULT_MAX_ROUNDS)))
    if not save_config(cfg):
        return False
    if transport == TRANSPORT_OPENCODE:
        ensure_agent_installed()
    return True

def restricted_agents_by_base(cfg: Config) -> dict[str, set[str]]:
    # {base_url: {agent names}} of restricted (websearch) agents per server
    by_base: dict[str, set[str]] = {}
    for _, a in role_agents(cfg):
        if not is_opencode(a):
            continue
        name = a.get("agent")
        if name:
            by_base.setdefault(oc_base_url(a), set()).add(name)
    return by_base

def restricted_agent_lines(cfg: Config) -> list[str]:
    # menu status lines for the restricted (websearch) agents per server
    by_base = restricted_agents_by_base(cfg)
    if not by_base:
        return ["  Restricted agent: none configured (no websearch for debaters)"]
    lines = []
    for base, names in sorted(by_base.items()):
        avail = oc_agent_list_cached(base, timeout=TIMEOUT_MENU)
        for name in sorted(names):
            if avail is None:
                lines.append(f"  Restricted agent '{name}': {base} unreachable "
                             "(is `opencode serve` running?)")
            elif name in avail:
                lines.append(f"  Restricted agent '{name}': active on {base}")
            else:
                lines.append(f"  Restricted agent '{name}': missing on {base} "
                             f"({INSTALL_HINT})")
    return lines

def print_status(cfg: Config | None, err: str | None) -> None:
    if cfg and not err:
        p = cfg["agents"][ROLE_PROPOSER]
        c = cfg["agents"][ROLE_CRITIC]
        print(f"  Config found: {p['name']}[{p['model']}] vs {c['name']}[{c['model']}]")
        j = cfg["agents"].get(ROLE_JUDGE)
        if isinstance(j, dict):
            print(f"  Judge: {j.get('name', 'Judge')}[{j['model']}]")
        else:
            print("  Judge: none configured - proposer will judge")
        for line in restricted_agent_lines(cfg):
            print(line)
    elif cfg:
        print(f"  Config invalid: {err}")
    else:
        print("  No config yet — run setup first.")
