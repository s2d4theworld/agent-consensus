# agent-consensus — interactive UI: prompts, setup wizard and menu status.
from ac_config import (
    AgentCfg,
    Config,
    DEFAULT_MAX_ROUNDS,
    INSTALL_HINT,
    ROLES,
    ROLE_CRITIC,
    ROLE_JUDGE,
    ROLE_PROPOSER,
    TIMEOUT_MENU,
    TRANSPORT_OPENCODE,
    TRANSPORT_OPENAI,
    coerce_max_rounds,
    ensure_agent_installed,
    is_opencode,
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

def _prompt_agent(key: str, transport: str) -> AgentCfg:
    print(f"  --- Agent: {ROLE_DESCRIPTIONS[key]} ---")
    # required keys are placeholder-filled here and set by the prompts below
    a: AgentCfg = {"name": "", "api_url": "", "model": "", "transport": transport}
    a["name"] = input_optional("    Display name", DEFAULT_NAMES[key])
    if transport == TRANSPORT_OPENCODE:
        a["api_url"] = input_optional("    Server URL", "http://127.0.0.1:4096")
        agent_name = input_optional(
            "    Restricted agent ('-' = none; 'debater' = websearch only)", "debater"
        )
        if agent_name != "-":
            a["agent"] = agent_name
    else:
        a["api_url"] = input_optional(
            "    API URL (chat completions)",
            "http://127.0.0.1:8645/v1/chat/completions")
        key_env = f"CONSENSUS_API_KEY_{key.upper()}"
        raw = input_optional(
            f"    API key (Enter = 'proxy'; type '-' to use env {key_env}): ", "proxy"
        )
        a["api_key"] = "" if raw == "-" else raw
    a["model"] = input_optional("    Model", DEFAULT_MODELS[transport][key])
    if transport == TRANSPORT_OPENAI:
        default_temp = "0.0" if key == ROLE_JUDGE else "0.7"
        t = input_optional("    Temperature (0.0-2.0)", default_temp)
        try:
            a["temperature"] = float(t)
        except ValueError:
            a["temperature"] = float(default_temp)
    print()
    return a

def setup_wizard() -> None:
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
        input_optional("    Max discussion rounds", str(DEFAULT_MAX_ROUNDS)))
    save_config(cfg)
    if transport == TRANSPORT_OPENCODE:
        ensure_agent_installed()

def restricted_agents_by_base(cfg: Config) -> dict[str, set[str]]:
    # {base_url: {agent names}} of restricted (websearch) agents per server
    by_base: dict[str, set[str]] = {}
    for a in cfg["agents"].values():
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
