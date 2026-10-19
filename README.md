# agent-consensus

CLI that makes two LLM agents (proposer vs critic) debate a topic until
they reach consensus, with a neutral judge scoring from round 2.

## Transports

- `opencode` (default) — uses `opencode serve` sessions. **Stateful**: the
  server keeps the conversation history, the script sends only the new
  message each turn. No credentials needed (zen free models).
- `openai` — any OpenAI-compatible `/v1/chat/completions` endpoint (e.g.
  a local proxy). Stateless: the script re-sends the full conversation on
  every call.

## Setup (opencode transport)

Start the server first:

    opencode serve --port 4096

Check it is up:

    curl http://127.0.0.1:4096/config

Then run `python agent_consensus.py setup` and accept the defaults:

- Proposer = `opencode/longcat-2.5-preview-free`
- Critic = `opencode/beam-free`
- Judge = `opencode/deepseek-v4.1-flash-free` (verified YES/NO verdicts)

These are the free models that existed when this repo was made. Free model
rosters change at any time and nothing here guarantees availability at any
point in the future: when a model starts failing (e.g. 404/503 from the
server), re-run `setup` and configure a model your server currently offers.

Model format is `provider/model` from the local server. Each role gets its
own session (`agent-consensus <role> <timestamp>`), deleted when the run
ends; orphans from crashed runs are swept at the next start. Round 1 is
never judged: opening positions can't be consensus yet, and the judge's
session starts with the full transcript at round 2. A `max_rounds` of 1
therefore ends unjudged with no consensus.

### Web search for debaters (optional `agent` field)

A role can run under a restricted serve-side agent via the optional
`"agent"` field in config.json. The defaults use `"debater"` — a
websearch-only agent defined in this repo at `.opencode/agents/debater.md`.
Install it into opencode's global agents dir (any machine):

    python agent_consensus.py install-agent

This copies the file to `~/.config/opencode/agents/debater.md` (the `setup`
wizard does it too). **Restart `opencode serve` to load it.** Until then
the script warns and falls back to the default agent with the no-tools
prompt, so nothing breaks.

The agent allows `websearch` and denies bash/edit/glob/grep/list/task/
webfetch/skill/question/lsp/external_directory (verified by probing the
serve API):

- `read` must stay allowed: any request using an agent with `read: deny`
  is rejected with the free-tier 403 ("can only be used from within
  OpenCode"). `external_directory: deny` still blocks reads outside the
  serve's directory.
- Free-tier 403 also triggers on sending a `tools` field, so tool access
  is controlled only via the agent definition (the script never sends
  `tools`).
- When a restricted agent is active, the role prompt allows the websearch
  tool for fact-checking; otherwise prompts say "reply with text only".

Notes from testing the serve API:

- Do NOT send a `tools` field: free-tier models reject it with a 403
  (the script never does; prompts also tell models to reply text-only).
- `system` per message works and carries the role prompts.
- Free models on provider `opencode` at the time of writing:
  longcat-2.5-preview-free, beam-free, deepseek-v4.1-flash-free,
  glm-5.3-flash-free, space-bunny-free, mimo-v2.6-flash-free
  (nemotron-3-ultra-free was 503 upstream). This roster is a snapshot from
  when the repo was made, not a promise of future availability.

## Usage

1. `python agent_consensus.py setup`
2. `python agent_consensus.py` — pick `[2] Start discussion` and type a
   topic; proposer and critic take turns debating; the judge scores from
   round 2 until consensus / max rounds (`max_rounds` in config.json).
3. Transcripts are saved to `transcripts/discussion_<timestamp>.txt`.

## openai transport (optional)

Pick `openai` in the wizard to use any chat-completions endpoint
(stateless). Example: a local proxy on `http://127.0.0.1:8645/v1`
(`hermes proxy start --provider nous`; any bearer token works — the proxy
attaches the real credentials). Empty api_key = env var
`CONSENSUS_API_KEY_<ROLE>` (`CONSENSUS_API_KEY_PROPOSER` / `_CRITIC` /
`_JUDGE`). Example free proxy models from when this repo was made
(availability is not guaranteed — re-pick if they disappear): proposer =
`meituan/longcat-2.5-preview:free`, critic =
`reflection/beam-1.0-501b-a23b:free`, judge = `poolside/laguna-s-2.1:free`.

Structure (Python 3.11+, stdlib only):

    agent_consensus.py       — CLI entry point (argparse, menu, setup dispatch)
    ac_errors.py             — shared exception types (ConsensusError and below)
    ac_config.py             — paths, constants, config file I/O, agent installer
    ac_transports.py         — HTTP helpers + both transports (opencode / openai)
    ac_engine.py             — debate engine (rounds, sessions, judge)
    ac_ui.py                 — prompts, setup wizard, menu status
    config.json              — agent configuration (created by setup)
    .opencode/agents/        — websearch agent template (install-agent copies it)
    transcripts/             — one transcript per session
    tests/                   — offline regression tests, one file per module
    tools/                   — end-to-end golden-output check (tools/golden/)

## Tests

    python -m unittest discover -s tests -v

Everything runs offline against a local fake `opencode serve`: engine flows
(consensus, no-consensus, missing-agent fallback, judge isolation, orphan
sweep, fail-fast on dead servers), transport details (deterministic
messageIDs, agent-list cache, HTTP retry policy, error envelopes), config
validation, the setup wizard and the menu. The suite is split per module
(`test_config` / `test_transports` / `test_engine` / `test_ui` / `test_entry`)
with shared fakes in `tests/helpers.py`.

For a byte-level check of complete runs (stdout, transcript, recorded server
requests), replay the golden scenarios:

    python tools/golden_check.py            # verify
    python tools/golden_check.py --update   # rewrite after intentional changes

Goldens live in `tools/golden/`; temp paths, timestamps, durations and
ephemeral ports are normalized before comparison.

CI (`.github/workflows/`) runs the test suite, the golden check, pylint
(`.pylintrc`) and mypy (`mypy.ini`) on every push.
