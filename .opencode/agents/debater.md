---
description: "Restricted agent for debate bots: web search only; no shell, file writes, or subagents"
mode: subagent
hidden: true
# opencode applies this budget per message send, restarting the count every
# turn; proposer and critic never share it. 100 is "no practical limit" for
# one turn (the transport's 300s timeout is the hard stop). If it is ever
# reached mid-reply, opencode forces a text-only summary turn.
steps: 100
permission:
  bash: deny
  edit: deny
  glob: deny
  grep: deny
  list: deny
  task: deny
  webfetch: deny
  websearch: allow
  skill: deny
  question: deny
  lsp: deny
  external_directory: deny
---
You are a participant in a structured debate. Reply with text only. You may use the websearch tool to verify facts when needed, but do not use any other tools.
