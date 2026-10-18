---
description: "Restricted agent for debate bots: web search only; no shell, file writes, or subagents"
mode: subagent
hidden: true
steps: 2
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
