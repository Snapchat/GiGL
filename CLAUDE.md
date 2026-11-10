# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coordination With AGENTS.md and CONTEXT.md

- CONTEXT.md contains the canonical shared project context, architecture intent, and workflow conventions; read it for
  repository work.
- AGENTS.md should direct agents to read CLAUDE.md and discover/use skills under `.claude/skills`.

## Custom Agents

- At the start of each session, discover repository-local custom agents in `.claude/agents/` and read the agent(s)
  relevant to the task before using them. Coding agents must follow these as repository guidance.
- Before opening a PR or requesting code review, invoke the `setup-pr` agent in `.claude/agents/setup-pr.md`.
