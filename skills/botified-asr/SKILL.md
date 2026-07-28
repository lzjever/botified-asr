---
name: botified-asr
description: Check whether a configured Botified ASR service is reachable and ready. Use when Codex needs to verify Botified ASR client configuration, authentication, or service readiness.
---

# Botified ASR Readiness

Resolve `scripts/botified-asr` relative to this `SKILL.md` (the skill root), then
run `scripts/botified-asr health` before relying on the service.

Return the helper's JSON unchanged. Treat a nonzero exit as a failed readiness
check and report its stable error code without exposing credentials or raw
configuration.

Read `references/api.md` only when the health request or response contract is
needed.
