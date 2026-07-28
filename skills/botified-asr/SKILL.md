---
name: botified-asr
description: Check a configured Botified ASR service's readiness or transcribe a local audio file with its basic synchronous transcription endpoint. Use when Codex needs to verify client configuration and authentication, check readiness, or obtain a basic transcription.
---

# Botified ASR Client

Resolve `scripts/botified-asr` relative to this `SKILL.md` (the skill root), then
first run `scripts/botified-asr health`. Only after it returns ready, run
`scripts/botified-asr transcribe AUDIO_FILE` for a basic transcription.

Return the helper's JSON unchanged. Treat a nonzero exit as a failed readiness
check or transcription and report its stable error code without exposing
credentials, local paths, or raw configuration.

Read `references/api.md` only when the request or response contract is needed.
