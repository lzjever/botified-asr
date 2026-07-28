---
name: botified-asr
description: Check a configured Botified ASR service's readiness, transcribe local audio, submit long transcription jobs, or query submitted jobs. Use when Codex needs to verify client configuration and authentication, check readiness, obtain a basic transcription, or work with a long transcription job.
---

# Botified ASR Client

Resolve `scripts/botified-asr` relative to this `SKILL.md` (the skill root), then
first run `scripts/botified-asr health`. Only after it returns ready, run
`scripts/botified-asr transcribe AUDIO_FILE` for a basic transcription, or
`scripts/botified-asr transcribe-long AUDIO_FILE` to submit a long transcription
job. Use the returned job ID with `scripts/botified-asr job-get JOB_ID` to query
the job's current state or result, or with
`scripts/botified-asr job-wait JOB_ID TIMEOUT_SECONDS` to wait for its terminal
response. The wait command outputs only that final response, never intermediate
active responses.

Return the helper's JSON unchanged. Treat a nonzero exit as a failed readiness
check, transcription, submission, query, or wait and report its stable error
code without exposing credentials, local paths, or raw configuration.

Read `references/api.md` only when the request or response contract is needed.
