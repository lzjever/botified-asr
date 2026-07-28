---
name: botified-asr
description: Check a configured Botified ASR service's readiness, transcribe local audio, submit or query long transcription jobs, or delete a job when explicitly requested. Use when Codex needs to verify client configuration and authentication, check readiness, obtain a basic transcription, or work with a long transcription job.
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
active responses. Only when the user explicitly asks to cancel or delete a job,
run `scripts/botified-asr job-delete JOB_ID`. HTTP 202 reports the immediate
request without waiting; terminal HTTP 204 produces no output.

Return the helper's JSON unchanged, except that terminal `job-delete` HTTP 204
intentionally has no output. Treat a nonzero exit as a failed readiness check,
transcription, submission, query, wait, or deletion and report its stable error
code without exposing credentials, local paths, or raw configuration.

Read `references/api.md` only when the request or response contract is needed.
