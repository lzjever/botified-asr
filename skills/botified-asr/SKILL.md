---
name: botified-asr
description: Check a configured Botified ASR service's readiness, transcribe local audio, submit or query long transcription jobs, register speaker profiles from explicitly provided samples, list or query profiles, or update metadata or delete a job or profile when explicitly requested. Use when Codex needs to verify client configuration and authentication, obtain a transcription, work with a long transcription job, or explicitly register or manage a speaker profile.
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

Use `scripts/botified-asr speaker-list` to list existing speaker profiles and
`scripts/botified-asr speaker-get SPEAKER_ID` to query one. Only when the user
explicitly asks to register a speaker and provides two to five local sample
files, run
`scripts/botified-asr speaker-add NAME SAMPLE_FILE_1 SAMPLE_FILE_2 [SAMPLE_FILE_3 ... SAMPLE_FILE_5]`.
Do not upload those voice samples for any other purpose or expose their paths
or contents. Set an optional description afterward with `speaker-put`.
Only when the user
explicitly asks to update an existing profile's metadata, run
`scripts/botified-asr speaker-put SPEAKER_ID NAME [DESCRIPTION]`. `NAME` is
required. Omit `DESCRIPTION` to preserve it, pass an empty value to clear it,
or pass a nonempty value to replace it. Only when the user explicitly asks to
delete a speaker profile, run
`scripts/botified-asr speaker-delete SPEAKER_ID`.

Return the helper's JSON unchanged. Successful `job-delete` and
`speaker-delete` HTTP 204 responses intentionally have no output. Treat a
nonzero exit as a failed readiness check, transcription, submission, query,
registration, metadata update, wait, or deletion and report its stable error
code without exposing credentials, local paths, voice samples, or raw
configuration.

Read `references/api.md` only when the request or response contract is needed.
