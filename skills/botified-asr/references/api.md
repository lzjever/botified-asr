# Health

`scripts/botified-asr health` sends an authenticated `GET` to
`/health/ready`.

A ready service returns HTTP 200:

```json
{"status":"ready"}
```

Service errors are returned unchanged. Helper-local failures return JSON with a
stable `error.code`.

# Basic synchronous transcription

`scripts/botified-asr transcribe AUDIO_FILE` sends an authenticated
POST `/v1/audio/transcriptions` request with a readable local file,
`model=sensevoice`, and `response_format=json`.

A successful request returns the service JSON unchanged:

```json
{"text":"transcribed text"}
```

HTTP error bodies are also returned unchanged. Transport and helper-local
failures return JSON with a stable `error.code`.

# Long transcription submission

`scripts/botified-asr transcribe-long AUDIO_FILE` sends an authenticated
POST `/v1/audio/transcriptions` request with `Prefer: respond-async`, a readable
local file, `model=sensevoice`, `response_format=json`, and
`chunking_strategy=auto`.

An accepted request returns the service JSON unchanged:

```json
{"id":"7K3M9Q2W","status":"queued"}
```

HTTP error bodies are also returned unchanged. Transport and helper-local
failures return JSON with a stable `error.code`.

# Meeting transcription submission

`scripts/botified-asr transcribe-meeting AUDIO_FILE [SPEAKER_ID ...]` sends an
authenticated POST `/v1/audio/transcriptions` request with `Prefer:
respond-async`, a readable local file, `model=sensevoice-diarize`,
`response_format=diarized_json`, and `chunking_strategy=auto`. It accepts zero
through 32 unique uppercase Crockford Base32 speaker IDs and sends each provided
ID as a repeated `known_speaker_ids[]` field. The helper does not list or select
speakers.

Only HTTP 202 is accepted. The service JSON is returned unchanged; use its job
ID with `job-wait`. After a successful terminal response, the Agent can project
`result.segments` into a speaker/timestamp meeting transcript while preserving
`Unknown` speaker labels. The helper does not summarize.

# Get a transcription job

`scripts/botified-asr job-get JOB_ID` sends an authenticated
GET `/v1/audio/transcriptions/{job_id}` request. `JOB_ID` must be exactly eight
uppercase Crockford Base32 characters.

The service response is returned unchanged. A job may be `queued` or `running`
with progress information, `succeeded` with its result, or `failed` or
`cancelled` with the corresponding service details.

HTTP error bodies are also returned unchanged. Transport and helper-local
failures return JSON with a stable `error.code`.

# Wait for a transcription job

`scripts/botified-asr job-wait JOB_ID TIMEOUT_SECONDS` immediately queries the
same job endpoint, then waits while the service returns HTTP 202. Timeout
seconds must be a decimal integer from 1 through 999999999.

The command emits no intermediate active responses. It returns the terminal
HTTP 200 JSON unchanged, or one final service or helper error JSON. A local
deadline expiry returns `job_wait_timeout`.

# Delete or cancel a transcription job

`scripts/botified-asr job-delete JOB_ID` sends an authenticated
DELETE `/v1/audio/transcriptions/{job_id}` request with the same strict job ID
validation as the query commands.

HTTP 202 returns the service JSON unchanged as the immediate cancellation
request and does not wait. Terminal HTTP 204 succeeds with no output. HTTP error
bodies are returned unchanged; transport and helper-local failures return JSON
with a stable `error.code`.

# Existing speaker profiles

`scripts/botified-asr speaker-list` sends an authenticated
GET `/v1/speakers`. `scripts/botified-asr speaker-get SPEAKER_ID` sends an
authenticated GET `/v1/speakers/{speaker_id}`. `SPEAKER_ID` must be exactly
eight uppercase Crockford Base32 characters.

List and get return HTTP 200 service JSON unchanged.
Only when the user explicitly requests registration,
`scripts/botified-asr speaker-add NAME SAMPLE_FILE_1 SAMPLE_FILE_2 [SAMPLE_FILE_3 ... SAMPLE_FILE_5]`
sends an authenticated POST `/v1/speakers`. It submits `name` literally and
two to five readable, non-empty regular files as repeated `samples[]` parts.
It does not submit a description; use `speaker-put` afterward when one is
requested. HTTP 201 service JSON is returned unchanged. Local sample paths,
sample contents, and credentials are not emitted.

`scripts/botified-asr speaker-put SPEAKER_ID NAME [DESCRIPTION]` sends an
authenticated PUT `/v1/speakers/{speaker_id}`. It passes provided metadata with
curl `--form-string` and submits no `samples[]` or `file` field. HTTP 200
service JSON is returned unchanged.

`scripts/botified-asr speaker-delete SPEAKER_ID` sends authenticated
DELETE `/v1/speakers/{speaker_id}`, and HTTP 204 succeeds with no output. HTTP
error bodies are returned unchanged.
