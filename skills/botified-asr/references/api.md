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

# Get a transcription job

`scripts/botified-asr job-get JOB_ID` sends an authenticated
GET `/v1/audio/transcriptions/{job_id}` request. `JOB_ID` must be exactly eight
uppercase Crockford Base32 characters.

The service response is returned unchanged. A job may be `queued` or `running`
with progress information, `succeeded` with its result, or `failed` or
`cancelled` with the corresponding service details.

HTTP error bodies are also returned unchanged. Transport and helper-local
failures return JSON with a stable `error.code`.
