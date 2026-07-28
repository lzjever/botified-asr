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
