# Health

`scripts/botified-asr health` sends an authenticated `GET` to
`/health/ready`.

A ready service returns HTTP 200:

```json
{"status":"ready"}
```

Service errors are returned unchanged. Helper-local failures return JSON with a
stable `error.code`.
