# Botified ASR

Botified ASR is a single-instance, multi-user speech recognition service for
Botified agents. It accepts concurrent HTTP requests, bounds persisted work,
limits inference concurrency, supports cancellation, persists results, and
restarts unfinished jobs from the beginning after a service restart.

The current CPU runtime supports Linux x86_64 and aarch64. It provides:

- synchronous transcription and persistent asynchronous jobs through the same
  HTTP resource;
- SenseVoice transcription with optional emotion and audio-event annotations;
- offline diarization for audio up to 30 minutes;
- best-effort anonymous speaker separation and matching against selected,
  enrolled speakers.

Speaker diarization and identity matching are probabilistic. An unknown or
incorrect speaker label is possible and must not be treated as authentication
or authorization.

## Run the CPU container

The supported production artifact is the fixed-version, multi-architecture CPU
image. It supports Linux x86_64 and aarch64; no mutable `latest` tag is
published.

Create the client connection file with a random API key:

```bash
client_dir="${XDG_CONFIG_HOME:-$HOME/.config}/botified-asr"
client_env="$client_dir/client.env"
umask 077
mkdir -p "$client_dir"
chmod 700 "$client_dir"
api_key="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
cat > "$client_env" <<EOF
BOTIFIED_ASR_BASE_URL=http://127.0.0.1:17770
BOTIFIED_ASR_API_KEY=$api_key
EOF
chmod 600 "$client_env"
```

The service and the Botified ASR Skill use this same mode `0600` file. Start
the current fixed version:

```bash
docker run --detach \
  --name botified-asr \
  --restart on-failure:3 \
  --env-file "${XDG_CONFIG_HOME:-$HOME/.config}/botified-asr/client.env" \
  --publish 127.0.0.1:17770:17770 \
  --mount type=volume,src=botified-asr-data,dst=/data \
  ghcr.io/lzjever/botified-asr:v0.0.0
```

Docker creates the named volume if needed. `/data/state` holds the database,
jobs, results, and speaker data; its sibling `/data/models` holds the model
cache. The first startup downloads pinned model revisions and can take several
minutes.

Inspect the logs, then make an authenticated ready request:

```bash
docker logs botified-asr

client_env="${XDG_CONFIG_HOME:-$HOME/.config}/botified-asr/client.env"
base_url="$(sed -n 's/^BOTIFIED_ASR_BASE_URL=//p' "$client_env")"
api_key="$(sed -n 's/^BOTIFIED_ASR_API_KEY=//p' "$client_env")"
curl --fail-with-body \
  --header "Authorization: Bearer $api_key" \
  "$base_url/health/ready"
```

### Advanced container configuration

The image already contains its complete default configuration. To change
inference lanes or capacity, start with
[config/container.yaml](config/container.yaml) from the exact Git tag matching
the image version. Copy and edit the whole file, keeping `/data/state` and
`/data/models` as non-overlapping siblings, then run:

```bash
docker run --detach \
  --name botified-asr \
  --restart on-failure:3 \
  --env-file "${XDG_CONFIG_HOME:-$HOME/.config}/botified-asr/client.env" \
  --publish 127.0.0.1:17770:17770 \
  --mount type=volume,src=botified-asr-data,dst=/data \
  --mount type=bind,src=/absolute/path/config.yaml,dst=/etc/botified-asr/custom.yaml,readonly \
  ghcr.io/lzjever/botified-asr:v0.0.0 \
  --config /etc/botified-asr/custom.yaml
```

Botified ASR loads one complete YAML; it does not merge fragments or offer
per-field environment aliases.

## Develop and build from source

Python 3.11.13, `uv` 0.9.26, and `ffmpeg`/`ffprobe` are required.

```bash
uv sync --frozen
mkdir -p ~/.config/botified-asr
cat > ~/.config/botified-asr/config.yaml <<'YAML'
server:
  listen: "127.0.0.1:17770"
YAML
export BOTIFIED_ASR_API_KEY='replace-with-a-long-random-token'
uv run botified-asr --config ~/.config/botified-asr/config.yaml
```

The example above listens on `127.0.0.1:17770`. In another shell:

```bash
export BOTIFIED_ASR_API_KEY='replace-with-a-long-random-token'
curl --fail-with-body \
  --header "Authorization: Bearer $BOTIFIED_ASR_API_KEY" \
  --form model=sensevoice \
  --form file=@audio.flac \
  http://127.0.0.1:17770/v1/audio/transcriptions
```

Add `Prefer: respond-async` to submit a persistent asynchronous job.

Only basic synchronous transcription is an OpenAI Audio API compatible subset.
Asynchronous jobs, speaker profiles, diarization, and `include[]` are Botified
extensions. Generate the exact wire description on demand from the checkout
matching the service artifact:

```bash
uv run scripts/generate-openapi openapi.json
```

Power users can build a non-official image from the current checkout and package
the Skill source:

```bash
docker build \
  --build-arg BOTIFIED_ASR_VERSION=0.0.0 \
  --tag botified-asr:local \
  .
scripts/build-skill-tarball /tmp/botified-asr-skill.tar.gz
```

The local image supports the same complete read-only config mount and
`--config` override shown above.

The [Botified ASR Skill](skills/botified-asr/) provides the corresponding agent
client commands. For an official image, use the Skill from its matching exact
Git tag; for a custom image, use the checkout that built it. Explicitly place
or point `skills/botified-asr` at one runtime location: Codex
`~/.codex/skills/botified-asr`, OpenClaw
`~/.agents/skills/botified-asr`, or Botified
`~/.local/share/botified/skills/botified-asr`. The project does not discover or
copy the Skill automatically. After configuring `client.env`, from that Skill
root first run `scripts/botified-asr health`.

The runtime does not serve an OpenAPI endpoint.

## Models

Model weights are not included in the source package or CPU image. On first
startup the service downloads these exact revisions from Hugging Face into the
configured persistent model cache, so initial startup requires network access
unless the cache is already populated.

| Model | Source and revision | License |
|---|---|---|
| `FunAudioLLM/SenseVoiceSmall` | [Hugging Face snapshot `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7`](https://huggingface.co/FunAudioLLM/SenseVoiceSmall/tree/3847d57b6bdf2dd8875cb1508d2af43d80a16bf7) | [FunASR Model Open Source License Agreement 1.1](https://github.com/modelscope/FunASR/blob/8a34247dc5ff71bea61b37e57f941680b456753f/MODEL_LICENSE) |
| `funasr/fsmn-vad` | [Hugging Face snapshot `df20e6b30c653645fa4ff125cacfcabd1020a669`](https://huggingface.co/funasr/fsmn-vad/tree/df20e6b30c653645fa4ff125cacfcabd1020a669) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| `funasr/campplus` | [Hugging Face snapshot `e4b6ede7ce16997aff4ae69fbca1f0175e2afede`](https://huggingface.co/funasr/campplus/tree/e4b6ede7ce16997aff4ae69fbca1f0175e2afede) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |

SenseVoiceSmall uses a custom model agreement, not the MIT license covering this
repository. This project does not represent that agreement as a grant for
commercial use; read and follow its terms before using the model.

## License

Botified ASR source code is available under the [MIT License](LICENSE).
Third-party software and model notices are recorded in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
