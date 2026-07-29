# Botified ASR

Botified ASR is a single-instance, multi-user speech recognition service for
Botified agents. It accepts concurrent HTTP requests, bounds persisted work,
limits inference concurrency, supports cancellation, persists results, and
restarts unfinished jobs from the beginning after a service restart.

The official v0.1.1 CPU image supports Linux x86_64 (`linux/amd64`) only. The
source dependency lock retains Linux aarch64 compatibility for custom Docker
builds, but v0.1.1 does not include or promise support for an official ARM64
artifact. An official ARM64 image will be considered only when explicitly
needed and after validation on a native Linux aarch64 host.

The service provides:

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

The fixed `v0.1.1` CPU image targets Linux x86_64 (`linux/amd64`); no mutable
`latest` tag is published.

Create a private Docker environment file with a random API key:

```bash
umask 077
export BOTIFIED_ASR_API_KEY="$(
  python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
)"
printf 'BOTIFIED_ASR_API_KEY=%s\n' "$BOTIFIED_ASR_API_KEY" \
  > ./botified-asr.env
chmod 600 ./botified-asr.env
```

This file configures only the service container. Run the fixed `v0.1.1` release:

```bash
docker run --detach \
  --name botified-asr \
  --restart on-failure:3 \
  --env-file ./botified-asr.env \
  --publish 127.0.0.1:17770:17770 \
  --mount type=volume,src=botified-asr-data,dst=/data \
  ghcr.io/lzjever/botified-asr:v0.1.1
```

Docker creates the named volume if needed. `/data/state` holds the database,
jobs, results, and speaker data; its sibling `/data/models` holds the model
cache. The first startup downloads pinned model revisions and can take several
minutes.

Inspect the logs, then make an authenticated ready request:

```bash
docker logs botified-asr

curl --fail-with-body \
  --header "Authorization: Bearer $BOTIFIED_ASR_API_KEY" \
  http://127.0.0.1:17770/health/ready
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
  --env-file ./botified-asr.env \
  --publish 127.0.0.1:17770:17770 \
  --mount type=volume,src=botified-asr-data,dst=/data \
  --mount type=bind,src=/absolute/path/config.yaml,dst=/etc/botified-asr/custom.yaml,readonly \
  ghcr.io/lzjever/botified-asr:v0.1.1 \
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
  --build-arg BOTIFIED_ASR_VERSION=0.1.1 \
  --tag botified-asr:local \
  .
scripts/build-skill-tarball /tmp/asr-skill.tar.gz
```

The local image supports the same complete read-only config mount and
`--config` override shown above.

The [ASR Skill](skills/asr/) provides the corresponding agent client commands.
For an official image, use the Skill from its matching exact Git tag; for a
custom image, use the checkout that built it.

Install it into Botified's resolved Agent root:

```bash
AGENTS_DIR=/absolute/path/to/resolved-agents-dir
: "${AGENTS_DIR:?set AGENTS_DIR to the resolved Agent root}"

install -d -m 0700 \
  "${AGENTS_DIR}/skills/asr/agents" \
  "${AGENTS_DIR}/skills/asr/references" \
  "${AGENTS_DIR}/skills/asr/scripts" \
  "${AGENTS_DIR}/env.d"
install -m 0644 \
  skills/asr/SKILL.md \
  "${AGENTS_DIR}/skills/asr/SKILL.md"
install -m 0644 \
  skills/asr/agents/openai.yaml \
  "${AGENTS_DIR}/skills/asr/agents/openai.yaml"
install -m 0644 \
  skills/asr/references/api.md \
  "${AGENTS_DIR}/skills/asr/references/api.md"
install -m 0755 \
  skills/asr/scripts/botified-asr \
  "${AGENTS_DIR}/skills/asr/scripts/botified-asr"
```

Create the Skill client's private configuration as an atomic update:

```bash
install -m 0600 /dev/null \
  "${AGENTS_DIR}/env.d/botified-asr.env.tmp"
# Write exactly these two literal, unquoted NAME=VALUE entries to the .tmp file:
# BOTIFIED_ASR_BASE_URL=http://asr-host:17770
# BOTIFIED_ASR_API_KEY=replace_with_actual_key
mv \
  "${AGENTS_DIR}/env.d/botified-asr.env.tmp" \
  "${AGENTS_DIR}/env.d/botified-asr.env"
```

Define each name only once across `env.d/*.env`. Botified injects them into new
helper processes; the helper does not locate or parse configuration files.
From the Skill root, first run `scripts/botified-asr health`.

For another Agent runtime, explicitly place or point `skills/asr` at its Skill
root and inject the same complete environment pair into the helper process.

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
