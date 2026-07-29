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

## Run from source

Python 3.11.13, `uv` 0.9.26, and `ffmpeg`/`ffprobe` are required.

```bash
uv sync --frozen
mkdir -p ~/.config/botified-asr
cat > ~/.config/botified-asr/config.yaml <<'YAML'
server:
  listen: "127.0.0.1:17770"
  public_base_url: "http://127.0.0.1:17770"
YAML
export BOTIFIED_ASR_API_KEY='replace-with-a-long-random-token'
uv run botified-asr
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

Add `Prefer: respond-async` to submit a persistent asynchronous job. For remote
access, keep the service or container host port bound to loopback and place an
authenticated TLS reverse proxy in front of it. Botified ASR does not configure
TLS or a firewall.

Generate the versioned offline API description with:

```bash
uv run scripts/generate-openapi openapi.json
```

The [Botified ASR Skill](skills/botified-asr/) provides the corresponding agent
client commands. The runtime does not serve an OpenAPI endpoint.

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
