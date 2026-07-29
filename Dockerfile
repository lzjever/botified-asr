# syntax=docker/dockerfile:1

ARG BOTIFIED_ASR_VERSION

FROM python:3.11.13-slim-bookworm@sha256:86adf8dbadc3d6e82ee5dd2c74bec2e1c2467cdad47886280501df722372d2e1 AS python-base

FROM python-base AS builder

ENV UV_PROJECT_ENVIRONMENT=/opt/botified-asr
WORKDIR /build

RUN python -m pip install --no-cache-dir uv==0.9.26

COPY pyproject.toml uv.lock README.md LICENSE THIRD_PARTY_NOTICES ./
COPY src/ src/

RUN uv sync --frozen --no-dev --no-editable

ARG BOTIFIED_ASR_VERSION

RUN actual_version=$(/opt/botified-asr/bin/botified-asr --version) \
    && expected_version="botified-asr ${BOTIFIED_ASR_VERSION:-}" \
    && if [ -z "${BOTIFIED_ASR_VERSION:-}" ] \
       || [ "$actual_version" != "$expected_version" ]; then \
           printf 'botified-asr version mismatch: expected "%s", got "%s"\n' \
               "$expected_version" "$actual_version" >&2; \
           exit 1; \
       fi

FROM python-base AS runtime

ENV PATH=/opt/botified-asr/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
       ca-certificates \
       ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/botified-asr /opt/botified-asr
COPY --from=builder \
     /build/LICENSE \
     /build/THIRD_PARTY_NOTICES \
     /usr/share/doc/botified-asr/
COPY config/container.yaml /etc/botified-asr/config.yaml

ARG BOTIFIED_ASR_VERSION

LABEL org.opencontainers.image.source=https://github.com/lzjever/botified-asr \
      org.opencontainers.image.version=$BOTIFIED_ASR_VERSION

RUN groupadd --gid 10001 botified-asr \
    && useradd \
       --uid 10001 \
       --gid 10001 \
       --no-create-home \
       --home-dir /nonexistent \
       --shell /usr/sbin/nologin \
       botified-asr \
    && mkdir -p \
       /data/state \
       /data/models \
    && chown -R 10001:10001 \
       /data

USER 10001:10001

ENTRYPOINT ["botified-asr"]
CMD ["--config", "/etc/botified-asr/config.yaml"]
