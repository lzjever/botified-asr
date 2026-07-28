# syntax=docker/dockerfile:1

ARG BOTIFIED_ASR_VERSION

FROM python:3.11.13-slim-bookworm AS python-base

FROM python-base AS builder

ENV UV_PROJECT_ENVIRONMENT=/opt/botified-asr
WORKDIR /build

RUN python --version \
    && python -m pip install --no-cache-dir uv==0.9.26 \
    && uv --version

COPY pyproject.toml uv.lock ./
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
       fi \
    && /opt/botified-asr/bin/botified-asr --help >/dev/null \
    && /opt/botified-asr/bin/python -c \
       'import botified_asr, torch; assert not torch.cuda.is_available()'

FROM python-base AS runtime

ENV PATH=/opt/botified-asr/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
       ca-certificates \
       ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/botified-asr /opt/botified-asr

ARG BOTIFIED_ASR_VERSION

LABEL org.opencontainers.image.version=$BOTIFIED_ASR_VERSION

RUN groupadd --gid 10001 botified-asr \
    && useradd \
       --uid 10001 \
       --gid 10001 \
       --no-create-home \
       --home-dir /nonexistent \
       --shell /usr/sbin/nologin \
       botified-asr \
    && mkdir -p \
       /var/lib/botified-asr \
       /var/cache/botified-asr/models \
    && chown -R 10001:10001 \
       /var/lib/botified-asr \
       /var/cache/botified-asr \
    && python --version \
    && botified-asr --version >/dev/null \
    && python -c 'import botified_asr, torch; assert not torch.cuda.is_available()' \
    && ffmpeg -version >/dev/null \
    && ffprobe -version >/dev/null

USER 10001:10001

ENTRYPOINT ["botified-asr"]
