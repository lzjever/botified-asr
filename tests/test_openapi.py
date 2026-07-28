from __future__ import annotations

import json
import os
import subprocess
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = PROJECT_ROOT / "scripts" / "generate-openapi"
EXPLICIT_ROUTES = {
    ("GET", "/health/live"),
    ("GET", "/health/ready"),
    ("GET", "/v1/models"),
    ("GET", "/v1/models/{model_id}"),
    ("GET", "/v1/speakers"),
    ("POST", "/v1/speakers"),
    ("GET", "/v1/speakers/{speaker_id}"),
    ("PUT", "/v1/speakers/{speaker_id}"),
    ("DELETE", "/v1/speakers/{speaker_id}"),
    ("POST", "/v1/audio/transcriptions"),
    ("GET", "/v1/audio/transcriptions/{job_id}"),
    ("DELETE", "/v1/audio/transcriptions/{job_id}"),
}


def _generate(output: Path) -> dict[str, object]:
    completed = subprocess.run(
        [GENERATOR, output],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(output.read_bytes())


def test_openapi_matches_runtime_routes_and_security(tmp_path: Path) -> None:
    document = _generate(tmp_path / "openapi.json")

    assert "EXPLICIT_ROUTES" not in GENERATOR.read_text(encoding="utf-8")
    operations = {
        (method.upper(), path)
        for path, path_item in document["paths"].items()
        for method in path_item
    }
    assert document["openapi"] == "3.1.0"
    assert operations == EXPLICIT_ROUTES
    assert "/docs" not in document["paths"]
    assert "/openapi.json" not in document["paths"]
    assert document["security"] == [{"BearerAuth": []}]
    assert document["paths"]["/health/live"]["get"]["security"] == []
    assert document["paths"]["/health/live"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/LiveHealth"}
    assert document["paths"]["/health/ready"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/ReadyHealth"}
    assert document["components"]["schemas"]["LiveHealth"]["properties"]["status"][
        "const"
    ] == "ok"
    assert document["components"]["schemas"]["ReadyHealth"]["properties"]["status"][
        "const"
    ] == "ready"
    assert document["components"]["securitySchemes"]["BearerAuth"] == {
        "type": "http",
        "scheme": "bearer",
    }


def test_openapi_covers_transcription_job_and_speaker_boundaries(
    tmp_path: Path,
) -> None:
    document = _generate(tmp_path / "openapi.json")
    schemas = document["components"]["schemas"]
    transcription = document["paths"]["/v1/audio/transcriptions"]["post"]
    multipart = transcription["requestBody"]["content"]["multipart/form-data"][
        "schema"
    ]

    from botified_asr import canonical_options, speaker_profiles
    from botified_asr.api import MAX_SPEAKER_SAMPLE_BYTES

    assert len(multipart["oneOf"]) == 3
    variants = {variant["title"]: variant for variant in multipart["oneOf"]}
    standard = variants["SenseVoice JSON request"]
    text = variants["SenseVoice text request"]
    diarized = variants["SenseVoice diarized request"]
    assert standard["required"] == ["file", "model"]
    assert text["required"] == ["file", "model", "response_format"]
    assert diarized["required"] == [
        "file",
        "model",
        "response_format",
        "chunking_strategy",
    ]
    assert standard["properties"]["model"]["const"] == "sensevoice"
    assert standard["properties"]["response_format"]["enum"] == [
        "json",
        "verbose_json",
    ]
    assert text["properties"]["response_format"]["const"] == "text"
    assert "include[]" not in text["properties"]
    assert "known_speaker_ids[]" not in text["properties"]
    assert diarized["properties"]["model"]["const"] == "sensevoice-diarize"
    assert diarized["properties"]["response_format"]["const"] == "diarized_json"
    assert diarized["properties"]["chunking_strategy"]["const"] == "auto"
    assert "include[]" in diarized["properties"]
    assert "known_speaker_ids[]" in diarized["properties"]
    assert all(variant["additionalProperties"] is False for variant in variants.values())
    assert "deployment-configured" in transcription["description"]
    assert "sensevoice-diarize" in multipart["description"]
    assert "chunking_strategy=auto" in multipart["description"]
    assert "response_format=diarized_json" in multipart["description"]
    assert "response_format=text" in multipart["description"]
    assert "known_speaker_ids[]" in multipart["description"]
    assert "maxLength" not in standard["properties"]["file"]
    assert standard["properties"]["language"]["enum"] == list(
        canonical_options.LANGUAGE_VALUES
    )
    assert standard["properties"]["chunking_strategy"]["enum"] == list(
        canonical_options.CHUNKING_STRATEGY_VALUES
    )
    assert standard["properties"]["include[]"]["items"]["enum"] == list(
        canonical_options.INCLUDE_VALUES
    )
    assert diarized["properties"]["known_speaker_ids[]"]["items"]["pattern"] == (
        "^[0-9A-HJKMNP-TV-Z]{8}$"
    )
    assert set(transcription["responses"]["200"]["content"]) == {
        "application/json",
        "text/plain",
    }
    json_result = {"$ref": "#/components/schemas/JsonTranscriptionResult"}
    assert transcription["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == json_result
    assert {
        item["$ref"] for item in schemas["JsonTranscriptionResult"]["oneOf"]
    } == {
        "#/components/schemas/SimpleTranscriptionResult",
        "#/components/schemas/VerboseTranscriptionResult",
        "#/components/schemas/DiarizedTranscriptionResult",
    }
    assert schemas["VerboseTranscriptionResult"]["properties"]["segments"][
        "items"
    ] == {"$ref": "#/components/schemas/TranscriptionSegment"}
    assert schemas["DiarizedTranscriptionResult"]["properties"]["segments"][
        "items"
    ] == {"$ref": "#/components/schemas/DiarizedTranscriptionSegment"}
    assert schemas["SimpleTranscriptionResult"]["properties"]["funasr"] == {
        "$ref": "#/components/schemas/FunASRAnnotations"
    }
    assert set(schemas["FunASRAnnotations"]["properties"]) == {
        "emotion",
        "audio_events",
    }
    assert schemas["DiarizedTranscriptionSegment"]["properties"]["funasr"] == {
        "$ref": "#/components/schemas/FunASRSpeakerMatch"
    }
    assert set(transcription["responses"]["202"]["headers"]) == {
        "Location",
        "Preference-Applied",
    }

    job_path = document["paths"]["/v1/audio/transcriptions/{job_id}"]
    assert job_path["get"]["responses"]["202"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/ActiveJob"}
    assert len(schemas["ActiveJob"]["oneOf"]) == 2
    assert len(schemas["TerminalJob"]["oneOf"]) == 3
    assert schemas["SucceededJob"]["properties"]["result"] == json_result
    assert {"202", "204"}.issubset(job_path["delete"]["responses"])

    model_schema = schemas["Model"]["properties"]
    assert model_schema["id"]["enum"] == ["sensevoice", "sensevoice-diarize"]
    assert model_schema["created"] == {"type": "integer", "const": 1785024000}
    assert model_schema["owned_by"] == {
        "type": "string",
        "const": "botified-asr",
    }
    assert document["paths"]["/v1/models/{model_id}"]["get"]["parameters"][0][
        "schema"
    ]["enum"] == ["sensevoice", "sensevoice-diarize"]

    speaker_collection = document["paths"]["/v1/speakers"]
    speaker_item = document["paths"]["/v1/speakers/{speaker_id}"]
    post_schema = speaker_collection["post"]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]
    put_schema = speaker_item["put"]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]
    assert "samples[]" in post_schema["required"]
    assert "samples[]" not in put_schema["required"]
    for schema in (post_schema, put_schema):
        name = schema["properties"]["name"]
        assert set(name) == {"type", "description"}
        assert "trim" in name["description"]
        assert "1 to 80" in name["description"]
        samples = schema["properties"]["samples[]"]
        assert (samples["minItems"], samples["maxItems"]) == (2, 5)
        assert samples["items"]["minLength"] == 1
        assert samples["items"]["maxLength"] == MAX_SPEAKER_SAMPLE_BYTES
        assert "non-empty" in samples["items"]["description"]
        assert "20 MiB" in samples["items"]["description"]
        assert schema["properties"]["description"]["type"] == "string"
        assert schema["properties"]["description"]["maxLength"] == (
            speaker_profiles.SPEAKER_PROFILE_DESCRIPTION_MAX_CHARS
        )
    assert "empty" in post_schema["properties"]["description"]["description"]
    assert "Omit to preserve" in put_schema["properties"]["description"]["description"]
    assert "embedding" not in schemas["Speaker"]["properties"]
    assert schemas["Speaker"]["properties"]["name"]["maxLength"] == (
        speaker_profiles.SPEAKER_PROFILE_NAME_MAX_CHARS
    )
    assert schemas["Speaker"]["properties"]["description"]["maxLength"] == (
        speaker_profiles.SPEAKER_PROFILE_DESCRIPTION_MAX_CHARS
    )
    assert "enum" not in schemas["ErrorDetail"]["properties"]["code"]
    assert schemas["ErrorDetail"]["properties"]["type"]["enum"] == [
        "authentication_error",
        "invalid_request_error",
        "rate_limit_error",
        "server_error",
    ]
    assert speaker_collection["post"]["responses"]["201"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/Speaker"}
    assert set(speaker_collection["post"]["responses"]) == {
        "201",
        "400",
        "401",
        "409",
        "413",
        "429",
        "500",
        "503",
    }
    assert set(speaker_item["put"]["responses"]) == {
        "200",
        "400",
        "401",
        "404",
        "409",
        "413",
        "429",
        "500",
        "503",
    }
    assert set(speaker_collection["get"]["responses"]) == {
        "200",
        "401",
        "500",
        "503",
    }
    assert set(speaker_item["get"]["responses"]) == {
        "200",
        "401",
        "404",
        "500",
        "503",
    }
    assert set(speaker_item["delete"]["responses"]) == {
        "204",
        "401",
        "404",
        "500",
        "503",
    }
    assert "500" in job_path["delete"]["responses"]


def test_openapi_cli_is_deterministic_versioned_and_writes_safely(
    tmp_path: Path,
) -> None:
    output = tmp_path / "openapi.json"
    output.write_text("old", encoding="utf-8")
    document = _generate(output)
    first = output.read_bytes()
    _generate(output)

    version = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    assert document["info"]["version"] == version
    assert first == output.read_bytes()
    assert first == (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    assert os.access(GENERATOR, os.X_OK)
    assert not list(tmp_path.glob(".openapi.json.*.tmp"))

    target = tmp_path / "target"
    target.write_text("keep", encoding="utf-8")
    symlink = tmp_path / "linked.json"
    symlink.symlink_to(target)
    rejected = subprocess.run(
        [GENERATOR, symlink],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert target.read_text(encoding="utf-8") == "keep"
