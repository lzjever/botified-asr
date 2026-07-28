from __future__ import annotations

import struct
import threading
from datetime import datetime, timedelta, timezone
from inspect import Parameter, signature
from pathlib import Path

import numpy as np
import pytest

from botified_asr import speaker_profiles
from botified_asr import storage as storage_module
from botified_asr.config import LimitsConfig


CREATED_AT = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
UPDATED_AT = datetime(2026, 7, 27, 12, 1, tzinfo=timezone.utc)
KEEP = object()
THREAD_TIMEOUT = 5.0
PROCESSOR_FINGERPRINT = "3" * 64


def _storage(path: Path) -> storage_module.Storage:
    quantum = storage_module.RESERVATION_QUANTUM
    limits = LimitsConfig(
        max_upload_bytes=quantum,
        sync_max_upload_bytes=quantum,
        max_active_uploads=2,
        max_job_storage_bytes=quantum,
        min_filesystem_free_bytes=1,
    )
    return storage_module.Storage(path, limits, current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _path: 1 << 40)


def _embedding(
    dimension: int = 2,
    axis: int = 0,
    values: np.ndarray | None = None,
) -> speaker_profiles.SpeakerEmbedding:
    if values is None:
        values = np.zeros(dimension, dtype=np.float32)
        values[axis] = 1.0
    return speaker_profiles.SpeakerEmbedding.from_numpy(values, dimension=dimension)


def _profile(
    ordinal: int = 0,
    *,
    profile_id: str | None = None,
    name: str | None = None,
    description: str | None = "metadata",
    embedding: speaker_profiles.SpeakerEmbedding | None = None,
    created_at: datetime = CREATED_AT,
    updated_at: datetime | None = None,
) -> speaker_profiles.SpeakerProfile:
    vector = embedding or _embedding()
    return speaker_profiles.SpeakerProfile(
        id=profile_id or f"{ordinal:08X}",
        name=name or f"person-{ordinal}",
        description=description,
        embedding=vector,
        embedding_model_id="funasr/campplus",
        embedding_model_revision="1" * 40,
        embedding_dimension=vector.dimension,
        embedding_policy_fingerprint="a" * 64,
        sample_count=2,
        created_at=created_at,
        updated_at=updated_at or created_at,
    )


def _replacement() -> speaker_profiles.SpeakerEmbeddingReplacement:
    return speaker_profiles.SpeakerEmbeddingReplacement(
        embedding=_embedding(3, axis=1),
        embedding_model_id="funasr/other-campplus",
        embedding_model_revision="2" * 40,
        embedding_dimension=3,
        embedding_policy_fingerprint="b" * 64,
        sample_count=5,
    )


def _update(
    name: str,
    *,
    description: object = KEEP,
    embedding: object = KEEP,
    updated_at: datetime = UPDATED_AT,
) -> speaker_profiles.SpeakerProfileUpdate:
    return speaker_profiles.SpeakerProfileUpdate(
        name=name,
        description=(
            speaker_profiles.KEEP_EXISTING if description is KEEP else description
        ),
        embedding=(speaker_profiles.KEEP_EXISTING if embedding is KEEP else embedding),
        updated_at=updated_at,
    )


def _embedding_state(profile: speaker_profiles.SpeakerProfile) -> tuple[object, ...]:
    return (
        profile.embedding.to_bytes(),
        profile.embedding_model_id,
        profile.embedding_model_revision,
        profile.embedding_dimension,
        profile.embedding_policy_fingerprint,
        profile.sample_count,
    )


@pytest.mark.parametrize("name", ("A", "Unknown A"))
def test_reserved_profile_name_has_a_typed_value_error(name: str) -> None:
    with pytest.raises(
        speaker_profiles.ReservedSpeakerProfileNameError
    ) as caught:
        speaker_profiles.canonicalize_speaker_profile_name(name)

    assert isinstance(caught.value, ValueError)


def test_round_trip_reopen_canonical_utc_and_stable_list_order(
    tmp_path: Path,
) -> None:
    plus_eight = timezone(timedelta(hours=8))
    first = _profile(
        2,
        name="first",
        embedding=_embedding(values=np.array([0.6, 0.8], dtype=np.float32)),
        created_at=datetime(2026, 7, 27, 20, 0, tzinfo=plus_eight),
        updated_at=datetime(2026, 7, 27, 20, 1, tzinfo=plus_eight),
    )
    low = _profile(10, name="second", created_at=UPDATED_AT)
    high = _profile(11, name="third", created_at=UPDATED_AT)
    storage = _storage(tmp_path)
    for profile in (high, first, low):
        assert storage.create_speaker_profile(profile) == profile
    assert storage.get_speaker_profile(first.id) == first
    assert storage.list_speaker_profiles() == (first, low, high)
    raw = storage._connection.execute(
        "SELECT embedding, created_at, updated_at FROM speaker_profiles WHERE id=?",
        (first.id,),
    ).fetchone()
    assert tuple(raw) == (
        struct.pack("<ff", 0.6, 0.8),
        "2026-07-27T12:00:00.000000Z",
        "2026-07-27T12:01:00.000000Z",
    )
    storage.close()

    reopened = _storage(tmp_path)
    try:
        assert reopened.get_speaker_profile(first.id) == first
        assert reopened.list_speaker_profiles() == (first, low, high)
    finally:
        reopened.close()


def test_typed_create_conflicts_leave_no_residual_write(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    original = _profile(name="Straße")
    storage.create_speaker_profile(original)
    try:
        with pytest.raises(
            storage_module.SpeakerProfileNameConflictError
        ) as name_error:
            storage.create_speaker_profile(_profile(1, name="STRAsse"))
        with pytest.raises(storage_module.SpeakerProfileIdCollisionError) as id_error:
            storage.create_speaker_profile(
                _profile(profile_id=original.id, name="different")
            )
        assert isinstance(name_error.value, storage_module.SpeakerProfileStorageError)
        assert isinstance(id_error.value, storage_module.SpeakerProfileStorageError)
        assert storage.list_speaker_profiles() == (original,)
    finally:
        storage.close()


@pytest.mark.parametrize(
    ("conflict", "error_name"),
    (
        ("casefold-name", "SpeakerProfileNameConflictError"),
        ("same-id", "SpeakerProfileIdCollisionError"),
    ),
)
def test_two_connections_resolve_create_conflicts_atomically(
    tmp_path: Path,
    conflict: str,
    error_name: str,
) -> None:
    if conflict == "casefold-name":
        profiles = (
            _profile(1, name="Straße"),
            _profile(2, name="STRAsse"),
        )
    else:
        profiles = (
            _profile(profile_id="000000AA", name="Alice"),
            _profile(profile_id="000000AA", name="Bob"),
        )
    stores = (_storage(tmp_path), _storage(tmp_path))
    barrier = threading.Barrier(2)
    outcomes: list[object] = []

    def create(
        storage: storage_module.Storage,
        profile: speaker_profiles.SpeakerProfile,
    ) -> None:
        try:
            barrier.wait(timeout=THREAD_TIMEOUT)
            outcomes.append(storage.create_speaker_profile(profile))
        except BaseException as error:
            outcomes.append(error)

    threads = tuple(
        threading.Thread(target=create, args=pair)
        for pair in zip(stores, profiles, strict=True)
    )
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=THREAD_TIMEOUT)
        assert all(not thread.is_alive() for thread in threads)
        successes = [
            result
            for result in outcomes
            if isinstance(result, speaker_profiles.SpeakerProfile)
        ]
        failures = [result for result in outcomes if result not in successes]
        assert len(successes) == len(failures) == 1
        assert isinstance(failures[0], getattr(storage_module, error_name))
        assert stores[0].list_speaker_profiles() in ((profiles[0],), (profiles[1],))
    finally:
        for storage in stores:
            storage.close()


def test_two_connections_cannot_race_past_atomic_256_cap(
    tmp_path: Path,
) -> None:
    first = _storage(tmp_path)
    for ordinal in range(255):
        first.create_speaker_profile(_profile(ordinal))
    second = _storage(tmp_path)
    barrier = threading.Barrier(2)
    outcomes: list[object] = []

    def create(storage: storage_module.Storage, ordinal: int) -> None:
        try:
            barrier.wait(timeout=THREAD_TIMEOUT)
            storage.create_speaker_profile(_profile(ordinal))
            outcomes.append("created")
        except BaseException as error:
            outcomes.append(error)

    threads = (
        threading.Thread(target=create, args=(first, 256)),
        threading.Thread(target=create, args=(second, 257)),
    )
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=THREAD_TIMEOUT)
        assert all(not thread.is_alive() for thread in threads)
        failures = [result for result in outcomes if result != "created"]
        assert outcomes.count("created") == 1
        assert len(failures) == 1
        assert isinstance(failures[0], storage_module.SpeakerProfileLimitReachedError)
        assert isinstance(failures[0], storage_module.SpeakerProfileStorageError)
        assert len(first.list_speaker_profiles()) == 256
    finally:
        first.close()
        second.close()


def test_update_description_tristate_keep_and_atomic_embedding_replace(
    tmp_path: Path,
) -> None:
    storage = _storage(tmp_path)
    original = _profile(name="Alice", description="original")
    old_embedding = _embedding_state(original)
    replacement = _replacement()
    storage.create_speaker_profile(original)
    try:
        preserved = storage.update_speaker_profile(original.id, _update("alice"))
        assert preserved.description == "original"
        assert _embedding_state(preserved) == old_embedding

        cleared = storage.update_speaker_profile(
            original.id,
            _update("alice", description=None, updated_at=UPDATED_AT),
        )
        assert cleared.description is None
        replaced = storage.update_speaker_profile(
            original.id,
            _update(
                "alice",
                description="replacement",
                embedding=replacement,
                updated_at=UPDATED_AT + timedelta(minutes=1),
            ),
        )
        assert replaced.description == "replacement"
        assert replaced.created_at == original.created_at
        assert _embedding_state(replaced) == (
            replacement.embedding.to_bytes(),
            replacement.embedding_model_id,
            replacement.embedding_model_revision,
            replacement.embedding_dimension,
            replacement.embedding_policy_fingerprint,
            replacement.sample_count,
        )
    finally:
        storage.close()
    reopened = _storage(tmp_path)
    try:
        assert reopened.get_speaker_profile(original.id) == replaced
    finally:
        reopened.close()


def test_update_commands_require_complete_fields_and_reject_invalid_changes(
    tmp_path: Path,
) -> None:
    replacement_parameters = signature(
        speaker_profiles.SpeakerEmbeddingReplacement
    ).parameters
    update_parameters = signature(speaker_profiles.SpeakerProfileUpdate).parameters
    assert tuple(replacement_parameters) == (
        "embedding",
        "embedding_model_id",
        "embedding_model_revision",
        "embedding_dimension",
        "embedding_policy_fingerprint",
        "sample_count",
    )
    assert tuple(update_parameters) == (
        "name",
        "description",
        "embedding",
        "updated_at",
    )
    assert all(
        parameter.default is Parameter.empty
        for parameter in (*replacement_parameters.values(), *update_parameters.values())
    )
    with pytest.raises(TypeError):
        speaker_profiles.SpeakerEmbeddingReplacement(embedding=_embedding())
    with pytest.raises(TypeError):
        speaker_profiles.SpeakerProfileUpdate(name="Alice")

    storage = _storage(tmp_path)
    original = _profile(name="Alice", updated_at=UPDATED_AT)
    storage.create_speaker_profile(original)
    invalid_embedding = speaker_profiles.SpeakerProfileUpdate(
        name="Alice",
        description=speaker_profiles.KEEP_EXISTING,
        embedding=object(),
        updated_at=UPDATED_AT + timedelta(minutes=1),
    )
    try:
        with pytest.raises(TypeError):
            storage.update_speaker_profile(original.id, invalid_embedding)
        assert storage.get_speaker_profile(original.id) == original

        stale = _update(
            "Alice",
            description="must roll back",
            updated_at=CREATED_AT + timedelta(seconds=30),
        )
        clamped = storage.update_speaker_profile(original.id, stale)
        assert clamped.description == "must roll back"
        assert clamped.updated_at == original.updated_at

        invalid_time = _update(
            "Alice",
            description="must not commit",
            updated_at=datetime(2026, 7, 27, 12, 2),
        )
        with pytest.raises(TypeError):
            storage.update_speaker_profile(original.id, invalid_time)
        assert storage.get_speaker_profile(original.id) == clamped
    finally:
        storage.close()


def test_concurrent_keep_updates_merge_from_latest_row(tmp_path: Path) -> None:
    first = _storage(tmp_path)
    original = _profile(name="Alice", description="original")
    first.create_speaker_profile(original)
    second = _storage(tmp_path)
    replacement = _replacement()
    barrier = threading.Barrier(2)
    outcomes: list[object] = []

    def update(
        storage: storage_module.Storage,
        command: speaker_profiles.SpeakerProfileUpdate,
    ) -> None:
        try:
            barrier.wait(timeout=THREAD_TIMEOUT)
            outcomes.append(storage.update_speaker_profile(original.id, command))
        except BaseException as error:
            outcomes.append(error)

    threads = (
        threading.Thread(
            target=update,
            args=(first, _update("Alice", description="new metadata")),
        ),
        threading.Thread(
            target=update,
            args=(second, _update("Alice", embedding=replacement)),
        ),
    )
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=THREAD_TIMEOUT)
        assert all(not thread.is_alive() for thread in threads)
        assert all(
            isinstance(result, speaker_profiles.SpeakerProfile) for result in outcomes
        )
        final = first.get_speaker_profile(original.id)
        assert final.description == "new metadata"
        assert final.embedding == replacement.embedding
        assert final.sample_count == replacement.sample_count
        assert final.updated_at == UPDATED_AT
    finally:
        first.close()
        second.close()


def test_update_conflict_not_found_and_delete_are_atomic(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    first = _profile(1, name="Alice")
    second = _profile(2, name="Bob")
    storage.create_speaker_profile(first)
    storage.create_speaker_profile(second)
    try:
        with pytest.raises(storage_module.SpeakerProfileNameConflictError):
            storage.update_speaker_profile(
                first.id,
                _update(
                    second.name,
                    description="must roll back",
                    embedding=_replacement(),
                ),
            )
        assert storage.get_speaker_profile(first.id) == first
        assert storage.get_speaker_profile("ZZZZZZZZ") is None
        assert storage.update_speaker_profile("ZZZZZZZZ", _update("missing")) is None
        assert storage.delete_speaker_profile("ZZZZZZZZ") is False
        assert storage.delete_speaker_profile(first.id) is True
    finally:
        storage.close()
    reopened = _storage(tmp_path)
    try:
        assert reopened.get_speaker_profile(first.id) is None
        assert reopened.delete_speaker_profile(first.id) is False
    finally:
        reopened.close()


def test_get_many_is_one_sorted_query_and_returns_detached_values(
    tmp_path: Path,
) -> None:
    storage = _storage(tmp_path)
    profiles = tuple(
        _profile(i, name=name) for i, name in enumerate(("Alice", "Bob", "Carol"), 1)
    )
    for profile in reversed(profiles):
        storage.create_speaker_profile(profile)
    queries: list[str] = []

    def trace(sql: str) -> None:
        normalized = sql.lstrip().upper()
        if normalized.startswith("SELECT") and "SPEAKER_PROFILES" in normalized:
            queries.append(sql)

    storage._connection.set_trace_callback(trace)
    try:
        assert storage.get_speaker_profiles_by_ids(()) == ()
        assert queries == []
        selected = storage.get_speaker_profiles_by_ids(
            (profiles[2].id, "ZZZZZZZZ", profiles[0].id)
        )
        assert selected == (profiles[0], profiles[2])
        assert len(queries) == 1
        storage._connection.set_trace_callback(None)
        storage.update_speaker_profile(profiles[0].id, _update("updated"))
        storage.delete_speaker_profile(profiles[2].id)
        assert selected == (profiles[0], profiles[2])
        view = selected[0].embedding.as_numpy()
        assert view.flags.writeable is False
        assert storage.get_speaker_profile(profiles[0].id).name == "updated"
    finally:
        storage._connection.set_trace_callback(None)
        storage.close()


@pytest.mark.parametrize(
    "corruption",
    ("name-key", "name", "description", "timestamp", "embedding"),
)
def test_all_read_paths_wrap_corrupt_rows_as_schema_errors(
    tmp_path: Path,
    corruption: str,
) -> None:
    storage = _storage(tmp_path)
    profile = _profile(name="Alice")
    storage.create_speaker_profile(profile)
    if corruption == "name-key":
        storage._connection.execute(
            "UPDATE speaker_profiles SET name_key='wrong' WHERE id=?", (profile.id,)
        )
    elif corruption == "name":
        storage._connection.execute(
            "UPDATE speaker_profiles SET name=' Alice ' WHERE id=?", (profile.id,)
        )
    elif corruption == "description":
        storage._connection.execute(
            "UPDATE speaker_profiles SET description='' WHERE id=?", (profile.id,)
        )
    elif corruption == "timestamp":
        storage._connection.execute(
            "UPDATE speaker_profiles SET created_at='not-a-time' WHERE id=?",
            (profile.id,),
        )
    else:
        storage._connection.execute(
            "UPDATE speaker_profiles SET embedding=? WHERE id=?",
            (struct.pack("<ff", 0.0, 0.0), profile.id),
        )
    readers = (
        lambda: storage.get_speaker_profile(profile.id),
        storage.list_speaker_profiles,
        lambda: storage.get_speaker_profiles_by_ids((profile.id,)),
    )
    try:
        for read in readers:
            with pytest.raises(storage_module.StorageSchemaError):
                read()
    finally:
        storage.close()
