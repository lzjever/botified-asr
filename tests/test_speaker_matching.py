from __future__ import annotations

import math
from dataclasses import MISSING, FrozenInstanceError, fields
from importlib import import_module
from types import ModuleType

import numpy as np
import pytest

from botified_asr import speaker_profiles, speaker_snapshot, speakers


DIMENSION = speakers.SPEAKER_EMBEDDING_DIMENSION


def _matching_module() -> ModuleType:
    return import_module("botified_asr.speaker_matching")


def _unit(*components: float, dimension: int = DIMENSION) -> tuple[float, ...]:
    values = np.zeros(dimension, dtype=np.float64)
    values[: len(components)] = components
    values /= np.linalg.norm(values)
    return tuple(float(value) for value in values)


def _cluster(
    label: str = "A",
    *components: float,
) -> speakers.AnonymousSpeakerCluster:
    return speakers.AnonymousSpeakerCluster(
        label=label,
        centroid=_unit(*(components or (1.0, 0.0))),
    )


def _embedding(
    *components: float,
    dimension: int = DIMENSION,
) -> speaker_profiles.SpeakerEmbedding:
    values = np.zeros(dimension, dtype=np.float32)
    values[: len(components)] = components
    values /= np.linalg.norm(values)
    return speaker_profiles.SpeakerEmbedding.from_numpy(
        values,
        dimension=dimension,
    )


def _selected(
    profile_id: str,
    name: str,
    *components: float,
    dimension: int = DIMENSION,
) -> speaker_snapshot.SelectedSpeaker:
    return speaker_snapshot.SelectedSpeaker(
        id=profile_id,
        name=name,
        embedding=_embedding(
            *(components or (1.0, 0.0)),
            dimension=dimension,
        ),
    )


def _cosine(
    cluster: speakers.AnonymousSpeakerCluster,
    selected: speaker_snapshot.SelectedSpeaker,
) -> float:
    left = np.asarray(cluster.centroid, dtype=np.float64)
    right = selected.embedding.as_numpy().astype(np.float64)
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def test_matcher_types_are_exact_required_frozen_slots_and_policy_is_strict() -> None:
    matching = _matching_module()
    expected_fields = {
        matching.KnownSpeakerMatchPolicy: (
            "match_threshold",
            "top_two_margin",
        ),
        matching.KnownSpeakerMatch: (
            "speaker_id",
            "speaker_name",
            "similarity",
        ),
        matching.SpeakerLabelResolution: (
            "anonymous_speaker",
            "match",
        ),
        matching.SpeakerLabelMapping: ("resolutions",),
    }
    for data_type, expected_names in expected_fields.items():
        data_fields = fields(data_type)
        assert tuple(item.name for item in data_fields) == expected_names
        assert all(
            item.default is MISSING and item.default_factory is MISSING
            for item in data_fields
        )

    policy_type = matching.KnownSpeakerMatchPolicy
    assert not any(type(value) is policy_type for value in vars(matching).values())
    policy = matching.KnownSpeakerMatchPolicy(0.0, 0.0)
    match = matching.KnownSpeakerMatch("00000001", "Alice", 1.0)
    resolution = matching.SpeakerLabelResolution("A", match)
    mapping = matching.SpeakerLabelMapping((resolution,))
    for value in (policy, match, resolution, mapping):
        assert not hasattr(value, "__dict__")
    with pytest.raises(FrozenInstanceError):
        policy.match_threshold = 0.5
    with pytest.raises(FrozenInstanceError):
        mapping.resolutions = ()

    for kwargs in (
        {"match_threshold": True, "top_two_margin": 0.0},
        {"match_threshold": 0.0, "top_two_margin": False},
    ):
        with pytest.raises(TypeError):
            matching.KnownSpeakerMatchPolicy(**kwargs)
    for kwargs in (
        {"match_threshold": math.nan, "top_two_margin": 0.0},
        {"match_threshold": 0.0, "top_two_margin": math.inf},
        {"match_threshold": -1.01, "top_two_margin": 0.0},
        {"match_threshold": 1.01, "top_two_margin": 0.0},
        {"match_threshold": 0.0, "top_two_margin": -0.01},
        {"match_threshold": 0.0, "top_two_margin": 2.01},
    ):
        with pytest.raises(ValueError):
            matching.KnownSpeakerMatchPolicy(**kwargs)

    error = matching.SpeakerMatchInputError()
    assert isinstance(error, ValueError)
    assert not hasattr(error, "code")


def test_empty_selected_snapshot_is_anonymous_mode_with_no_mapping() -> None:
    matching = _matching_module()
    clusters = (
        _cluster("A", 1.0, 0.0),
        _cluster("B", 0.0, 1.0),
    )

    assert matching.match_selected_speakers(
        clusters,
        speaker_snapshot.SelectedSpeakerSnapshot(()),
        matching.KnownSpeakerMatchPolicy(1.0, 2.0),
    ) == matching.SpeakerLabelMapping(())
    assert matching.match_selected_speakers(
        (),
        speaker_snapshot.SelectedSpeakerSnapshot(
            (_selected("00000001", "Alice", 1.0, 0.0),)
        ),
        matching.KnownSpeakerMatchPolicy(1.0, 2.0),
    ) == matching.SpeakerLabelMapping(())


def test_single_candidate_threshold_is_inclusive_without_margin() -> None:
    matching = _matching_module()
    cluster = _cluster("A", 1.0, 0.0)
    selected = _selected("00000001", "Alice", 0.0, 1.0)
    snapshot = speaker_snapshot.SelectedSpeakerSnapshot((selected,))
    similarity = _cosine(cluster, selected)
    assert similarity == 0.0

    matched = matching.match_selected_speakers(
        (cluster,),
        snapshot,
        matching.KnownSpeakerMatchPolicy(0.0, 2.0),
    )
    assert matched == matching.SpeakerLabelMapping(
        (
            matching.SpeakerLabelResolution(
                "A",
                matching.KnownSpeakerMatch(
                    selected.id,
                    selected.name,
                    similarity,
                ),
            ),
        )
    )

    rejected = matching.match_selected_speakers(
        (cluster,),
        snapshot,
        matching.KnownSpeakerMatchPolicy(
            math.nextafter(0.0, math.inf),
            0.0,
        ),
    )
    assert rejected == matching.SpeakerLabelMapping(
        (matching.SpeakerLabelResolution("A", None),)
    )


def test_multi_candidate_margin_is_inclusive_but_ties_are_unknown() -> None:
    matching = _matching_module()
    cluster = _cluster("A", 1.0, 0.0)
    best = _selected("00000001", "Alice", 1.0, 0.0)
    second = _selected("00000002", "Bob", 0.0, 1.0)
    snapshot = speaker_snapshot.SelectedSpeakerSnapshot((best, second))
    best_similarity = _cosine(cluster, best)
    margin = best_similarity - _cosine(cluster, second)
    assert best_similarity == 1.0
    assert margin == 1.0

    at_margin = matching.match_selected_speakers(
        (cluster,),
        snapshot,
        matching.KnownSpeakerMatchPolicy(1.0, 1.0),
    )
    assert at_margin.resolutions[0].match == matching.KnownSpeakerMatch(
        best.id,
        best.name,
        best_similarity,
    )
    above_margin = matching.match_selected_speakers(
        (cluster,),
        snapshot,
        matching.KnownSpeakerMatchPolicy(
            1.0,
            math.nextafter(1.0, math.inf),
        ),
    )
    assert above_margin == matching.SpeakerLabelMapping(
        (matching.SpeakerLabelResolution("A", None),)
    )

    tied = speaker_snapshot.SelectedSpeakerSnapshot(
        (
            _selected("00000001", "Alice", 1.0, 0.0),
            _selected("00000002", "Bob", 1.0, 0.0),
        )
    )
    tie_result = matching.match_selected_speakers(
        (cluster,),
        tied,
        matching.KnownSpeakerMatchPolicy(1.0, 0.0),
    )
    assert tie_result == matching.SpeakerLabelMapping(
        (matching.SpeakerLabelResolution("A", None),)
    )


def test_multiple_clusters_may_match_the_same_selected_speaker() -> None:
    matching = _matching_module()
    selected = _selected("00000001", "Alice", 1.0, 0.0)
    result = matching.match_selected_speakers(
        (
            _cluster("A", 1.0, 0.0),
            _cluster("B", 0.8, 0.6),
        ),
        speaker_snapshot.SelectedSpeakerSnapshot((selected,)),
        matching.KnownSpeakerMatchPolicy(0.75, 2.0),
    )
    assert result.resolutions[0].match == matching.KnownSpeakerMatch(
        selected.id,
        selected.name,
        1.0,
    )
    assert result.resolutions[1].anonymous_speaker == "B"
    assert result.resolutions[1].match is not None
    assert result.resolutions[1].match.speaker_id == selected.id
    assert result.resolutions[1].match.speaker_name == selected.name
    assert result.resolutions[1].match.similarity == pytest.approx(0.8)
    assert tuple(item.anonymous_speaker for item in result.resolutions) == (
        "A",
        "B",
    )


def test_nonfinite_cluster_is_rejected() -> None:
    matching = _matching_module()
    valid = _unit(1.0, 0.0)
    nonfinite = list(valid)
    nonfinite[0] = math.nan
    cluster = speakers.AnonymousSpeakerCluster("A", tuple(nonfinite))
    snapshot = speaker_snapshot.SelectedSpeakerSnapshot(
        (_selected("00000001", "Alice", 1.0, 0.0),)
    )

    with pytest.raises(matching.SpeakerMatchInputError):
        matching.match_selected_speakers(
            (cluster,),
            snapshot,
            matching.KnownSpeakerMatchPolicy(0.0, 0.0),
        )


def test_selected_embedding_dimension_must_match_clusters() -> None:
    matching = _matching_module()
    snapshot = speaker_snapshot.SelectedSpeakerSnapshot(
        (
            _selected(
                "00000001",
                "Alice",
                1.0,
                0.0,
                dimension=2,
            ),
        )
    )

    with pytest.raises(matching.SpeakerMatchInputError):
        matching.match_selected_speakers(
            (_cluster("A", 1.0, 0.0),),
            snapshot,
            matching.KnownSpeakerMatchPolicy(0.0, 0.0),
        )
