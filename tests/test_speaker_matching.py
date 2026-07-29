from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
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


def test_matcher_policy_and_values_are_frozen_and_strict() -> None:
    matching = _matching_module()
    policy = matching.KnownSpeakerMatchPolicy(0.31)
    match = matching.KnownSpeakerMatch("00000001", "Alice", 1.0)
    resolution = matching.SpeakerLabelResolution("A", match)
    mapping = matching.SpeakerLabelMapping((resolution,))
    with pytest.raises(FrozenInstanceError):
        policy.match_threshold = 0.5
    with pytest.raises(FrozenInstanceError):
        mapping.resolutions = ()
    assert not hasattr(policy, "top_two_margin")

    with pytest.raises(TypeError):
        matching.KnownSpeakerMatchPolicy(True)
    for value in (math.nan, math.inf, -1.01, 1.01):
        with pytest.raises(ValueError):
            matching.KnownSpeakerMatchPolicy(value)

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
        matching.KnownSpeakerMatchPolicy(1.0),
    ) == matching.SpeakerLabelMapping(())
    assert matching.match_selected_speakers(
        (),
        speaker_snapshot.SelectedSpeakerSnapshot(
            (_selected("00000001", "Alice", 1.0, 0.0),)
        ),
        matching.KnownSpeakerMatchPolicy(1.0),
    ) == matching.SpeakerLabelMapping(())


def test_unique_maximum_at_threshold_matches_but_below_or_exact_tie_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matching = _matching_module()
    cluster = _cluster("A", 1.0, 0.0)
    alice = _selected("00000001", "Alice", 1.0, 0.0)
    bob = _selected("00000002", "Bob", 0.0, 1.0)
    snapshot = speaker_snapshot.SelectedSpeakerSnapshot((alice, bob))
    policy = matching.KnownSpeakerMatchPolicy(0.31)

    def resolve(*similarities: float) -> matching.SpeakerLabelMapping:
        values = iter(similarities)
        monkeypatch.setattr(
            matching,
            "_cosine",
            lambda *_args: next(values),
        )
        return matching.match_selected_speakers(
            (cluster,),
            snapshot,
            policy,
        )

    assert resolve(0.31, 0.30).resolutions[0].match == (
        matching.KnownSpeakerMatch(alice.id, alice.name, 0.31)
    )
    assert resolve(math.nextafter(0.31, -math.inf), 0.30) == (
        matching.SpeakerLabelMapping(
            (matching.SpeakerLabelResolution("A", None),)
        )
    )
    assert resolve(0.8, 0.8) == matching.SpeakerLabelMapping(
        (matching.SpeakerLabelResolution("A", None),)
    )
    almost_best = math.nextafter(0.8, -math.inf)
    assert resolve(0.8, almost_best).resolutions[0].match == (
        matching.KnownSpeakerMatch(alice.id, alice.name, 0.8)
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
        matching.KnownSpeakerMatchPolicy(0.75),
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
            matching.KnownSpeakerMatchPolicy(0.0),
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
            matching.KnownSpeakerMatchPolicy(0.0),
        )
