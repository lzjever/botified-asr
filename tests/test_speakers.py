from __future__ import annotations

from dataclasses import MISSING, FrozenInstanceError, fields, replace

import numpy as np
import pytest

from botified_asr import speakers
from botified_asr.errors import PipelineError


EMBEDDING_DIMENSION = 192


def _synthetic_nonproduction_embedding_policy(
    **changes: object,
) -> speakers.SpeakerEmbeddingPolicy:
    values: dict[str, object] = {
        "model_id": "synthetic/test-campplus",
        "model_revision": "1" * 40,
        "embedding_dimension": 192,
        "sample_rate": 16_000,
        "downmix_policy_version": "mono-average-v1",
        "window_samples": 24_000,
        "window_shift_samples": 12_000,
        "padding_policy_version": "right-zero-pad-v1",
        "normalization_policy_version": "int16-div-32768-l2-v1",
        "enrollment_aggregation_policy_version": ("sample-centroid-equal-average-v1"),
    }
    values.update(changes)
    return speakers.SpeakerEmbeddingPolicy(**values)  # type: ignore[arg-type]


def _clustering_policy(
    *,
    pruning_p: float = 1.0,
    low_frequency_beta: float = 2.0,
    normalized_gap_gamma: float = 0.5,
) -> speakers.AnonymousSpeakerClusteringPolicy:
    return speakers.AnonymousSpeakerClusteringPolicy(
        pruning_p,
        low_frequency_beta,
        normalized_gap_gamma,
    )


def _embedding_rows(*rows: tuple[float, ...]) -> np.ndarray:
    values = np.zeros((len(rows), EMBEDDING_DIMENSION), dtype=np.float32)
    for index, row in enumerate(rows):
        values[index, : len(row)] = row
    if len(values):
        values /= np.linalg.norm(values, axis=1, keepdims=True)
    return np.ascontiguousarray(values)


def _common_component_rows(count: int) -> np.ndarray:
    values = np.zeros((count, EMBEDDING_DIMENSION), dtype=np.float32)
    values[:, 0] = 1.0
    values[np.arange(count), np.arange(1, count + 1)] = 1.0
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return np.ascontiguousarray(values)


@pytest.mark.parametrize(
    ("ordinal", "label"),
    (
        (0, "A"),
        (25, "Z"),
        (26, "AA"),
        (31, "AF"),
        (32, "AG"),
        (39, "AN"),
    ),
)
def test_anonymous_speaker_labels_are_algorithmic_and_strict(
    ordinal: int,
    label: str,
) -> None:
    assert speakers.anonymous_speaker_label(ordinal) == label
    assert speakers.is_anonymous_speaker_label(label)


@pytest.mark.parametrize(
    "value",
    (None, True, 1, "", "a", "A0", "Unknown A", " A", "A ", "Ａ"),
)
def test_anonymous_speaker_label_validation_rejects_noncanonical_values(
    value: object,
) -> None:
    assert not speakers.is_anonymous_speaker_label(value)


def test_embedding_policy_has_only_explicit_required_frozen_fields() -> None:
    expected_names = (
        "model_id",
        "model_revision",
        "embedding_dimension",
        "sample_rate",
        "downmix_policy_version",
        "window_samples",
        "window_shift_samples",
        "padding_policy_version",
        "normalization_policy_version",
        "enrollment_aggregation_policy_version",
    )
    policy_fields = fields(speakers.SpeakerEmbeddingPolicy)

    assert tuple(item.name for item in policy_fields) == expected_names
    assert all(
        item.default is MISSING and item.default_factory is MISSING
        for item in policy_fields
    )

    policy = _synthetic_nonproduction_embedding_policy()
    with pytest.raises(FrozenInstanceError):
        policy.sample_rate = 8_000  # type: ignore[misc]


def test_embedding_policy_canonical_bytes_and_fingerprint_are_exact() -> None:
    policy = _synthetic_nonproduction_embedding_policy()

    assert policy.canonical_bytes == (
        b'{"downmix_policy_version":"mono-average-v1",'
        b'"embedding_dimension":192,'
        b'"enrollment_aggregation_policy_version":'
        b'"sample-centroid-equal-average-v1",'
        b'"model_id":"synthetic/test-campplus",'
        b'"model_revision":"1111111111111111111111111111111111111111",'
        b'"normalization_policy_version":"int16-div-32768-l2-v1",'
        b'"padding_policy_version":"right-zero-pad-v1",'
        b'"sample_rate":16000,"window_samples":24000,'
        b'"window_shift_samples":12000}'
    )
    assert policy.canonical_bytes.decode("utf-8").encode("utf-8") == (
        policy.canonical_bytes
    )
    assert policy.fingerprint == (
        "7887ccc2255b7894bc2eddd01d9bd7a17fb80e7587864f4ee3d7020a64a8b197"
    )
    assert len(policy.fingerprint) == 64
    assert policy.fingerprint == policy.fingerprint.lower()


def test_embedding_policy_fingerprint_is_stable_and_every_field_is_bound() -> None:
    policy = _synthetic_nonproduction_embedding_policy()
    assert _synthetic_nonproduction_embedding_policy() == policy
    assert _synthetic_nonproduction_embedding_policy().fingerprint == (
        policy.fingerprint
    )

    valid_changes: dict[str, object] = {
        "model_id": "synthetic/other-campplus",
        "model_revision": "2" * 40,
        "embedding_dimension": 256,
        "sample_rate": 8_000,
        "downmix_policy_version": "mono-left-v1",
        "window_samples": 23_000,
        "window_shift_samples": 11_000,
        "padding_policy_version": "right-zero-pad-v2",
        "normalization_policy_version": "int16-div-32768-l2-v2",
        "enrollment_aggregation_policy_version": ("sample-centroid-equal-average-v2"),
    }

    changed_fingerprints = {
        replace(policy, **{name: value}).fingerprint
        for name, value in valid_changes.items()
    }
    assert policy.fingerprint not in changed_fingerprints
    assert len(changed_fingerprints) == len(valid_changes)


def test_embedding_policy_integer_fields_are_strict_positive_and_ordered() -> None:
    integer_fields = (
        "embedding_dimension",
        "sample_rate",
        "window_samples",
        "window_shift_samples",
    )
    for name in integer_fields:
        for invalid in (True, 1.0, "1"):
            with pytest.raises(TypeError):
                _synthetic_nonproduction_embedding_policy(**{name: invalid})
        for invalid in (0, -1):
            with pytest.raises(ValueError):
                _synthetic_nonproduction_embedding_policy(**{name: invalid})

    with pytest.raises(ValueError):
        _synthetic_nonproduction_embedding_policy(
            window_samples=12_000,
            window_shift_samples=12_001,
        )


def test_embedding_policy_rejects_invalid_identity_and_processing_values() -> None:
    for invalid in (None, True, 1):
        with pytest.raises(TypeError):
            _synthetic_nonproduction_embedding_policy(model_id=invalid)
        with pytest.raises(TypeError):
            _synthetic_nonproduction_embedding_policy(model_revision=invalid)

    for invalid_model_id in (
        "",
        "campplus",
        "/campplus",
        "synthetic/",
        "synthetic//campplus",
        "synthetic/campplus/extra",
        "../campplus",
        "synthetic/..",
    ):
        with pytest.raises(ValueError):
            _synthetic_nonproduction_embedding_policy(model_id=invalid_model_id)

    for invalid_revision in ("", "1" * 39, "1" * 41, "G" * 40, "A" * 40):
        with pytest.raises(ValueError):
            _synthetic_nonproduction_embedding_policy(model_revision=invalid_revision)

    version_fields = (
        "downmix_policy_version",
        "padding_policy_version",
        "normalization_policy_version",
        "enrollment_aggregation_policy_version",
    )
    for invalid in (None, True, 1):
        for name in version_fields:
            with pytest.raises(TypeError):
                _synthetic_nonproduction_embedding_policy(**{name: invalid})

    for invalid_token in (
        "",
        "mono-average",
        "mono-average-v0",
        "Mono-average-v1",
        "mono_average_v1",
        "mono-平均-v1",
        " mono-average-v1",
        "mono-average-v1 ",
    ):
        for name in version_fields:
            with pytest.raises(ValueError):
                _synthetic_nonproduction_embedding_policy(**{name: invalid_token})


def test_finalized_cluster_dto_is_exact_and_frozen() -> None:
    cluster_fields = fields(speakers.AnonymousSpeakerCluster)
    assert tuple(item.name for item in cluster_fields) == ("label", "centroid")
    assert all(
        item.default is MISSING and item.default_factory is MISSING
        for item in cluster_fields
    )
    cluster = speakers.AnonymousSpeakerCluster(
        label="AG",
        centroid=(1.0,) + (0.0,) * (EMBEDDING_DIMENSION - 1),
    )
    assert not hasattr(cluster, "__dict__")
    with pytest.raises(FrozenInstanceError):
        cluster.label = "AH"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        cluster.centroid = ()  # type: ignore[misc]


@pytest.mark.parametrize("ordinal", (True, 1.0, "1"))
def test_anonymous_speaker_label_requires_an_exact_integer(ordinal: object) -> None:
    with pytest.raises(TypeError):
        speakers.anonymous_speaker_label(ordinal)  # type: ignore[arg-type]


def test_anonymous_speaker_label_rejects_negative_ordinals() -> None:
    with pytest.raises(ValueError):
        speakers.anonymous_speaker_label(-1)


def test_clustering_policy_and_input_contract_are_strict_and_frozen() -> None:
    policy_fields = fields(speakers.AnonymousSpeakerClusteringPolicy)
    assert tuple(item.name for item in policy_fields) == (
        "pruning_p",
        "low_frequency_beta",
        "normalized_gap_gamma",
    )
    policy = _clustering_policy()
    assert not hasattr(policy, "__dict__")
    with pytest.raises(FrozenInstanceError):
        policy.pruning_p = 0.5  # type: ignore[misc]
    result = speakers.AnonymousSpeakerClusteringResult((), ())
    assert tuple(item.name for item in fields(result)) == (
        "window_cluster_ordinals",
        "clusters",
    )
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.clusters = ()  # type: ignore[misc]

    for values, error_type in (
        ((True, 1.0, 0.5), TypeError),
        ((0.5, False, 0.5), TypeError),
        ((0.5, 1.0, "0.5"), TypeError),
        ((-0.01, 1.0, 0.5), ValueError),
        ((1.01, 1.0, 0.5), ValueError),
        ((0.5, 0.0, 0.5), ValueError),
        ((0.5, 1.0, 0.0), ValueError),
        ((0.5, 1.0, 1.01), ValueError),
        ((float("nan"), 1.0, 0.5), ValueError),
    ):
        with pytest.raises(error_type):
            speakers.AnonymousSpeakerClusteringPolicy(*values)  # type: ignore[arg-type]

    valid = _embedding_rows((1.0, 0.0), (0.0, 1.0))
    invalid_inputs: tuple[object, ...] = (
        valid.tolist(),
        valid.astype(np.float64),
        np.empty((0, EMBEDDING_DIMENSION - 1), dtype=np.float32),
        np.zeros((1, EMBEDDING_DIMENSION), dtype=np.float32),
        np.full((1, EMBEDDING_DIMENSION), np.nan, dtype=np.float32),
        np.ascontiguousarray(valid * np.float32(0.5)),
        np.zeros((2, EMBEDDING_DIMENSION * 2), dtype=np.float32)[:, ::2],
    )
    messages: set[str] = set()
    for invalid in invalid_inputs:
        with pytest.raises(PipelineError) as caught:
            speakers.cluster_anonymous_speakers(
                invalid,  # type: ignore[arg-type]
                policy=policy,
            )
        assert caught.value.code == "invalid_model_output"
        messages.add(str(caught.value))
    assert len(messages) == 1


@pytest.mark.parametrize("count", (0, 1, 2))
def test_trivial_anchor_counts_return_direct_canonical_results_without_eigh(
    count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = _embedding_rows(*((1.0, 0.0),) * count)
    monkeypatch.setattr(
        speakers.np.linalg,
        "eigh",
        lambda _matrix: (_ for _ in ()).throw(
            AssertionError("N <= 2 must not run spectral clustering")
        ),
    )

    result = speakers.cluster_anonymous_speakers(
        embeddings,
        policy=_clustering_policy(),
    )

    assert result.window_cluster_ordinals == (0,) * count
    if count == 0:
        assert result.clusters == ()
    else:
        assert tuple(cluster.label for cluster in result.clusters) == ("A",)
        assert result.clusters[0].centroid == tuple(
            float(value) for value in embeddings[0]
        )
    if count == 2:
        with pytest.raises(PipelineError) as caught:
            speakers.cluster_anonymous_speakers(
                _embedding_rows((1.0, 0.0), (-1.0, 0.0)),
                policy=_clustering_policy(),
            )
        assert caught.value.code == "invalid_model_output"


def test_signed_affinity_stable_pruning_float64_and_scaled_eigen_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_eigh = np.linalg.eigh
    captured: list[np.ndarray] = []

    def capture(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        captured.append(matrix.copy())
        return original_eigh(matrix)

    monkeypatch.setattr(speakers.np.linalg, "eigh", capture)
    signed = _embedding_rows((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0))
    signed_result = speakers.cluster_anonymous_speakers(
        signed,
        policy=_clustering_policy(
            pruning_p=1.0,
            low_frequency_beta=0.1,
        ),
    )
    assert signed_result.window_cluster_ordinals == (0, 0, 0)
    assert captured[-1].dtype == np.float64
    np.testing.assert_array_equal(
        captured[-1],
        np.array(
            (
                (1.0, 1.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
            dtype=np.float64,
        ),
    )

    pruned_result = speakers.cluster_anonymous_speakers(
        _common_component_rows(7),
        policy=_clustering_policy(
            pruning_p=0.0,
            low_frequency_beta=1e-9,
        ),
    )
    assert pruned_result.window_cluster_ordinals == (0,) * 7
    pruned_laplacian = captured[-1]
    assert pruned_laplacian[0, 1] == pytest.approx(0.0)
    assert pruned_laplacian[0, 2] == pytest.approx(-0.25)
    assert pruned_laplacian[1, 2] == pytest.approx(-0.5)
    np.testing.assert_allclose(
        np.diag(pruned_laplacian),
        (1.25, 2.5, 2.75, 2.75, 2.75, 2.75, 2.75),
    )

    large = _embedding_rows(*((1.0, 0.0),) * 100)

    def return_roundoff(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tolerance = (
            np.finfo(np.float64).eps
            * len(matrix)
            * np.linalg.norm(matrix, ord=np.inf)
        )
        eigenvalues = np.zeros(len(matrix), dtype=np.float64)
        eigenvalues[0] = -0.5 * tolerance
        return eigenvalues, np.eye(len(matrix), dtype=np.float64)

    monkeypatch.setattr(speakers.np.linalg, "eigh", return_roundoff)
    accepted = speakers.cluster_anonymous_speakers(
        large,
        policy=_clustering_policy(),
    )
    assert accepted.window_cluster_ordinals == (0,) * 100

    def return_invalid_negative(
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        tolerance = (
            np.finfo(np.float64).eps
            * len(matrix)
            * np.linalg.norm(matrix, ord=np.inf)
        )
        eigenvalues = np.zeros(len(matrix), dtype=np.float64)
        eigenvalues[0] = -1.01 * tolerance
        return eigenvalues, np.eye(len(matrix), dtype=np.float64)

    monkeypatch.setattr(speakers.np.linalg, "eigh", return_invalid_negative)
    with pytest.raises(PipelineError) as caught:
        speakers.cluster_anonymous_speakers(
            large,
            policy=_clustering_policy(),
        )
    assert caught.value.code == "invalid_model_output"


def test_k_selection_falls_back_and_equal_gaps_choose_the_smallest_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    all_zero = _embedding_rows(
        *(
            tuple(1.0 if row == column else 0.0 for column in range(7))
            for row in range(7)
        )
    )
    identical = _embedding_rows(*((1.0, 0.0),) * 5)
    for embeddings in (all_zero, identical):
        result = speakers.cluster_anonymous_speakers(
            embeddings,
            policy=_clustering_policy(
                low_frequency_beta=1e-6,
                normalized_gap_gamma=1.0,
            ),
        )
        assert result.window_cluster_ordinals == (0,) * len(embeddings)

    with pytest.raises(PipelineError) as caught:
        speakers.cluster_anonymous_speakers(
            identical,
            policy=_clustering_policy(
                low_frequency_beta=np.finfo(np.float64).max,
            ),
        )
    assert caught.value.code == "invalid_model_output"

    eigenvalues = np.array((0.0, 0.2, 0.4, 0.8, 1.6), dtype=np.float64)
    eigenvectors = np.eye(5, dtype=np.float64)
    monkeypatch.setattr(
        speakers.np.linalg,
        "eigh",
        lambda _matrix: (eigenvalues.copy(), eigenvectors.copy()),
    )
    tied = speakers.cluster_anonymous_speakers(
        _common_component_rows(5),
        policy=_clustering_policy(
            low_frequency_beta=1.0,
            normalized_gap_gamma=0.5,
        ),
    )
    assert set(tied.window_cluster_ordinals) == {0, 1}
    assert len(tied.clusters) == 2


def test_raw_spectral_kmeans_is_deterministic_and_all_failed_restarts_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eigenvalues = np.array(
        (0.0, 0.1, 0.2, 1.0, 1.0, 1.0, 1.0),
        dtype=np.float64,
    )
    raw_spectral = np.array(
        (
            (-0.19109207072515932, -0.024874611258027632, 0.3702944362923237),
            (-0.3022738183889701, -0.024467784879051654, -0.282221852327925),
            (-0.4561075998667295, 0.3335431874526359, 0.15092532182623494),
            (0.3681292456307176, 0.889872696325013, -0.04938527349229478),
            (0.33974452921394566, -0.015346358382908442, 0.6664234303680957),
            (0.4739738460839315, -0.2916990314346438, 0.16549376726688494),
            (0.43414689124223027, -0.1016576214631005, -0.5352636013830171),
        ),
        dtype=np.float64,
    )
    eigenvectors = np.pad(raw_spectral, ((0, 0), (0, 4)))
    monkeypatch.setattr(
        speakers.np.linalg,
        "eigh",
        lambda _matrix: (eigenvalues.copy(), eigenvectors.copy()),
    )
    embeddings = _common_component_rows(7)
    first = speakers.cluster_anonymous_speakers(
        embeddings,
        policy=_clustering_policy(),
    )
    second = speakers.cluster_anonymous_speakers(
        embeddings,
        policy=_clustering_policy(),
    )
    assert first == second
    assert first.window_cluster_ordinals == (0, 0, 0, 1, 2, 2, 2)
    original = embeddings.astype(np.float64)
    assignments = np.asarray(first.window_cluster_ordinals)
    for ordinal, cluster in enumerate(first.clusters):
        expected = np.mean(original[assignments == ordinal], axis=0)
        expected /= np.linalg.norm(expected)
        assert len(cluster.centroid) == EMBEDDING_DIMENSION
        assert np.linalg.norm(cluster.centroid) == pytest.approx(1.0)
        np.testing.assert_allclose(
            cluster.centroid,
            expected,
            rtol=0.0,
            atol=1e-15,
        )

    self_roundoff_spectral = np.array(
        (
            (-0.8067218764337434, -0.06683176996553297, 1.8796242753275327),
            (-0.8067219586839639, -0.06683192320119531, 1.8796241390789712),
            (-0.8067218954063923, -0.06683169639496515, 1.879624300218636),
            (-0.8067219532415832, -0.06683180730167099, 1.8796242459127255),
            (-0.806722062945618, -0.06683185515330682, 1.8796241661270106),
        ),
        dtype=np.float64,
    )
    roundoff_eigenvalues = np.array(
        (0.0, 0.1, 0.2, 1.0, 1.0),
        dtype=np.float64,
    )
    roundoff_eigenvectors = np.pad(self_roundoff_spectral, ((0, 0), (0, 2)))
    monkeypatch.setattr(
        speakers.np.linalg,
        "eigh",
        lambda _matrix: (
            roundoff_eigenvalues.copy(),
            roundoff_eigenvectors.copy(),
        ),
    )
    roundoff = speakers.cluster_anonymous_speakers(
        _common_component_rows(5),
        policy=_clustering_policy(),
    )
    assert roundoff.window_cluster_ordinals == (0, 1, 0, 0, 2)

    empty_then_success = np.array(
        (
            (0.0, -3.0, 0.0),
            (2.0, 2.0, 0.0),
            (-3.0, -3.0, 0.0),
            (1.0, 3.0, 0.0),
            (3.0, 2.0, 0.0),
            (2.0, 1.0, 0.0),
            (3.0, -3.0, 0.0),
            (3.0, 2.0, 0.0),
        ),
        dtype=np.float64,
    )
    restart_eigenvalues = np.array(
        (0.0, 0.1, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0),
        dtype=np.float64,
    )
    restart_eigenvectors = np.pad(empty_then_success, ((0, 0), (0, 5)))
    monkeypatch.setattr(
        speakers.np.linalg,
        "eigh",
        lambda _matrix: (
            restart_eigenvalues.copy(),
            restart_eigenvectors.copy(),
        ),
    )
    recovered = speakers.cluster_anonymous_speakers(
        _common_component_rows(8),
        policy=_clustering_policy(),
    )
    assert recovered.window_cluster_ordinals == (0, 1, 0, 1, 1, 1, 2, 1)

    monkeypatch.setattr(
        speakers.np.linalg,
        "eigh",
        lambda _matrix: (
            restart_eigenvalues.copy(),
            np.ones((8, 8), dtype=np.float64),
        ),
    )
    with pytest.raises(PipelineError) as caught:
        speakers.cluster_anonymous_speakers(
            _common_component_rows(8),
            policy=_clustering_policy(),
        )
    assert caught.value.code == "invalid_model_output"


def test_repeated_embeddings_can_form_more_than_thirty_two_clusters() -> None:
    values = np.zeros((66, EMBEDDING_DIMENSION), dtype=np.float32)
    for ordinal in range(33):
        values[ordinal * 2 : ordinal * 2 + 2, ordinal] = 1.0
    embeddings = np.ascontiguousarray(values)

    first = speakers.cluster_anonymous_speakers(
        embeddings,
        policy=_clustering_policy(
            low_frequency_beta=2.0,
            normalized_gap_gamma=0.5,
        ),
    )
    second = speakers.cluster_anonymous_speakers(
        embeddings,
        policy=_clustering_policy(
            low_frequency_beta=2.0,
            normalized_gap_gamma=0.5,
        ),
    )

    assert first == second
    assert first.window_cluster_ordinals == tuple(
        ordinal for ordinal in range(33) for _ in range(2)
    )
    assert len(first.clusters) == 33
    assert first.clusters[-1].label == "AG"
