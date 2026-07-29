from pathlib import Path

from botified_asr.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTAINER_CONFIG = PROJECT_ROOT / "config" / "container.yaml"


def test_container_config_has_loadable_sibling_data_roots() -> None:
    config = load_config(CONTAINER_CONFIG)

    assert config.server.listen == "0.0.0.0:17770"
    assert config.runtime.device == "cpu"
    assert config.runtime.inference_lanes == 1
    assert config.storage.data_dir == Path("/data/state")
    assert config.runtime.model_cache_dir == Path("/data/models")
    assert config.storage.data_dir.parent == Path("/data")
    assert config.runtime.model_cache_dir.parent == Path("/data")
