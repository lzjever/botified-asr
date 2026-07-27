from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from botified_asr.api import Readiness, create_app
from botified_asr.audio import FfmpegAudioFrontend
from botified_asr.config import load_api_key, load_config
from botified_asr.pipeline import PipelineNotReady, Processor
from botified_asr.storage import Storage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=_default_config_path())
    args = parser.parse_args()
    config = load_config(args.config)
    api_key = load_api_key()
    storage = Storage(config.storage.data_dir, config.limits)
    host, port_text = config.server.listen.rsplit(":", 1)
    processor = Processor(
        FfmpegAudioFrontend(),
        _ModelsNotLoadedAdapter(),
    )
    app = create_app(
        api_key=api_key,
        readiness=Readiness(database=True, models=False, executor=False),
        storage=storage,
        processor=processor,
    )
    uvicorn.run(app, host=host, port=int(port_text), workers=1)


def _default_config_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_home) if config_home else Path("~/.config").expanduser()
    return root / "botified-asr" / "config.yaml"


class _ModelsNotLoadedAdapter:
    def transcribe(self, _pcm):
        raise PipelineNotReady()
