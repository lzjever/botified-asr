from __future__ import annotations


class PipelineError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PipelineNotReady(PipelineError):
    def __init__(self) -> None:
        super().__init__(
            "pipeline_not_ready",
            "The requested audio pipeline is not ready",
        )
