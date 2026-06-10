from typing import Any

__all__ = ["MultiAgentRagWorkflow", "build_default_workflow"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from orchestrator.workflow import MultiAgentRagWorkflow, build_default_workflow

        exports = {
            "MultiAgentRagWorkflow": MultiAgentRagWorkflow,
            "build_default_workflow": build_default_workflow,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
