"""Opik tracing helpers.

Centralizes how the ragbot pipeline talks to Opik so instrumentation can be
sprinkled across the ingestion and query pipelines without every call site
having to worry about Opik being unconfigured, unreachable, or explicitly
disabled. Tracing must never be able to break the pipeline it is observing:

- `track` mirrors `opik.track` but becomes a no-op decorator when tracing is
  disabled via RAGBOT_ENABLE_OPIK=false.
- `get_langchain_tracer` builds an OpikTracer callback for LangChain
  `.invoke(..., config={"callbacks": [...]})` calls, returning None (rather
  than raising) when tracing is disabled or the tracer can't be built.
- `update_trace_metadata` safely attaches metadata/tags to whatever Opik
  trace is currently active, swallowing the OpikException Opik raises when
  there is no active trace (tracing disabled, or called outside a
  `@track`-decorated function) instead of letting it propagate.

Opik itself reads OPIK_API_KEY / OPIK_WORKSPACE / OPIK_URL_OVERRIDE directly
from the environment (see https://www.comet.com/docs/opik), so those are not
duplicated into Settings here. Until they're set, tracing calls are safe
no-ops (Opik falls back to a local, usually-unreachable default endpoint and
just logs a warning) - it will not raise or add meaningful latency to
requests.
"""

from __future__ import annotations

from typing import Any, Callable

import opik

from ragbot.config.settings import settings
from ragbot.utils.logger import get_logger

logger = get_logger("opik_tracing")

_TRACING_ENABLED = settings.enable_opik_tracing


def track(*decorator_args: Any, **decorator_kwargs: Any) -> Callable:
    """Drop-in replacement for `opik.track` that respects RAGBOT_ENABLE_OPIK.

    Usage is identical to `@opik.track(...)`. When tracing is disabled the
    wrapped function is returned completely unchanged - zero overhead, no
    dependency on Opik being installed correctly or reachable.
    """
    decorator_kwargs.setdefault("project_name", settings.opik_project_name)

    def _decorate(func: Callable) -> Callable:
        if not _TRACING_ENABLED:
            return func
        return opik.track(*decorator_args, **decorator_kwargs)(func)

    return _decorate


def get_langchain_tracer(**tracer_kwargs: Any):
    """Build an OpikTracer callback for a LangChain `.invoke()` call.

    Returns None when tracing is disabled or the tracer can't be
    constructed, so call sites can always write
    `config={"callbacks": [t]} if t else {}`. When invoked from inside a
    `@track`-decorated function, the resulting LLM span is automatically
    nested under the currently active trace.
    """
    if not _TRACING_ENABLED:
        return None
    try:
        from opik.integrations.langchain import OpikTracer

        tracer_kwargs.setdefault("project_name", settings.opik_project_name)
        return OpikTracer(**tracer_kwargs)
    except Exception:  # pragma: no cover - tracing must never break the pipeline
        logger.warning("Failed to initialize Opik LangChain tracer", exc_info=True)
        return None


def update_trace_metadata(**kwargs: Any) -> None:
    """Safely attach metadata/tags/etc. to the current Opik trace, if any.

    No-ops when tracing is disabled, and swallows the OpikException Opik
    raises when there is no active trace in context. Annotating a trace must
    never be able to break the pipeline it is observing.
    """
    if not _TRACING_ENABLED:
        return
    try:
        from opik import opik_context

        opik_context.update_current_trace(**kwargs)
    except Exception:  # pragma: no cover - tracing must never break the pipeline
        logger.debug("Failed to update Opik trace metadata", exc_info=True)
