"""Models. The CNN control is ``MiniNetSeg``. The transformer comes later."""


def __getattr__(name):
    if name in {"MiniNetSeg", "count_parameters"}:
        from sw.models import cnn

        return getattr(cnn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
