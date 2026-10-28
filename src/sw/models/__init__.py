"""Models. ``MiniNetSeg`` is the CNN control. ``BandTransformer`` is the scaffold."""

_CNN = {"MiniNetSeg", "count_parameters"}
_TRANSFORMER = {
    "BandTransformer",
    "SpectralBlock",
    "SpectralPool",
    "SpatialPE",
    "SpatialBlock",
    "SegHead",
}


def __getattr__(name):
    if name in _CNN:
        from sw.models import cnn

        return getattr(cnn, name)
    if name in _TRANSFORMER:
        from sw.models import transformer

        return getattr(transformer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
