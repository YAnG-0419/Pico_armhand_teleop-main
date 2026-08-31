"""Wuji hand adapter public API with lazy optional dependencies."""

__all__ = ["WujiHandPipeline"]


def __getattr__(name):
    if name == "WujiHandPipeline":
        from .pipeline import WujiHandPipeline

        return WujiHandPipeline
    raise AttributeError(name)
