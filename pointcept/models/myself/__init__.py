"""Opt-in registration of the SegmentAnyTreeV2 myself implementation."""

from importlib import import_module

SegmentAnyTreeV2 = import_module(".sat-v1m1", __name__).SegmentAnyTreeV2

__all__ = ["SegmentAnyTreeV2"]
