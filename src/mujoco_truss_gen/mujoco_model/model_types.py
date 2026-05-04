from __future__ import annotations

from typing import Any

Vector = list[float]
NodeDict = dict[str, Vector]
TriangleDict = dict[str, list[str]]
ShapeDefinition = dict[str, Any]
ShapeDict = dict[str, ShapeDefinition]
EdgeKey = tuple[str, str]
EdgeTendonMap = dict[EdgeKey, str]
