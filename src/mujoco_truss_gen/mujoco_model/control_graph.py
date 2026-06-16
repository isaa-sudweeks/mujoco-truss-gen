from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Literal

import mujoco

CONTROL_GRAPH_TEXT_NAME = "mujoco_truss_gen_control_graph"
ControlEdgeType = Literal["actuated", "connector"]


@dataclass(frozen=True, slots=True)
class ControlGraphEdge:
    from_node: str
    to_node: str
    type: ControlEdgeType


@dataclass(frozen=True, slots=True)
class ControlGraphActuatorEdge:
    from_node: str
    to_node: str
    tendon: str


@dataclass(frozen=True, slots=True)
class ControlGraphMetadata:
    control_node_names: list[str]
    control_node_to_physical_node: dict[str, str]
    control_node_to_logical_node: dict[str, str]
    edges: list[ControlGraphEdge]
    actuator_edges: list[ControlGraphActuatorEdge]
    passive_control_node_names: list[str]

    @classmethod
    def empty(cls) -> ControlGraphMetadata:
        return cls(
            control_node_names=[],
            control_node_to_physical_node={},
            control_node_to_logical_node={},
            edges=[],
            actuator_edges=[],
            passive_control_node_names=[],
        )

    @property
    def enabled(self) -> bool:
        return bool(self.control_node_names)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> ControlGraphMetadata:
        payload = json.loads(raw)
        return cls(
            control_node_names=list(payload.get("control_node_names", [])),
            control_node_to_physical_node=dict(
                payload.get("control_node_to_physical_node", {})
            ),
            control_node_to_logical_node=dict(payload.get("control_node_to_logical_node", {})),
            edges=[
                ControlGraphEdge(
                    from_node=str(edge["from_node"]),
                    to_node=str(edge["to_node"]),
                    type=_coerce_edge_type(edge["type"]),
                )
                for edge in payload.get("edges", [])
            ],
            actuator_edges=[
                ControlGraphActuatorEdge(
                    from_node=str(edge["from_node"]),
                    to_node=str(edge["to_node"]),
                    tendon=str(edge["tendon"]),
                )
                for edge in payload.get("actuator_edges", [])
            ],
            passive_control_node_names=list(payload.get("passive_control_node_names", [])),
        )


def add_control_graph_metadata(
    spec: mujoco.MjSpec,
    metadata: ControlGraphMetadata,
) -> None:
    spec.add_text(name=CONTROL_GRAPH_TEXT_NAME, data=metadata.to_json())


def control_graph_metadata_from_xml(xml: str | None) -> ControlGraphMetadata:
    if not xml:
        return ControlGraphMetadata.empty()

    root = ET.fromstring(xml)
    text = root.find(f"./custom/text[@name='{CONTROL_GRAPH_TEXT_NAME}']")
    if text is None:
        return ControlGraphMetadata.empty()

    data = text.get("data")
    if not data:
        return ControlGraphMetadata.empty()

    return ControlGraphMetadata.from_json(data)


def connector_edges_for_logical_nodes(
    control_node_names: list[str],
    control_node_to_logical_node: dict[str, str],
) -> list[ControlGraphEdge]:
    by_logical_node: dict[str, list[str]] = {}
    for control_node_name in control_node_names:
        logical_node = control_node_to_logical_node[control_node_name]
        by_logical_node.setdefault(logical_node, []).append(control_node_name)

    connector_edges = []
    for instances in by_logical_node.values():
        if len(instances) <= 1:
            continue
        for index, from_node in enumerate(instances):
            for to_node in instances[index + 1 :]:
                connector_edges.append(
                    ControlGraphEdge(
                        from_node=from_node,
                        to_node=to_node,
                        type="connector",
                    )
                )
    return connector_edges


def unique_control_edges(edges: list[ControlGraphEdge]) -> list[ControlGraphEdge]:
    unique_edges = []
    seen: set[tuple[str, str, str]] = set()
    for edge in edges:
        key = (*sorted((edge.from_node, edge.to_node)), edge.type)
        if key in seen:
            continue
        seen.add(key)
        unique_edges.append(edge)
    return unique_edges


def _coerce_edge_type(raw: str) -> ControlEdgeType:
    if raw not in {"actuated", "connector"}:
        raise ValueError(f"Unsupported control graph edge type: {raw!r}")
    return raw
