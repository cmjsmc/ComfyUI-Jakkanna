"""Build UI and API graphs from one declarative ComfyUI node mapping."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib
import urllib.request


JAKKANNA_ROOT = Path(__file__).resolve().parents[2]
PRIMITIVES = {"INT", "FLOAT", "STRING", "BOOLEAN"}


def jakkanna_version() -> str:
    with (JAKKANNA_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def object_info(server: str) -> dict:
    with urllib.request.urlopen(f"{server.rstrip('/')}/object_info") as response:
        return json.load(response)


def node_spec(info: dict, node_type: str) -> tuple[list[dict], list[dict]]:
    node_info = info[node_type]
    inputs = []
    for section in ("required", "optional"):
        for name, data in node_info["input"].get(section, {}).items():
            input_type = data[0]
            config = data[1] if len(data) > 1 else {}
            if isinstance(input_type, list) or input_type == "COMBO":
                options = input_type if isinstance(input_type, list) else config.get("options", [])
                inputs.append({
                    "name": name,
                    "type": "COMBO",
                    "widget": True,
                    "default": config.get("default", options[0] if options else ""),
                    "seed": False,
                })
            elif input_type in PRIMITIVES and not config.get("forceInput"):
                inputs.append({
                    "name": name,
                    "type": input_type,
                    "widget": True,
                    "default": config.get("default", 0),
                    "seed": bool(config.get("control_after_generate")),
                })
            else:
                inputs.append({
                    "name": name,
                    "type": input_type if isinstance(input_type, str) else "COMBO",
                    "widget": False,
                    "default": None,
                    "seed": False,
                })
    outputs = [
        {"name": name, "type": output_type}
        for output_type, name in zip(node_info["output"], node_info["output_name"])
    ]
    return inputs, outputs


def build(
    server: str,
    workflow: dict,
    note: str,
    titles: dict,
    columns: dict,
    graph_id: str,
) -> dict:
    info = object_info(server)
    specs = {
        node_type: node_spec(info, node_type)
        for node_type, _values in workflow.values()
    }
    id_map = {name: index + 1 for index, name in enumerate(workflow)}
    rows = {}
    nodes_out = []
    links = []

    for order, (name, (node_type, values)) in enumerate(workflow.items()):
        column = columns[name]
        row = rows.get(column, 0)
        rows[column] = row + 1
        properties = {"Node name for S&R": node_type}
        if node_type.startswith("Jakkanna") or node_type.startswith("VNCCS"):
            properties.update({
                "cnr_id": "jakkanna",
                "ver": jakkanna_version(),
                "aux_id": "teenu/ComfyUI-Jakkanna",
            })
        node = {
            "id": id_map[name],
            "type": node_type,
            "pos": [column * 380 + 20, row * 190 + 40],
            "size": [340, 150],
            "flags": {},
            "order": order,
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "properties": properties,
            "widgets_values": [],
        }
        if name in titles:
            node["title"] = titles[name]

        for spec in specs[node_type][0]:
            value = values.get(spec["name"], "__unset__")
            connected = isinstance(value, list) and len(value) == 2 and value[0] in id_map
            if spec["widget"]:
                node["widgets_values"].append(
                    spec["default"] if connected or value == "__unset__" else value
                )
                if spec["seed"]:
                    node["widgets_values"].append("fixed")
                if connected:
                    node["inputs"].append({
                        "name": spec["name"],
                        "type": spec["type"],
                        "widget": {"name": spec["name"]},
                        "link": None,
                    })
            else:
                node["inputs"].append({
                    "name": spec["name"],
                    "type": spec["type"],
                    "link": None,
                })

        if node_type in {"LoadVideo", "LoadImage", "SCAIL2ReferenceImageLoader"}:
            node["widgets_values"].append("image")
        for output_index, output in enumerate(specs[node_type][1]):
            node["outputs"].append({
                "name": output["name"],
                "type": output["type"],
                "links": [],
                "slot_index": output_index,
            })
        if node_type == "VHS_VideoCombine":
            node["widgets_values"] = {
                key: value
                for key, value in values.items()
                if not (isinstance(value, list) and len(value) == 2 and value[0] in id_map)
            }
            node["widgets_values"]["videopreview"] = {"hidden": False, "paused": False}
        nodes_out.append(node)

    node_map = {node["id"]: node for node in nodes_out}
    link_id = 0
    for name, (_node_type, values) in workflow.items():
        destination = node_map[id_map[name]]
        for input_name, value in values.items():
            if not (isinstance(value, list) and len(value) == 2 and value[0] in id_map):
                continue
            source_id, source_slot = id_map[value[0]], value[1]
            destination_slot = next(
                index
                for index, item in enumerate(destination["inputs"])
                if item["name"] == input_name
            )
            link_id += 1
            destination["inputs"][destination_slot]["link"] = link_id
            node_map[source_id]["outputs"][source_slot]["links"].append(link_id)
            links.append([
                link_id,
                source_id,
                source_slot,
                destination["id"],
                destination_slot,
                destination["inputs"][destination_slot]["type"],
            ])

    nodes_out.append({
        "id": 900,
        "type": "MarkdownNote",
        "pos": [20, -520],
        "size": [1080, 480],
        "flags": {},
        "order": len(nodes_out),
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "properties": {"Node name for S&R": "MarkdownNote"},
        "widgets_values": [note],
    })
    return {
        "id": graph_id,
        "revision": 0,
        "last_node_id": 900,
        "last_link_id": link_id,
        "nodes": nodes_out,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def to_api(server: str, workflow: dict) -> dict:
    info = object_info(server)
    id_map = {name: str(index + 1) for index, name in enumerate(workflow)}
    prompt = {}
    for name, (node_type, values) in workflow.items():
        inputs = {
            spec["name"]: spec["default"]
            for spec in node_spec(info, node_type)[0]
            if spec["widget"]
        }
        for key, value in values.items():
            if isinstance(value, list) and len(value) == 2 and value[0] in id_map:
                inputs[key] = [id_map[value[0]], value[1]]
            else:
                inputs[key] = value
        prompt[id_map[name]] = {"class_type": node_type, "inputs": inputs}
    return prompt
