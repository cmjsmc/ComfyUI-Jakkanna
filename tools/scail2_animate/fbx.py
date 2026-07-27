#!/usr/bin/env python3
"""Minimal binary-FBX reader for skeletal animation clips.

Covers exactly what a Mixamo-style humanoid clip needs: the node records, the
Model hierarchy with its transform properties, and the AnimationCurve data,
evaluated to per-bone local matrices at arbitrary times.

Conventions follow the FBX SDK (and three.js's FBXLoader, which is what the
existing Pose Studio importer uses):

    Local = T . Roff . Rp . Rpre . R . Rpost^-1 . Rp^-1 . Soff . Sp . S . Sp^-1

and rotation order 0 (eEulerXYZ) means extrinsic XYZ, i.e. R = Rz @ Ry @ Rx,
which is scipy's lowercase "xyz".
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

FBX_TIME_UNIT = 46186158000  # FBX ticks per second

# FBX RotationOrder enum -> scipy extrinsic euler sequence.
EULER_ORDERS = {
    0: "xyz", 1: "xzy", 2: "yzx", 3: "yxz", 4: "zxy", 5: "zyx", 6: "xyz",
}

_ARRAY_TYPES = {"f": ("f", 4), "d": ("d", 8), "l": ("q", 8), "i": ("i", 4), "b": ("b", 1)}
_SCALAR_TYPES = {"Y": ("h", 2), "C": ("?", 1), "I": ("i", 4), "F": ("f", 4), "D": ("d", 8), "L": ("q", 8)}


@dataclass
class FbxNode:
    name: str
    props: list = field(default_factory=list)
    children: list["FbxNode"] = field(default_factory=list)

    def find(self, name: str) -> "FbxNode | None":
        for child in self.children:
            if child.name == name:
                return child
        return None

    def find_all(self, name: str):
        return [child for child in self.children if child.name == name]


class _Reader:
    def __init__(self, data: bytes, version: int):
        self.data = data
        self.pos = 0
        self.wide = version >= 7500

    def _unpack(self, fmt: str, size: int):
        value = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return value

    def read_node(self) -> FbxNode | None:
        if self.wide:
            end_offset, num_props, prop_len = self._unpack("<QQQ", 24)
        else:
            end_offset, num_props, prop_len = self._unpack("<III", 12)
        (name_len,) = self._unpack("<B", 1)
        if end_offset == 0:
            return None
        name = self.data[self.pos:self.pos + name_len].decode("utf-8", "replace")
        self.pos += name_len

        props = [self.read_property() for _ in range(num_props)]
        node = FbxNode(name=name, props=props)

        while self.pos < end_offset:
            child = self.read_node()
            if child is None:
                break
            node.children.append(child)
        self.pos = end_offset
        return node

    def read_property(self):
        type_code = chr(self.data[self.pos])
        self.pos += 1

        if type_code in _SCALAR_TYPES:
            fmt, size = _SCALAR_TYPES[type_code]
            return self._unpack("<" + fmt, size)[0]

        if type_code in _ARRAY_TYPES:
            fmt, item_size = _ARRAY_TYPES[type_code]
            length, encoding, comp_len = self._unpack("<III", 12)
            raw = self.data[self.pos:self.pos + comp_len]
            self.pos += comp_len
            if encoding == 1:
                raw = zlib.decompress(raw)
            return np.frombuffer(raw[:length * item_size], dtype=np.dtype("<" + fmt))

        if type_code in ("S", "R"):
            (length,) = self._unpack("<I", 4)
            raw = self.data[self.pos:self.pos + length]
            self.pos += length
            return raw.decode("utf-8", "replace") if type_code == "S" else raw

        raise ValueError(f"unsupported FBX property type {type_code!r} at {self.pos}")


def parse(path: Path) -> tuple[FbxNode, int]:
    """Parse a binary FBX file into a node tree."""
    data = Path(path).read_bytes()
    if not data.startswith(b"Kaydara FBX Binary"):
        raise ValueError(f"{path} is not a binary FBX file")
    version = struct.unpack_from("<I", data, 23)[0]

    reader = _Reader(data, version)
    reader.pos = 27
    root = FbxNode(name="__root__")
    while reader.pos < len(data) - 160:
        node = reader.read_node()
        if node is None:
            break
        root.children.append(node)
    return root, version


def properties70(node: FbxNode) -> dict:
    """Read a Properties70 block into {name: value}."""
    out = {}
    block = node.find("Properties70")
    if block is None:
        return out
    for prop in block.find_all("P"):
        name = prop.props[0]
        values = prop.props[4:]
        if len(values) == 1:
            out[name] = values[0]
        elif values:
            out[name] = list(values)
    return out


@dataclass
class Bone:
    uid: int
    name: str
    parent: int | None
    translation: np.ndarray
    rotation: np.ndarray          # degrees
    scaling: np.ndarray
    pre_rotation: np.ndarray      # degrees
    post_rotation: np.ndarray     # degrees
    rotation_offset: np.ndarray
    rotation_pivot: np.ndarray
    scaling_offset: np.ndarray
    scaling_pivot: np.ndarray
    order: str


def _vec(props: dict, key: str, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    value = props.get(key, default)
    if not isinstance(value, (list, tuple, np.ndarray)):
        value = default
    return np.asarray(value, dtype=np.float64)[:3]


def _euler_matrix(degrees: np.ndarray, order: str) -> np.ndarray:
    if not np.any(degrees):
        return np.eye(3)
    return Rotation.from_euler(order, np.asarray(degrees, dtype=np.float64), degrees=True).as_matrix()


def _translation(vec: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = vec
    return matrix


def _rotation4(matrix3: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = matrix3
    return matrix


class FbxClip:
    """A parsed FBX skeleton plus its first animation clip."""

    def __init__(self, path: Path):
        self.root, self.version = parse(path)
        objects = self.root.find("Objects")
        connections = self.root.find("Connections")
        if objects is None or connections is None:
            raise ValueError("FBX file has no Objects/Connections sections")

        self.objects = {}
        for node in objects.children:
            if not node.props:
                continue
            uid = node.props[0]
            raw_name = node.props[1] if len(node.props) > 1 else ""
            name = raw_name.split("\x00\x01")[0] if isinstance(raw_name, str) else ""
            self.objects[uid] = (node.name, name, node)

        # child -> [(parent, property_name)]
        self.parents: dict[int, list[tuple[int, str | None]]] = {}
        for conn in connections.find_all("C"):
            kind = conn.props[0]
            child, parent = conn.props[1], conn.props[2]
            prop = conn.props[3] if kind == "OP" and len(conn.props) > 3 else None
            self.parents.setdefault(child, []).append((parent, prop))

        self.bones = self._build_bones()
        self.curves = self._build_curves()
        self.duration = self._clip_duration()

    # -- skeleton ----------------------------------------------------------
    def _build_bones(self) -> dict[int, Bone]:
        bones: dict[int, Bone] = {}
        for uid, (kind, name, node) in self.objects.items():
            if kind != "Model":
                continue
            props = properties70(node)
            order = EULER_ORDERS.get(int(props.get("RotationOrder", 0) or 0), "xyz")
            parent = None
            for parent_uid, prop in self.parents.get(uid, []):
                if prop is None and parent_uid in self.objects and self.objects[parent_uid][0] == "Model":
                    parent = parent_uid
                    break
            bones[uid] = Bone(
                uid=uid, name=name, parent=parent,
                translation=_vec(props, "Lcl Translation"),
                rotation=_vec(props, "Lcl Rotation"),
                scaling=_vec(props, "Lcl Scaling", (1.0, 1.0, 1.0)),
                pre_rotation=_vec(props, "PreRotation"),
                post_rotation=_vec(props, "PostRotation"),
                rotation_offset=_vec(props, "RotationOffset"),
                rotation_pivot=_vec(props, "RotationPivot"),
                scaling_offset=_vec(props, "ScalingOffset"),
                scaling_pivot=_vec(props, "ScalingPivot"),
                order=order,
            )
        return bones

    # -- animation ---------------------------------------------------------
    def _build_curves(self) -> dict[int, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]]:
        """{bone_uid: {"Lcl Rotation": {"X": (times, values), ...}, ...}}"""
        curves: dict[int, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]] = {}
        for uid, (kind, _name, node) in self.objects.items():
            if kind != "AnimationCurve":
                continue
            times = node.find("KeyTime")
            values = node.find("KeyValueFloat")
            if times is None or values is None:
                continue
            key_times = np.asarray(times.props[0], dtype=np.float64) / FBX_TIME_UNIT
            key_values = np.asarray(values.props[0], dtype=np.float64)

            for curve_node_uid, channel in self.parents.get(uid, []):
                if channel is None or curve_node_uid not in self.objects:
                    continue
                axis = channel.split("|")[-1].upper()
                if axis not in ("X", "Y", "Z"):
                    continue
                for bone_uid, target in self.parents.get(curve_node_uid, []):
                    if target is None or bone_uid not in self.bones:
                        continue
                    curves.setdefault(bone_uid, {}).setdefault(target, {})[axis] = (key_times, key_values)
        return curves

    def _clip_duration(self) -> float:
        longest = 0.0
        for channels in self.curves.values():
            for axes in channels.values():
                for times, _ in axes.values():
                    if len(times):
                        longest = max(longest, float(times[-1]))
        return longest

    def sample_channel(self, uid: int, channel: str, time: float, default: np.ndarray) -> np.ndarray:
        axes = self.curves.get(uid, {}).get(channel)
        if not axes:
            return default.copy()
        out = default.astype(np.float64).copy()
        for index, axis in enumerate("XYZ"):
            entry = axes.get(axis)
            if entry is None:
                continue
            times, values = entry
            if len(times) == 0:
                continue
            out[index] = float(np.interp(time, times, values))
        return out

    def local_matrix(self, bone: Bone, time: float) -> np.ndarray:
        """Full FBX local transform of one bone at `time`."""
        translation = self.sample_channel(bone.uid, "Lcl Translation", time, bone.translation)
        rotation = self.sample_channel(bone.uid, "Lcl Rotation", time, bone.rotation)
        scaling = self.sample_channel(bone.uid, "Lcl Scaling", time, bone.scaling)

        r_pre = _rotation4(_euler_matrix(bone.pre_rotation, bone.order))
        r_main = _rotation4(_euler_matrix(rotation, bone.order))
        r_post = _rotation4(_euler_matrix(bone.post_rotation, bone.order))
        scale = np.diag([scaling[0], scaling[1], scaling[2], 1.0])

        return (
            _translation(translation)
            @ _translation(bone.rotation_offset)
            @ _translation(bone.rotation_pivot)
            @ r_pre @ r_main @ np.linalg.inv(r_post)
            @ _translation(-bone.rotation_pivot)
            @ _translation(bone.scaling_offset)
            @ _translation(bone.scaling_pivot)
            @ scale
            @ _translation(-bone.scaling_pivot)
        )

    def world_matrices(self, time: float) -> dict[int, np.ndarray]:
        """World matrix of every bone at `time`, parents resolved first."""
        cache: dict[int, np.ndarray] = {}

        def resolve(uid: int) -> np.ndarray:
            if uid in cache:
                return cache[uid]
            bone = self.bones[uid]
            local = self.local_matrix(bone, time)
            matrix = local if bone.parent is None else resolve(bone.parent) @ local
            cache[uid] = matrix
            return matrix

        for uid in self.bones:
            resolve(uid)
        return cache

    def animated_bones(self) -> set[int]:
        return {uid for uid in self.curves if uid in self.bones}
