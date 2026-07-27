#!/usr/bin/env python3
"""Bake an 81-view SCAIL-2 turntable onto a posed Jakkanna GLB."""

import argparse
import json
import math
from pathlib import Path
import struct

import cv2
import numpy as np
from PIL import Image
import trimesh


FRAME_COUNT = 81
VIEW_WEIGHT_POWER = 8


def _rotation_matrix(degrees):
    x, y, z = np.radians(degrees)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def _camera(metadata):
    capture = metadata["capture"]
    target = np.asarray(metadata["mesh_center"], dtype=np.float32)
    target += np.array(
        [-capture["offset_x"], -capture["offset_y"], 0],
        dtype=np.float32,
    )

    yaw = math.radians(capture["yaw_degrees"])
    pitch = math.radians(capture["pitch_degrees"])
    distance = capture["distance"]
    offset = np.array(
        [
            distance * math.cos(pitch) * math.sin(yaw),
            -distance * math.sin(pitch),
            distance * math.cos(pitch) * math.cos(yaw),
        ],
        dtype=np.float32,
    )
    position = target + offset
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0, 1, 0], dtype=np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return position, right, up, forward


def _project(points, camera, width, height, fov_degrees, zoom):
    position, right, up, forward = camera
    relative = points - position
    depth = relative @ forward
    tan_half_fov = math.tan(math.radians(fov_degrees) * 0.5) / zoom
    aspect = width / height
    x = (relative @ right) / (depth * tan_half_fov * aspect)
    y = (relative @ up) / (depth * tan_half_fov)
    return np.column_stack(((x + 1) * width * 0.5, (1 - y) * height * 0.5)), depth


def _bbox(points):
    return np.array([points.min(axis=0), points.max(axis=0)], dtype=np.float32)


def _map_bbox(points, source_bbox, destination_bbox):
    source_size = np.maximum(source_bbox[1] - source_bbox[0], 1)
    destination_size = np.maximum(destination_bbox[1] - destination_bbox[0], 1)
    return (
        (points - source_bbox[0]) * destination_size / source_size
        + destination_bbox[0]
    )


def _mask_bbox(mask):
    y, x = np.where(mask)
    if not len(x):
        raise ValueError("SCAIL-2 mask is empty")
    return np.array([[x.min(), y.min()], [x.max(), y.max()]], dtype=np.float32)


def _rasterize_uv(mesh, size):
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    uv = np.asarray(mesh.visual.uv, dtype=np.float32)
    uv_pixels = np.column_stack((uv[:, 0], 1 - uv[:, 1])) * (size - 1)

    positions = np.zeros((size, size, 3), dtype=np.float32)
    surface_normals = np.zeros_like(positions)
    valid = np.zeros((size, size), dtype=bool)

    for face in faces:
        triangle = uv_pixels[face]
        lower = np.maximum(np.floor(triangle.min(axis=0)).astype(int), 0)
        upper = np.minimum(np.ceil(triangle.max(axis=0)).astype(int), size - 1)
        if np.any(upper < lower):
            continue

        x = np.arange(lower[0], upper[0] + 1, dtype=np.float32)
        y = np.arange(lower[1], upper[1] + 1, dtype=np.float32)
        px, py = np.meshgrid(x, y)
        a, b, c = triangle
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-8:
            continue
        w0 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / denominator
        w1 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / denominator
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue

        rows = py[inside].astype(int)
        columns = px[inside].astype(int)
        weights = np.column_stack((w0[inside], w1[inside], w2[inside]))
        positions[rows, columns] = weights @ vertices[face]
        surface_normals[rows, columns] = weights @ normals[face]
        valid[rows, columns] = True

    lengths = np.linalg.norm(surface_normals[valid], axis=1, keepdims=True)
    surface_normals[valid] /= np.maximum(lengths, 1e-8)
    return positions, surface_normals, valid


def _turntable_rotations(base_rotation):
    rotations = []
    for index in range(FRAME_COUNT):
        degrees = list(base_rotation)
        degrees[1] += index * 360.0 / FRAME_COUNT
        rotations.append(_rotation_matrix(degrees))
    return rotations


def _depth_maps(mesh, rotations, camera, width, height, fov_degrees, zoom, scale=4):
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    map_width = math.ceil(width / scale)
    map_height = math.ceil(height / scale)
    depth_maps = []
    for rotation in rotations:
        projected, depth = _project(vertices @ rotation.T, camera, width, height, fov_degrees, zoom)
        triangles = np.rint(projected[faces] / scale).astype(np.int32)
        face_depth = depth[faces].mean(axis=1)
        depth_map = np.full((map_height, map_width), np.inf, dtype=np.float32)
        for face_index in np.argsort(face_depth)[::-1]:
            cv2.fillConvexPoly(depth_map, triangles[face_index], float(face_depth[face_index]))
        depth_maps.append(depth_map)
    return depth_maps, scale


def _visible(projected, depth, depth_map, scale, tolerance=0.02):
    x = np.rint(projected[:, 0] / scale).astype(np.int32)
    y = np.rint(projected[:, 1] / scale).astype(np.int32)
    inside = (x >= 0) & (x < depth_map.shape[1]) & (y >= 0) & (y < depth_map.shape[0]) & (depth > 0)
    visible = np.zeros(len(projected), dtype=bool)
    visible[inside] = depth[inside] <= depth_map[y[inside], x[inside]] + tolerance
    return visible


def _maximum_visible_scores(positions, normals, rotations, camera, width, height, fov_degrees, zoom, depth_maps):
    maps, scale = depth_maps
    best_score = np.full(len(positions), -np.inf, dtype=np.float32)
    for index, rotation in enumerate(rotations):
        rotated_positions = positions @ rotation.T
        rotated_normals = normals @ rotation.T
        view = camera[0] - rotated_positions
        view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-8)
        score = np.sum(rotated_normals * view, axis=1)
        projected, depth = _project(rotated_positions, camera, width, height, fov_degrees, zoom)
        update = (score > best_score) & _visible(projected, depth, maps[index], scale)
        best_score[update] = score[update]
    return best_score


def _load_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read {path}")
    return mask > 127


def _remap_points(image, points, interpolation):
    columns = min(len(points), 16384)
    rows = math.ceil(len(points) / columns)
    padded = rows * columns
    map_x = np.zeros(padded, dtype=np.float32)
    map_y = np.zeros(padded, dtype=np.float32)
    map_x[:len(points)] = points[:, 0]
    map_y[:len(points)] = points[:, 1]
    result = cv2.remap(
        image,
        map_x.reshape(rows, columns),
        map_y.reshape(rows, columns),
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return result.reshape(-1, *image.shape[2:])[:len(points)]


def _flow_map(points, source_bbox, depth_map, depth_scale, destination_mask, optical_flow):
    destination_bbox = _mask_bbox(destination_mask)
    mapped = _map_bbox(points, source_bbox, destination_bbox)

    height, width = destination_mask.shape
    flow_width = max(1, width // 2)
    flow_height = max(1, height // 2)
    output_scale = np.array([flow_width / width, flow_height / height], dtype=np.float32)
    source_size = np.maximum(source_bbox[1] - source_bbox[0], 1)
    destination_size = np.maximum(destination_bbox[1] - destination_bbox[0], 1)
    affine_scale = depth_scale * destination_size / source_size * output_scale
    affine_offset = (
        destination_bbox[0] - source_bbox[0] * destination_size / source_size
    ) * output_scale
    transform = np.array(
        [
            [affine_scale[0], 0, affine_offset[0]],
            [0, affine_scale[1], affine_offset[1]],
        ],
        dtype=np.float32,
    )
    source_mask = np.isfinite(depth_map).astype(np.uint8) * 255
    warped_source = cv2.warpAffine(
        source_mask,
        transform,
        (flow_width, flow_height),
        flags=cv2.INTER_NEAREST,
    )
    target = cv2.resize(
        destination_mask.astype(np.uint8) * 255,
        (flow_width, flow_height),
        interpolation=cv2.INTER_NEAREST,
    )
    flow = optical_flow.calc(warped_source, target, None)
    correction = _remap_points(
        flow,
        mapped * output_scale,
        cv2.INTER_LINEAR,
    )
    return mapped + correction / output_scale


def _pad_texture(texture, valid, iterations=16):
    padded = texture.copy()
    filled = valid.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(iterations):
        expanded = cv2.dilate(filled.astype(np.uint8), kernel).astype(bool)
        edge = expanded & ~filled
        if not edge.any():
            break
        for channel in range(3):
            dilated = cv2.dilate(padded[:, :, channel], kernel)
            padded[:, :, channel][edge] = dilated[edge]
        filled = expanded
    return padded


def _validate_manifest(dataset):
    path = dataset / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Failed to read dataset manifest: {error}") from error
    frames = manifest.get("frames")
    if manifest.get("frame_count") != FRAME_COUNT or not isinstance(frames, list) or len(frames) != FRAME_COUNT:
        raise ValueError("Dataset manifest must describe exactly 81 frames")
    if manifest.get("duplicate_endpoint") is not False:
        raise ValueError("Dataset manifest must omit the duplicate 360-degree endpoint")
    for index, frame in enumerate(frames):
        expected_angle = index * 360.0 / FRAME_COUNT
        if (
            frame.get("index") != index
            or frame.get("filename") != f"frame_{index:04d}.png"
            or not math.isclose(frame.get("turntable_angle_degrees", math.inf), expected_angle, abs_tol=1e-6)
        ):
            raise ValueError(f"Dataset manifest frame {index} is not an exact 81-view turntable")


def _export_unlit_glb(mesh, path):
    blob = trimesh.exchange.gltf.export_glb(mesh)
    magic, version, _ = struct.unpack_from("<4sII", blob)
    if magic != b"glTF" or version != 2:
        raise ValueError("trimesh did not produce a GLB 2.0 asset")

    chunks = []
    offset = 12
    while offset < len(blob):
        length, kind = struct.unpack_from("<II", blob, offset)
        offset += 8
        chunks.append((kind, blob[offset:offset + length]))
        offset += length
    if not chunks or chunks[0][0] != 0x4E4F534A:
        raise ValueError("GLB has no JSON header")

    header = json.loads(chunks[0][1].rstrip(b" \x00").decode("utf-8"))
    used = header.setdefault("extensionsUsed", [])
    if "KHR_materials_unlit" not in used:
        used.append("KHR_materials_unlit")
    for material in header.get("materials", []):
        material.setdefault("extensions", {})["KHR_materials_unlit"] = {}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 4)
    chunks[0] = (chunks[0][0], encoded)

    body = b"".join(struct.pack("<II", len(chunk), kind) + chunk for kind, chunk in chunks)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


def bake(mesh, metadata, dataset, texture_size):
    if metadata.get("capture", {}).get("sam_projection"):
        raise ValueError("SAM projection camera metadata is not supported")
    _validate_manifest(dataset)

    geometry = dataset / "layers" / ".geometry"
    masks = dataset / "layers" / ".mask"
    frames = [geometry / f"frame_{index:04d}.png" for index in range(FRAME_COUNT)]
    mask_paths = [masks / f"frame_{index:04d}.png" for index in range(FRAME_COUNT)]
    if not all(path.is_file() for path in frames + mask_paths):
        raise ValueError("Dataset must contain 81 matched geometry images and masks")

    first = _load_rgb(frames[0])
    height, width = first.shape[:2]
    capture = metadata["capture"]
    if (width, height) != (capture["width"], capture["height"]):
        raise ValueError("Dataset images do not match the Pose Studio capture dimensions")
    camera = _camera(metadata)
    rotations = _turntable_rotations(metadata["model_rotation_degrees"])
    depth_maps = _depth_maps(
        mesh,
        rotations,
        camera,
        width,
        height,
        capture["fov_degrees"],
        capture["zoom"],
    )

    position_map, normal_map, valid = _rasterize_uv(mesh, texture_size)
    rows, columns = np.where(valid)
    positions = position_map[rows, columns]
    normals = normal_map[rows, columns]
    maximum_score = _maximum_visible_scores(
        positions,
        normals,
        rotations,
        camera,
        width,
        height,
        capture["fov_degrees"],
        capture["zoom"],
        depth_maps,
    )

    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    projected_bounds = []
    for rotation in rotations:
        projected, depth = _project(
            mesh_vertices @ rotation.T,
            camera,
            width,
            height,
            capture["fov_degrees"],
            capture["zoom"],
        )
        projected_bounds.append(_bbox(projected[depth > 0]))

    accumulator = np.zeros((len(positions), 3), dtype=np.float32)
    weights = np.zeros(len(positions), dtype=np.float32)
    optical_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    for frame_index, rotation in enumerate(rotations):
        image = first if frame_index == 0 else _load_rgb(frames[frame_index])
        mask = _load_mask(mask_paths[frame_index])
        if image.shape[:2] != (height, width) or mask.shape != (height, width):
            raise ValueError(f"Dataset frame {frame_index} dimensions do not match frame 0")
        rotated = positions @ rotation.T
        rotated_normals = normals @ rotation.T
        view = camera[0] - rotated
        view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-8)
        score = np.sum(rotated_normals * view, axis=1)
        projected, depth = _project(
            rotated,
            camera,
            width,
            height,
            capture["fov_degrees"],
            capture["zoom"],
        )
        visible = _visible(projected, depth, depth_maps[0][frame_index], depth_maps[1])
        selected = np.where(visible & (score > 0) & (maximum_score > 0))[0]
        if not len(selected):
            continue
        sample_points = _flow_map(
            projected[selected],
            projected_bounds[frame_index],
            depth_maps[0][frame_index],
            depth_maps[1],
            mask,
            optical_flow,
        )
        sampled = _remap_points(
            image,
            sample_points,
            cv2.INTER_LINEAR,
        )
        sampled_mask = _remap_points(
            mask.astype(np.uint8),
            sample_points,
            cv2.INTER_NEAREST,
        ).astype(bool)
        target = selected[sampled_mask]
        blend_weight = np.power(
            score[target] / maximum_score[target],
            VIEW_WEIGHT_POWER,
        )
        accumulator[target] += sampled[sampled_mask].astype(np.float32) * blend_weight[:, None]
        weights[target] += blend_weight

    texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    assigned = weights > 0
    colors = np.zeros_like(accumulator, dtype=np.uint8)
    colors[assigned] = np.clip(
        accumulator[assigned] / weights[assigned, None],
        0,
        255,
    ).astype(np.uint8)
    texture[rows[assigned], columns[assigned]] = colors[assigned]
    assigned_map = np.zeros_like(valid)
    assigned_map[rows[assigned], columns[assigned]] = True
    texture = _pad_texture(texture, assigned_map)
    return texture, int(assigned.sum()), int(len(assigned))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path, help="Posed Jakkanna GLB in metres")
    parser.add_argument("metadata", type=Path, help="Jakkanna mesh/camera metadata JSON")
    parser.add_argument("dataset", type=Path, help="RealityScan image-layer dataset")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Textured GLB output")
    parser.add_argument("--texture-size", type=int, default=2048, choices=(512, 1024, 2048, 4096))
    args = parser.parse_args()

    scene = trimesh.load(args.mesh, force="scene", process=False)
    if len(scene.geometry) != 1:
        raise SystemExit("Jakkanna proxy GLB must contain exactly one geometry")
    mesh = next(iter(scene.geometry.values()))
    if getattr(mesh.visual, "uv", None) is None:
        raise SystemExit("Jakkanna proxy GLB has no UV coordinates")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    texture, assigned, total = bake(mesh, metadata, args.dataset.resolve(), args.texture_size)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(texture),
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.asarray(mesh.visual.uv),
        material=material,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _export_unlit_glb(mesh, args.output)
    texture_path = args.output.with_name(args.output.stem + "_texture.png")
    Image.fromarray(texture).save(texture_path)
    print(f"Baked {assigned}/{total} UV texels from {FRAME_COUNT} SCAIL-2 views")
    print(f"Textured proxy: {args.output}")
    print(f"Texture atlas: {texture_path}")


if __name__ == "__main__":
    main()
