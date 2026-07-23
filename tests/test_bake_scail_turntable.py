import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from PIL import Image
import trimesh


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPEC = importlib.util.spec_from_file_location(
    "jakkanna_bake_scail_turntable",
    os.path.join(ROOT, "tools", "bake_scail_turntable.py"),
)
BAKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BAKER)


class ScailTurntableBakeTests(unittest.TestCase):
    def test_rotation_matrix_matches_y_turntable(self):
        rotated = BAKER._rotation_matrix([0, 90, 0]) @ np.asarray([1, 0, 0])
        np.testing.assert_allclose(rotated, [0, 0, -1], atol=1e-6)

    def test_default_camera_projects_target_to_image_center(self):
        metadata = {
            "mesh_center": [0, 0, 0],
            "capture": {
                "offset_x": 0,
                "offset_y": 0,
                "yaw_degrees": 0,
                "pitch_degrees": 0,
                "distance": 4.5,
            },
        }
        projected, depth = BAKER._project(
            np.asarray([[0, 0, 0]], dtype=np.float32),
            BAKER._camera(metadata),
            704,
            1280,
            30,
            1,
        )

        np.testing.assert_allclose(projected[0], [352, 640])
        np.testing.assert_allclose(depth, [4.5])

    def test_point_remap_handles_more_than_opencv_row_limit(self):
        image = np.asarray([[[10, 20, 30]]], dtype=np.uint8)
        points = np.zeros((40000, 2), dtype=np.float32)

        sampled = BAKER._remap_points(image, points, 0)

        self.assertEqual(sampled.shape, (40000, 3))
        np.testing.assert_array_equal(sampled[0], [10, 20, 30])
        np.testing.assert_array_equal(sampled[-1], [10, 20, 30])

    def test_glb_declares_unlit_material(self):
        mesh = trimesh.Trimesh(
            vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            faces=np.asarray([[0, 1, 2]]),
            visual=trimesh.visual.TextureVisuals(
                uv=np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.float32),
                material=trimesh.visual.material.PBRMaterial(
                    baseColorTexture=Image.new("RGB", (2, 2), "white"),
                ),
            ),
            process=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.glb"
            BAKER._export_unlit_glb(mesh, path)
            blob = path.read_bytes()
            length, kind = struct.unpack_from("<II", blob, 12)
            self.assertEqual(kind, 0x4E4F534A)
            header = json.loads(blob[20:20 + length])

        self.assertIn("KHR_materials_unlit", header["extensionsUsed"])
        self.assertEqual(
            header["materials"][0]["extensions"]["KHR_materials_unlit"],
            {},
        )

    def test_manifest_requires_exact_nonduplicated_turntable(self):
        manifest = {
            "frame_count": 81,
            "duplicate_endpoint": False,
            "frames": [
                {
                    "index": index,
                    "filename": f"frame_{index:04d}.png",
                    "turntable_angle_degrees": index * 360.0 / 81,
                }
                for index in range(81)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            BAKER._validate_manifest(Path(directory))

            manifest["duplicate_endpoint"] = True
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate 360-degree endpoint"):
                BAKER._validate_manifest(Path(directory))


if __name__ == "__main__":
    unittest.main()
