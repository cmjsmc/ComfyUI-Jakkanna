import importlib
import json
import os
import sys
import tempfile
import types
import unittest

import cv2
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_realityscan_module():
    package_name = "ComfyUI_Jakkanna"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [ROOT]
        sys.modules[package_name] = package

    nodes_name = f"{package_name}.nodes"
    nodes = sys.modules.get(nodes_name)
    if nodes is None:
        nodes = types.ModuleType(nodes_name)
        nodes.__path__ = [os.path.join(ROOT, "nodes")]
        sys.modules[nodes_name] = nodes

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: ""
    sys.modules["folder_paths"] = folder_paths

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    ldm = types.ModuleType("comfy.ldm")
    ldm.__path__ = []
    sam3 = types.ModuleType("comfy.ldm.sam3")
    sam3.__path__ = []
    tracker = types.ModuleType("comfy.ldm.sam3.tracker")

    def unpack_masks(packed):
        bits = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=packed.device)
        return (packed.unsqueeze(-1) & bits).bool().view(*packed.shape[:-1], -1)

    tracker.unpack_masks = unpack_masks
    sys.modules["comfy"] = comfy
    sys.modules["comfy.ldm"] = ldm
    sys.modules["comfy.ldm.sam3"] = sam3
    sys.modules["comfy.ldm.sam3.tracker"] = tracker
    return importlib.import_module(f"{nodes_name}.realityscan")


def pack_masks(masks):
    shifts = torch.arange(8, device=masks.device)
    return (masks.view(*masks.shape[:-1], -1, 8) * (1 << shifts)).sum(-1).byte()


class RealityScanImageSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.realityscan = load_realityscan_module()

    def test_writes_matched_geometry_and_binary_mask_layers(self):
        class TinyImageSetSave(self.realityscan.JakkannaRealityScanImageSetSave):
            WIDTH = 8
            HEIGHT = 2

        images = torch.ones((81, 2, 8, 3), dtype=torch.float32)
        masks = torch.zeros((81, 1, 2, 8), dtype=torch.bool)
        masks[:, :, :, :4] = True
        report = json.dumps({"frames": 81, "objects": 1})

        with tempfile.TemporaryDirectory() as output:
            self.realityscan.folder_paths.get_output_directory = lambda: output
            root, manifest_path = TinyImageSetSave().save(
                images,
                {"packed_masks": pack_masks(masks)},
                report,
                81,
                "Jakkanna/RealityScan/test",
            )

            geometry = os.path.join(root, "layers", ".geometry")
            mask_layer = os.path.join(root, "layers", ".mask")
            self.assertEqual(len(os.listdir(geometry)), 81)
            self.assertEqual(len(os.listdir(mask_layer)), 81)

            saved_image = cv2.imread(os.path.join(geometry, "frame_0000.png"), cv2.IMREAD_COLOR)
            saved_mask = cv2.imread(os.path.join(mask_layer, "frame_0000.png"), cv2.IMREAD_GRAYSCALE)
            self.assertTrue((saved_image[:, :4] == 255).all())
            self.assertTrue((saved_image[:, 4:] == 0).all())
            self.assertEqual(set(saved_mask.reshape(-1).tolist()), {0, 255})

            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["frame_count"], 81)
            self.assertFalse(manifest["duplicate_endpoint"])
            self.assertAlmostEqual(manifest["frames"][-1]["turntable_angle_degrees"], 80 * 360 / 81)

    def test_rejects_noncanonical_frame_count_before_writing(self):
        with self.assertRaisesRegex(ValueError, "exactly 81"):
            self.realityscan.JakkannaRealityScanImageSetSave().save(
                torch.zeros((1, 2, 8, 3)),
                {"packed_masks": torch.zeros((1, 1, 2, 1), dtype=torch.uint8)},
                json.dumps({"frames": 1, "objects": 1}),
                1,
                "Jakkanna/RealityScan/test",
            )


if __name__ == "__main__":
    unittest.main()
