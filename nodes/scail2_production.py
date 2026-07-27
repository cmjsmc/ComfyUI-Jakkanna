from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import torch

import comfy.model_management
import comfy.model_sampling
import comfy.samplers
import folder_paths
from comfy.ldm.sam3.tracker import unpack_masks

from .pose_studio import _execution_state_signature
from .scail2_flow_unipc import FlowUniPCMultistepScheduler


UPSTREAM_COMMIT = "5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5"

_VIDEO_FORMATS_DIR = str(Path(__file__).resolve().parents[1] / "video_formats")
_vhs_entry = folder_paths.folder_names_and_paths.get("VHS_video_formats")
if _vhs_entry is None:
    folder_paths.folder_names_and_paths["VHS_video_formats"] = ([_VIDEO_FORMATS_DIR], {".json"})
elif _VIDEO_FORMATS_DIR not in _vhs_entry[0]:
    folder_paths.folder_names_and_paths["VHS_video_formats"] = (
        list(_vhs_entry[0]) + [_VIDEO_FORMATS_DIR],
        set(_vhs_entry[1]) | {".json"},
    )


def _hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_HASH_CACHE_PATH = Path(folder_paths.get_temp_directory()) / "jakkanna_scail2_model_hash_cache.json"


def _hash_file_cached(path):
    path = os.path.realpath(path)
    stat = os.stat(path)
    try:
        cache = json.loads(_HASH_CACHE_PATH.read_text())
    except (OSError, ValueError):
        cache = {}
    entry = cache.get(path)
    if (
        isinstance(entry, dict)
        and entry.get("size") == stat.st_size
        and entry.get("mtime_ns") == stat.st_mtime_ns
    ):
        return entry["sha256"]
    digest = _hash_file(path)
    cache[path] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=_HASH_CACHE_PATH.parent,
        prefix=f".{_HASH_CACHE_PATH.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    try:
        temp_path.replace(_HASH_CACHE_PATH)
    finally:
        temp_path.unlink(missing_ok=True)
    return digest


def _hash_tensor(tensor, binary=False):
    digest = hashlib.sha256()
    frames = tensor if tensor.ndim > 3 else tensor.unsqueeze(0)
    for frame in frames:
        frame = frame.detach().to(device="cpu")
        if binary:
            data = (frame > 0.5).to(torch.uint8).contiguous()
        else:
            data = frame.float().clamp(0, 1).mul(65535).round().to(torch.uint16).contiguous()
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


def _json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _git_commit(root):
    git_path = Path(root) / ".git"
    try:
        if git_path.is_file():
            git_dir = (git_path.parent / git_path.read_text(encoding="utf-8").strip().removeprefix("gitdir: ")).resolve()
        else:
            git_dir = git_path
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = head[5:]
            ref_path = git_dir / ref
            if ref_path.exists():
                head = ref_path.read_text(encoding="utf-8").strip()
            else:
                for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
                    if line and not line.startswith(("#", "^")):
                        commit, name = line.split(" ", 1)
                        if name == ref:
                            head = commit
                            break
        if len(head) == 40 and all(character in "0123456789abcdef" for character in head.lower()):
            return head
    except OSError:
        pass
    return "unknown"


def _attention_backend():
    if comfy.model_management.sage_attention_enabled():
        return "SageAttention"
    if comfy.model_management.flash_attention_enabled():
        return "FlashAttention"
    if comfy.model_management.xformers_enabled():
        return "xFormers"
    if comfy.model_management.pytorch_attention_enabled():
        return "PyTorch SDPA"
    return "ComfyUI split/sub-quadratic"


_GRAPH_ARTIFACT_INPUTS = {
    "UNETLoader": (("unet_name", "diffusion_models"),),
    "LoraLoaderModelOnly": (("lora_name", "loras"),),
    "CheckpointLoaderSimple": (("ckpt_name", "checkpoints"),),
    "CLIPLoader": (("clip_name", "text_encoders"),),
    "DualCLIPLoader": (("clip_name1", "text_encoders"), ("clip_name2", "text_encoders")),
    "TripleCLIPLoader": (
        ("clip_name1", "text_encoders"),
        ("clip_name2", "text_encoders"),
        ("clip_name3", "text_encoders"),
    ),
    "VAELoader": (("vae_name", "vae"),),
    "CLIPVisionLoader": (("clip_name", "clip_vision"),),
}

_GRAPH_SETTING_INPUTS = {
    "KSampler": ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"),
    "BasicScheduler": ("scheduler", "steps", "denoise"),
    "KSamplerSelect": ("sampler_name",),
    "JakkannaSCAIL2FlowUniPC": ("steps", "shift"),
    "CFGGuider": ("cfg",),
    "WanSCAILToVideo": ("replacement_mode", "pose_strength", "pose_start", "pose_end"),
    "LoraLoaderModelOnly": ("lora_name", "strength_model"),
    "CLIPVisionEncode": ("crop",),
    "SAM3_VideoTrack": ("detection_threshold", "max_objects", "detect_interval"),
    "JakkannaSCAIL2TrackValidate": (
        "maximum_area_step_ratio",
        "maximum_centroid_jump",
        "minimum_adjacent_iou",
        "minimum_border_margin",
    ),
}


def _prompt_nodes(prompt, class_type):
    return [
        (str(node_id), node)
        for node_id, node in (prompt or {}).items()
        if isinstance(node, dict) and node.get("class_type") == class_type
    ]


def _single_prompt_node(prompt, class_type):
    nodes = _prompt_nodes(prompt, class_type)
    if len(nodes) != 1:
        raise ValueError(f"Manifest requires exactly one {class_type} node, got {len(nodes)}")
    return nodes[0][1]


def _is_prompt_link(prompt, value):
    return (
        isinstance(value, list)
        and len(value) == 2
        and str(value[0]) in (prompt or {})
        and isinstance(value[1], int)
    )


def _literal_prompt_input(prompt, node, name):
    value = node.get("inputs", {}).get(name)
    return None if _is_prompt_link(prompt, value) else value


def _graph_artifacts(prompt, hash_model_files):
    artifacts = []
    seen = set()
    for node_id, node in (prompt or {}).items():
        for input_name, category in _GRAPH_ARTIFACT_INPUTS.get(node.get("class_type"), ()):
            name = _literal_prompt_input(prompt, node, input_name)
            if not isinstance(name, str) or not name or name == "None":
                continue
            key = (category, name)
            if key in seen:
                continue
            seen.add(key)
            path = folder_paths.get_full_path(category, name)
            if path is None:
                raise ValueError(f"Could not resolve {category} artifact from node {node_id}: {name}")
            artifacts.append({
                "category": category,
                "name": name,
                "size_bytes": os.path.getsize(path),
                "sha256": _hash_file_cached(path) if hash_model_files else None,
            })
    return artifacts


def _graph_settings(prompt):
    settings = []
    for node_id, node in (prompt or {}).items():
        names = _GRAPH_SETTING_INPUTS.get(node.get("class_type"))
        if names is None:
            continue
        inputs = {
            name: value
            for name in names
            if (value := _literal_prompt_input(prompt, node, name)) is not None
        }
        settings.append({
            "node_id": str(node_id),
            "class_type": node.get("class_type"),
            "inputs": inputs,
        })
    return sorted(settings, key=lambda item: (item["class_type"], item["node_id"]))


class JakkannaSCAIL2ProductionInputValidate:
    CANONICAL_RATE = Fraction(16)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video": ("VIDEO",),
                "reference_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("width", "height", "length", "fps", "input_report")
    FUNCTION = "validate"
    CATEGORY = "Jakkanna/SCAIL-2"

    def validate(self, source_video, reference_image):
        components = source_video.get_components()
        images = components.images
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("SCAIL-2 production input must provide an IMAGE tensor batch")
        rate = Fraction(str(components.frame_rate)).limit_denominator(100000)
        if rate != self.CANONICAL_RATE:
            raise ValueError(
                f"SCAIL-2 production input must be exactly {self.CANONICAL_RATE} fps — the canonical "
                f"training rate; prepare the source at 16 fps before running this workflow — got {rate}"
            )
        frames, height, width = images.shape[:3]
        if frames != 81:
            raise ValueError(f"SCAIL-2 production input must contain exactly 81 frames, got {frames}")
        if (width, height) not in {(704, 1280), (1280, 704)}:
            raise ValueError(
                f"SCAIL-2 production input must be 704x1280 or 1280x704, got {width}x{height}"
            )
        if (
            not isinstance(reference_image, torch.Tensor)
            or reference_image.ndim != 4
            or reference_image.shape[-1] < 3
        ):
            raise ValueError("Replacement reference must provide an IMAGE tensor batch")
        if reference_image.shape[0] != 1:
            raise ValueError(f"Replacement reference must contain exactly one image, got {reference_image.shape[0]}")
        if (reference_image.shape[2], reference_image.shape[1]) != (width, height):
            raise ValueError(
                "Replacement reference must match the source canvas exactly: "
                f"source={width}x{height}, reference={reference_image.shape[2]}x{reference_image.shape[1]}"
            )

        minimum = float("inf")
        maximum = float("-inf")
        for index, frame in enumerate(images):
            if not torch.isfinite(frame).all():
                raise ValueError(f"Source frame {index} contains non-finite values")
            frame_min, frame_max = torch.aminmax(frame)
            minimum = min(minimum, float(frame_min.item()))
            maximum = max(maximum, float(frame_max.item()))
        if minimum < 0 or maximum > 1:
            raise ValueError(f"Source IMAGE values must be in [0, 1], got [{minimum:g}, {maximum:g}]")
        if not torch.isfinite(reference_image).all():
            raise ValueError("Replacement reference contains non-finite values")
        reference_min, reference_max = torch.aminmax(reference_image)
        if reference_min < 0 or reference_max > 1:
            raise ValueError(
                "Replacement reference IMAGE values must be in [0, 1], got "
                f"[{float(reference_min.item()):g}, {float(reference_max.item()):g}]"
            )

        report = {
            "source_kind": "image_tensor_batch",
            "source_rgb16_sha256": _hash_tensor(images[..., :3]),
            "source_dtype": str(images.dtype),
            "source_device": str(images.device),
            "source_value_range": [minimum, maximum],
            "width": width,
            "height": height,
            "frames": frames,
            "fps": f"{rate.numerator}/{rate.denominator}",
            "canonical_fps_enforced": int(self.CANONICAL_RATE),
            "duration_seconds": float(frames / rate),
            "constant_frame_rate": True,
            "color_metadata": "IMAGE tensors have no container transfer or primaries metadata",
        }
        return width, height, frames, float(rate), _json(report)


class JakkannaSCAIL2SingleSubjectMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "minimum_area_percent": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "maximum_area_percent": ("FLOAT", {"default": 95.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "mask_report")
    FUNCTION = "validate"
    CATEGORY = "Jakkanna/SCAIL-2"

    def validate(self, masks, minimum_area_percent, maximum_area_percent):
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if masks.shape[0] != 1:
            raise ValueError(f"SAM 3.1 must detect exactly one subject, got {masks.shape[0]}")
        binary = (masks > 0.5).float()
        area_percent = float(binary[0].mean().item() * 100)
        if not minimum_area_percent <= area_percent <= maximum_area_percent:
            raise ValueError(
                f"SAM 3.1 subject mask covers {area_percent:.3f}% of the image; "
                f"expected {minimum_area_percent:.3f}% to {maximum_area_percent:.3f}%"
            )
        report = {
            "objects": 1,
            "area_percent": area_percent,
            "binary_mask_sha256": _hash_tensor(binary, binary=True),
        }
        return binary, _json(report)



class JakkannaSCAIL2ReferencePNG16Save:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "scail2_prep/reference/SCAIL2_reference_704x1280"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("save_report",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Jakkanna/SCAIL-2"

    def save(self, images, filename_prefix):
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            images.shape[2],
            images.shape[1],
        )
        results = []
        files = []
        for index in range(images.shape[0]):
            # Same 16-bit quantization as _hash_tensor, so the saved pixels hash to
            # output_rgb16_sha256 and the production loader's decode reproduces it exactly.
            data = images[index].detach().cpu().float().clamp(0, 1).mul(65535).round().to(torch.uint16).numpy()
            file = f"{filename}_{counter + index:05}_.png"
            path = os.path.join(full_output_folder, file)
            if not cv2.imwrite(path, data[:, :, :3][:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 4]):
                raise RuntimeError(f"Failed to write 16-bit PNG: {path}")
            files.append({"file": file, "file_sha256": _hash_file(path)})
            results.append({"filename": file, "subfolder": subfolder, "type": "output"})
        report = {
            "files": files,
            "bit_depth": 16,
            "width": images.shape[2],
            "height": images.shape[1],
            "output_rgb16_sha256": _hash_tensor(images),
        }
        return {"ui": {"images": results}, "result": (_json(report),)}



class JakkannaSCAIL2TrackValidate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "track_data": ("SAM3_TRACK_DATA",),
                "reference_mask": ("MASK",),
                "expected_frames": ("INT", {"forceInput": True}),
                "maximum_area_step_ratio": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "maximum_centroid_jump": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.01}),
                "minimum_adjacent_iou": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "minimum_border_margin": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.25, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("object_indices", "track_report")
    FUNCTION = "validate"
    CATEGORY = "Jakkanna/SCAIL-2"

    def validate(
        self,
        track_data,
        reference_mask,
        expected_frames,
        maximum_area_step_ratio,
        maximum_centroid_jump,
        minimum_adjacent_iou,
        minimum_border_margin,
    ):
        packed = track_data.get("packed_masks")
        if packed is None:
            raise ValueError("SAM 3.1 produced no source track")
        if packed.shape[0] != expected_frames:
            raise ValueError(f"SAM 3.1 track has {packed.shape[0]} frames; expected {expected_frames}")
        if packed.shape[1] != 1:
            raise ValueError(f"SAM 3.1 track must contain exactly one subject, got {packed.shape[1]}")

        masks = unpack_masks(packed)[:, 0].bool()
        areas = masks.sum(dim=(-1, -2)).float()
        if (areas == 0).any():
            empty = (areas == 0).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(f"SAM 3.1 source track is empty on frames {empty}")

        smaller = torch.minimum(areas[:-1], areas[1:]).clamp(min=1)
        larger = torch.maximum(areas[:-1], areas[1:])
        area_step_ratio = larger / smaller

        height, width = masks.shape[-2:]
        grid_y = torch.arange(height, device=masks.device, dtype=torch.float32).view(1, height, 1)
        grid_x = torch.arange(width, device=masks.device, dtype=torch.float32).view(1, 1, width)
        centroids_y = (masks * grid_y).sum(dim=(-1, -2)) / areas
        centroids_x = (masks * grid_x).sum(dim=(-1, -2)) / areas
        dx = (centroids_x[1:] - centroids_x[:-1]) / width
        dy = (centroids_y[1:] - centroids_y[:-1]) / height
        centroid_jump = torch.sqrt(dx.square() + dy.square())

        intersections = (masks[:-1] & masks[1:]).sum(dim=(-1, -2)).float()
        unions = (masks[:-1] | masks[1:]).sum(dim=(-1, -2)).float().clamp(min=1)
        adjacent_iou = intersections / unions
        occupied_rows = masks.any(dim=2)
        occupied_columns = masks.any(dim=1)
        top = occupied_rows.float().argmax(dim=1)
        bottom = height - 1 - occupied_rows.flip(1).float().argmax(dim=1)
        left = occupied_columns.float().argmax(dim=1)
        right = width - 1 - occupied_columns.flip(1).float().argmax(dim=1)
        border_margins = torch.stack((
            top / height,
            (height - 1 - bottom) / height,
            left / width,
            (width - 1 - right) / width,
        ), dim=1)
        frame_border_margin = border_margins.min(dim=1).values

        max_area_step = float(area_step_ratio.max().item())
        max_jump = float(centroid_jump.max().item())
        min_iou = float(adjacent_iou.min().item())
        min_border = float(frame_border_margin.min().item())
        if max_area_step > maximum_area_step_ratio:
            raise ValueError(
                f"SAM 3.1 track area changes by {max_area_step:.3f}x between adjacent frames; "
                f"limit is {maximum_area_step_ratio:.3f}x"
            )
        if max_jump > maximum_centroid_jump:
            raise ValueError(
                f"SAM 3.1 track centroid jumps {max_jump:.3f} frame diagonals; "
                f"limit is {maximum_centroid_jump:.3f}"
            )
        if min_iou < minimum_adjacent_iou:
            raise ValueError(
                f"SAM 3.1 minimum adjacent-frame IoU is {min_iou:.4f}; "
                f"minimum is {minimum_adjacent_iou:.4f}"
            )
        if min_border < minimum_border_margin:
            clipped = (frame_border_margin < minimum_border_margin).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(
                f"SAM 3.1 source subject is within {minimum_border_margin:.3f} of the canvas edge "
                f"on frames {clipped}"
            )
        if reference_mask.shape[0] != 1 or not (reference_mask > 0.5).any():
            raise ValueError("Validated reference mask must contain exactly one nonempty subject")
        if reference_mask.shape[-2:] != (height, width):
            raise ValueError(
                f"Reference mask is {reference_mask.shape[-1]}x{reference_mask.shape[-2]}; "
                f"source track is {width}x{height}"
            )

        report = {
            "frames": masks.shape[0],
            "objects": 1,
            "minimum_area_percent": float((areas.min() / (height * width) * 100).item()),
            "maximum_area_percent": float((areas.max() / (height * width) * 100).item()),
            "maximum_adjacent_area_ratio": max_area_step,
            "maximum_centroid_jump": max_jump,
            "minimum_adjacent_iou": min_iou,
            "minimum_border_margin": min_border,
            "track_mask_sha256": _hash_tensor(masks, binary=True),
        }
        return "0", _json(report)


def _scheduler(steps, shift, device):
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(steps, device=device, shift=shift)
    return scheduler


def sample_flow_unipc(model, noise, sigmas, extra_args=None, callback=None, disable=None, steps=40, shift=3.0):
    extra_args = {} if extra_args is None else extra_args
    scheduler = _scheduler(steps, shift, noise.device)
    expected_sigmas = scheduler.sigmas.to(device=sigmas.device, dtype=sigmas.dtype)
    if sigmas.shape != expected_sigmas.shape or not torch.equal(sigmas, expected_sigmas):
        raise ValueError("Flow-UniPC sampler requires the matching upstream sigma schedule")

    denoise_mask = extra_args.get("denoise_mask")

    def apply_history(x):
        if denoise_mask is None:
            return x
        return x * denoise_mask + model.latent_image * (1.0 - denoise_mask)

    x = apply_history(model.noise.clone())
    for i, timestep in enumerate(scheduler.timesteps):
        scheduler_sigma = scheduler.sigmas[i].to(device=x.device, dtype=x.dtype)
        model_sigma = (timestep.to(device=x.device, dtype=x.dtype) / scheduler.config.num_train_timesteps).expand(x.shape[0])
        denoised = model(x, model_sigma, **extra_args)
        model_output = (x - denoised) / model_sigma.view((-1,) + (1,) * (x.ndim - 1))
        upstream_denoised = x - scheduler_sigma * model_output
        x = apply_history(scheduler.step(model_output, timestep, x, return_dict=False)[0])
        if callback is not None:
            callback({"x": x, "i": i, "sigma": scheduler_sigma, "sigma_hat": scheduler_sigma, "denoised": upstream_denoised})
    return x


class JakkannaSCAIL2FlowUniPC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 40, "min": 1, "max": 100}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SAMPLER", "SIGMAS")
    RETURN_NAMES = ("sampler", "sigmas")
    FUNCTION = "build"
    CATEGORY = "Jakkanna/SCAIL-2"

    def build(self, steps, shift):
        scheduler = _scheduler(steps, shift, "cpu")
        sampler = comfy.samplers.KSAMPLER(sample_flow_unipc, {"steps": steps, "shift": shift})
        return sampler, scheduler.sigmas.clone()



class UpstreamNoise:
    def __init__(self, noise, seed):
        self.noise = noise
        self.seed = seed

    def generate_noise(self, input_latent):
        expected = input_latent["samples"].shape
        if self.noise.shape != expected:
            raise ValueError(
                f"Upstream noise shape {tuple(self.noise.shape)} does not match latent shape {tuple(expected)}"
            )
        return self.noise


class JakkannaSCAIL2UpstreamNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "noise_seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("NOISE", "STRING")
    RETURN_NAMES = ("noise", "noise_report")
    FUNCTION = "generate"
    CATEGORY = "Jakkanna/SCAIL-2"

    def generate(self, latent, noise_seed):
        shape = latent["samples"].shape
        device = comfy.model_management.get_torch_device()
        generator = torch.Generator(device=device)
        generator.manual_seed(noise_seed)
        noise = torch.randn(shape, dtype=torch.float32, generator=generator, device=device)
        digest = hashlib.sha256(noise.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        report = {
            "seed": noise_seed,
            "shape": list(shape),
            "dtype": "float32",
            "generator_device": str(device),
            "noise_sha256": digest,
        }
        return UpstreamNoise(noise, noise_seed), _json(report)



def _simple_flow_sigmas(shift, steps):
    """The sigma schedule BasicScheduler("simple") produces for a discrete-flow model at `shift`."""
    model_sampling = comfy.model_sampling.ModelSamplingDiscreteFlow()
    model_sampling.set_parameters(shift=shift)
    return comfy.samplers.simple_scheduler(model_sampling, steps)


class JakkannaSCAIL2ProductionManifest:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_images": ("IMAGE",),
                "pose_video_mask": ("IMAGE",),
                "reference_image_mask": ("IMAGE",),
                "input_report": ("STRING", {"forceInput": True}),
                "reference_report": ("STRING", {"forceInput": True}),
                "source_initial_mask_report": ("STRING", {"forceInput": True}),
                "reference_mask_report": ("STRING", {"forceInput": True}),
                "track_report": ("STRING", {"forceInput": True}),
                "noise_report": ("STRING", {"forceInput": True}),
                "positive_prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"forceInput": True}),
                "diffusion_model": (folder_paths.get_filename_list("diffusion_models"),),
                "dpo_lora": (folder_paths.get_filename_list("loras"),),
                "sam_checkpoint": (folder_paths.get_filename_list("checkpoints"),),
                "text_encoder": (folder_paths.get_filename_list("text_encoders"),),
                "clip_vision": (folder_paths.get_filename_list("clip_vision"),),
                "vae": (folder_paths.get_filename_list("vae"),),
                "attention_backend": (["PyTorch SDPA", "FlashAttention", "SageAttention", "xFormers", "ComfyUI split/sub-quadratic"],),
                "hash_model_files": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "scail2_production/manifests/SCAIL2_replacement_production"}),
            },
            "optional": {
                "acceleration_lora": (["None"] + folder_paths.get_filename_list("loras"),),
                "acceleration_lora_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "sampler_name": ("STRING", {"default": "Flow-UniPC"}),
                "scheduler_name": ("STRING", {"default": "upstream Flow-UniPC"}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 100}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "effective_flow_shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Shift actually realized in the sigma schedule; 0 = same as flow_shift. When sigmas is connected, the run fails unless the wired schedule is the Euler/simple flow schedule at this shift."}),
                "mode": (["replacement", "animation"], {"default": "replacement"}),
                "sigmas": ("SIGMAS", {"tooltip": "Wire the sampled sigma schedule to record it verbatim and validate effective_flow_shift against it."}),
                "source_images": ("IMAGE", {"tooltip": "Raw source tensor batch used by SAM and SCAIL. Connect for authoritative capture provenance."}),
                "reference_image": ("IMAGE", {"tooltip": "Raw reference tensor used by SCAIL. Connect for authoritative reference provenance."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("manifest_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Jakkanna/SCAIL-2"

    def save(
        self,
        output_images,
        pose_video_mask,
        reference_image_mask,
        input_report,
        reference_report,
        source_initial_mask_report,
        reference_mask_report,
        track_report,
        noise_report,
        positive_prompt,
        negative_prompt,
        seed,
        diffusion_model,
        dpo_lora,
        sam_checkpoint,
        text_encoder,
        clip_vision,
        vae,
        attention_backend,
        hash_model_files,
        filename_prefix,
        acceleration_lora="None",
        acceleration_lora_strength=0.0,
        sampler_name="Flow-UniPC",
        scheduler_name="upstream Flow-UniPC",
        steps=40,
        flow_shift=3.0,
        cfg=5.0,
        effective_flow_shift=0.0,
        mode="replacement",
        sigmas=None,
        source_images=None,
        reference_image=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        effective_shift = effective_flow_shift if effective_flow_shift > 0 else flow_shift
        sigma_schedule = None
        if sigmas is not None:
            expected = _simple_flow_sigmas(effective_shift, steps)
            wired = sigmas.detach().cpu().float()
            if wired.shape != expected.shape or not torch.equal(wired, expected):
                raise ValueError(
                    "Wired sigma schedule is not the Euler/simple flow schedule at "
                    f"effective_flow_shift={effective_shift:g} with {steps} steps: "
                    f"got {[round(float(s), 6) for s in wired]}, "
                    f"expected {[round(float(s), 6) for s in expected]}"
                )
            sigma_schedule = [float(s) for s in wired]

        actual_attention_backend = _attention_backend()

        input_data = json.loads(input_report)
        reference_data = json.loads(reference_report)
        noise_data = json.loads(noise_report)
        if noise_data.get("seed") != seed:
            raise ValueError(
                f"Manifest seed {seed} does not match generated noise seed {noise_data.get('seed')}"
            )

        source_images_hash = None
        if source_images is not None:
            source_images_hash = _hash_tensor(source_images[..., :3])
            reported_source_hash = input_data.get("source_rgb16_sha256")
            if reported_source_hash and reported_source_hash != source_images_hash:
                raise ValueError("Raw source tensor does not match the validated production input")

        reference_image_hash = None
        if reference_image is not None:
            reference_image_hash = _hash_tensor(reference_image[..., :3])
            reported_reference_hash = reference_data.get("output_rgb16_sha256")
            if reported_reference_hash and reported_reference_hash != reference_image_hash:
                raise ValueError("Raw reference tensor does not match the saved reference report")

        graph_settings = _graph_settings(prompt)
        graph_artifacts = _graph_artifacts(prompt, hash_model_files)
        actual_sampler = sampler_name
        actual_scheduler = scheduler_name
        actual_steps = steps
        actual_cfg = cfg
        actual_mode = mode
        dpo_lora_strength = None
        actual_pose_settings = {
            "pose_strength": None,
            "pose_start": None,
            "pose_end": None,
        }
        clip_vision_crop = None

        if prompt:
            guider_node = _single_prompt_node(prompt, "CFGGuider")
            scail_node = _single_prompt_node(prompt, "WanSCAILToVideo")
            clip_vision_node = _single_prompt_node(prompt, "CLIPVisionEncode")

            scheduler_nodes = _prompt_nodes(prompt, "BasicScheduler")
            flow_unipc_nodes = _prompt_nodes(prompt, "JakkannaSCAIL2FlowUniPC")
            if len(scheduler_nodes) + len(flow_unipc_nodes) != 1:
                raise ValueError(
                    "Manifest requires exactly one BasicScheduler or "
                    f"JakkannaSCAIL2FlowUniPC node, got {len(scheduler_nodes) + len(flow_unipc_nodes)}"
                )
            if scheduler_nodes:
                sampler_node = _single_prompt_node(prompt, "KSamplerSelect")
                scheduler_node = scheduler_nodes[0][1]
                actual_scheduler = _literal_prompt_input(prompt, scheduler_node, "scheduler")
                actual_steps = _literal_prompt_input(prompt, scheduler_node, "steps")
                actual_sampler = _literal_prompt_input(prompt, sampler_node, "sampler_name")
            else:
                flow_unipc_node = flow_unipc_nodes[0][1]
                actual_scheduler = "upstream Flow-UniPC"
                actual_sampler = "Flow-UniPC"
                actual_steps = _literal_prompt_input(prompt, flow_unipc_node, "steps")
                actual_flow_shift = _literal_prompt_input(prompt, flow_unipc_node, "shift")
                if actual_flow_shift != flow_shift:
                    raise ValueError(
                        f"Manifest flow_shift={flow_shift:g} does not match "
                        f"JakkannaSCAIL2FlowUniPC shift={actual_flow_shift}"
                    )
            actual_cfg = _literal_prompt_input(prompt, guider_node, "cfg")
            replacement_mode = bool(_literal_prompt_input(prompt, scail_node, "replacement_mode"))
            actual_mode = "replacement" if replacement_mode else "animation"
            actual_pose_settings = {
                name: _literal_prompt_input(prompt, scail_node, name)
                for name in ("pose_strength", "pose_start", "pose_end")
            }
            clip_vision_crop = _literal_prompt_input(prompt, clip_vision_node, "crop")

            if actual_steps != steps:
                raise ValueError(f"Manifest steps={steps} does not match BasicScheduler steps={actual_steps}")
            if actual_cfg != cfg:
                raise ValueError(f"Manifest cfg={cfg:g} does not match CFGGuider cfg={actual_cfg}")
            if actual_mode != mode:
                raise ValueError(f"Manifest mode={mode} does not match WanSCAILToVideo mode={actual_mode}")
            if str(actual_sampler).lower() != str(sampler_name).lower():
                raise ValueError(
                    f"Manifest sampler={sampler_name} does not match KSamplerSelect sampler={actual_sampler}"
                )

            for _, lora_node in _prompt_nodes(prompt, "LoraLoaderModelOnly"):
                name = _literal_prompt_input(prompt, lora_node, "lora_name")
                strength = _literal_prompt_input(prompt, lora_node, "strength_model")
                if name == dpo_lora:
                    dpo_lora_strength = strength
                if name == acceleration_lora and strength != acceleration_lora_strength:
                    raise ValueError(
                        f"Manifest acceleration LoRA strength={acceleration_lora_strength:g} "
                        f"does not match the loader strength={strength}"
                    )
            if dpo_lora_strength is None:
                raise ValueError(f"Could not find the DPO LoRA loader in the prompt graph: {dpo_lora}")

        model_files = {
            "diffusion_model": ("diffusion_models", diffusion_model),
            "dpo_lora": ("loras", dpo_lora),
            "sam_checkpoint": ("checkpoints", sam_checkpoint),
            "text_encoder": ("text_encoders", text_encoder),
            "clip_vision": ("clip_vision", clip_vision),
            "vae": ("vae", vae),
        }
        if acceleration_lora != "None":
            model_files["acceleration_lora"] = ("loras", acceleration_lora)
        artifacts = {}
        for key, (category, name) in model_files.items():
            path = folder_paths.get_full_path(category, name)
            if path is None:
                raise ValueError(f"Could not resolve {category} artifact: {name}")
            artifacts[key] = {
                "name": name,
                "size_bytes": os.path.getsize(path),
                "sha256": _hash_file_cached(path) if hash_model_files else None,
            }

        if graph_artifacts:
            graph_artifact_keys = {
                (artifact["category"], artifact["name"])
                for artifact in graph_artifacts
            }
            missing = sorted(
                name
                for category, name in model_files.values()
                if (category, name) not in graph_artifact_keys
            )
            if missing:
                raise ValueError(
                    f"Manifest artifacts are not loaded by the prompt graph: {', '.join(missing)}"
                )

        pose_state_signature = None
        pose_animation = None
        pose_studio_id = None
        pose_nodes = _prompt_nodes(prompt, "VNCCS_PoseStudio") + _prompt_nodes(prompt, "VNCCS_PoseStudioOpenPose")
        if pose_nodes:
            if len(pose_nodes) != 1:
                raise ValueError(f"Manifest requires exactly one Pose Studio node, got {len(pose_nodes)}")
            pose_data = _literal_prompt_input(prompt, pose_nodes[0][1], "pose_data")
            if not isinstance(pose_data, str):
                raise ValueError("Pose Studio pose_data must be embedded in the prompt graph")
            parsed_pose_data = json.loads(pose_data)
            pose_state_signature = _execution_state_signature(parsed_pose_data)
            pose_animation = parsed_pose_data.get("animation")
            pose_studio_id = parsed_pose_data.get("studio_id")

        workflow = (extra_pnginfo or {}).get("workflow")
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline": f"SCAIL-2 single-subject 81-frame character {mode}",
            "upstream_scail2_commit": UPSTREAM_COMMIT,
            "comfyui_commit": _git_commit(Path(folder_paths.__file__).resolve().parent),
            "jakkanna_commit": _git_commit(Path(__file__).resolve().parents[1]),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "attention_backend": actual_attention_backend,
            "declared_attention_backend": attention_backend,
            "settings": {
                "resolution": "704x1280 or 1280x704",
                "frames": 81,
                "sampler": actual_sampler,
                "scheduler": actual_scheduler,
                "steps": actual_steps,
                "flow_shift": flow_shift,
                "effective_flow_shift": effective_shift,
                "sigma_schedule": sigma_schedule,
                "cfg": actual_cfg,
                "dpo_lora_strength": dpo_lora_strength,
                "acceleration_lora": None if acceleration_lora == "None" else {
                    "name": acceleration_lora,
                    "strength": acceleration_lora_strength,
                },
                "replacement_mode": actual_mode == "replacement",
                **actual_pose_settings,
                "clip_vision_crop": clip_vision_crop,
                "seed": seed,
            },
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "input": input_data,
            "reference": reference_data,
            "source_initial_mask": json.loads(source_initial_mask_report),
            "reference_mask": json.loads(reference_mask_report),
            "source_track": json.loads(track_report),
            "noise": noise_data,
            "source_capture_rgb16_sha256": source_images_hash,
            "reference_input_rgb16_sha256": reference_image_hash,
            "pose_state_sha256": pose_state_signature,
            "pose_studio_id": pose_studio_id,
            "pose_animation": pose_animation,
            "pose_video_mask_rgb16_sha256": _hash_tensor(pose_video_mask),
            "reference_image_mask_rgb16_sha256": _hash_tensor(reference_image_mask),
            "decoded_output_rgb16_sha256": _hash_tensor(output_images),
            "decoded_output_frames": output_images.shape[0],
            "artifacts": artifacts,
            "graph_artifacts": graph_artifacts,
            "graph_settings": graph_settings,
            "prompt_graph_sha256": hashlib.sha256(_json(prompt or {}).encode()).hexdigest(),
            "workflow_sha256": hashlib.sha256(_json(workflow or {}).encode()).hexdigest(),
            "production_node_sha256": _hash_file(__file__),
            "flow_unipc_node_sha256": _hash_file(Path(__file__).with_name("scail2_flow_unipc.py")),
        }

        full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            output_images.shape[2],
            output_images.shape[1],
        )
        output_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.json")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return {"ui": {"text": [output_path]}, "result": (output_path,)}

NODE_CLASS_MAPPINGS = {
    "JakkannaSCAIL2FlowUniPC": JakkannaSCAIL2FlowUniPC,
    "JakkannaSCAIL2ProductionInputValidate": JakkannaSCAIL2ProductionInputValidate,
    "JakkannaSCAIL2SingleSubjectMask": JakkannaSCAIL2SingleSubjectMask,
    "JakkannaSCAIL2ReferencePNG16Save": JakkannaSCAIL2ReferencePNG16Save,
    "JakkannaSCAIL2TrackValidate": JakkannaSCAIL2TrackValidate,
    "JakkannaSCAIL2UpstreamNoise": JakkannaSCAIL2UpstreamNoise,
    "JakkannaSCAIL2ProductionManifest": JakkannaSCAIL2ProductionManifest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JakkannaSCAIL2FlowUniPC": "Jakkanna SCAIL-2 Flow-UniPC",
    "JakkannaSCAIL2ProductionInputValidate": "Jakkanna SCAIL-2 Validate Input",
    "JakkannaSCAIL2SingleSubjectMask": "Jakkanna SCAIL-2 Validate Subject Mask",
    "JakkannaSCAIL2ReferencePNG16Save": "Jakkanna SCAIL-2 Save 16-bit Reference",
    "JakkannaSCAIL2TrackValidate": "Jakkanna SCAIL-2 Validate Track",
    "JakkannaSCAIL2UpstreamNoise": "Jakkanna SCAIL-2 Upstream Noise",
    "JakkannaSCAIL2ProductionManifest": "Jakkanna SCAIL-2 Save Manifest",
}
