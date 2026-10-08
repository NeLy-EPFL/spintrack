"""Where the ball and the animal are, from a promptable segmentation model (SAM 3).

The model only proposes the ball's silhouette; `spintrack.detect.ball_from_masks`
measures the rim on the image itself. The animal's silhouette places the camera
(`spintrack.autofit.place_camera`). The checkpoint (3.4 GB) downloads on first use.
"""

from __future__ import annotations

import functools
import hashlib
import logging

import numpy as np

log = logging.getLogger("spintrack")

# Where SAM 3 loads from, in order, each at a fixed commit. The first is an ungated
# copy of Meta's checkpoint, byte for byte (model.safetensors SHA-256 6d06f0a5...cc14a),
# redistributed under the SAM License; Meta's own repository is gated (request access,
# then `hf auth login`) and serves if the copy cannot be reached.
SOURCES = (
    ("tkclam/sam3", "4c7c7aa68a625b356934f4ed90597de58b74f3cf"),
    ("facebook/sam3", "3c879f39826c281e95690f02c7821c4de09afae7"),
)
# Masks of both prompts are pooled: on the lab test set "ball" alone missed one rig's
# ball that "sphere" found, and neither proposed a ball where there was none.
PROMPTS = ("ball", "sphere")
# The animal, to place the camera by: on the lab's rigs "insect" finds the fly at scores
# of 0.78-0.94, and "fly" at 0.11-0.32.
ANIMAL_PROMPTS = ("insect",)
MIN_SCORE = 0.1
MAX_MASKS = 5  # per prompt


class SegmenterUnavailable(RuntimeError):
    """Raised when the checkpoint cannot be loaded (offline and not cached, ...)."""


@functools.cache
def _model():
    import torch
    from transformers import Sam3Model, Sam3Processor
    from transformers.utils import logging as hf_logging

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # Keep the checkpoint download's progress, not the loading and HTTP chatter.
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # The Hub's notices (an anonymous download "should" log in); failures still raise.
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    log.info("loading SAM 3 on %s (the first use downloads 3.4 GB)", device)
    errors = []
    for repo, revision in SOURCES:
        try:
            processor = Sam3Processor.from_pretrained(repo, revision=revision)
            model = Sam3Model.from_pretrained(repo, revision=revision, dtype=dtype)
        except OSError as exc:  # unreachable, no access, offline and not cached
            errors.append(f"{repo}: {str(exc).splitlines()[0]}")
            continue
        return model.to(device).eval(), processor, device, dtype
    raise SegmenterUnavailable("cannot load SAM 3: " + "; ".join(errors))


_encoded: dict = {}  # the last image's key and encoding


def _vision(image: np.ndarray):
    """The model's encoding of a 2-D uint8 image, kept for the next prompt on it."""
    import torch

    key = (image.shape, hashlib.blake2b(image.tobytes(), digest_size=16).digest())
    if _encoded.get("key") != key:
        model, processor, device, dtype = _model()
        rgb = np.repeat(np.asarray(image, np.uint8)[..., None], 3, axis=2)
        pixels = processor(images=rgb, return_tensors="pt").pixel_values
        with torch.inference_mode():
            vision = model.get_vision_features(pixels.to(device, dtype))
        _encoded.clear()
        _encoded.update(key=key, vision=vision)
    return _encoded["vision"]


def _masks(image: np.ndarray, prompts) -> tuple[np.ndarray, np.ndarray]:
    """The masks the prompts find in `image`, pooled, best first: `(k, h, w)` uint8 and
    their scores. The image is encoded once and every prompt decoded against it."""
    import torch

    model, processor, device, _ = _model()
    h, w = image.shape
    vision = _vision(image)
    masks, scores = [], []
    with torch.inference_mode():
        for prompt in prompts:
            text = processor(text=prompt, return_tensors="pt").to(device)
            out = model(
                vision_embeds=vision,
                input_ids=text.input_ids,
                attention_mask=text.attention_mask,
            )
            found = processor.post_process_instance_segmentation(
                out, threshold=MIN_SCORE, mask_threshold=0.5, target_sizes=[(h, w)]
            )[0]
            best = found["scores"].float().cpu().numpy().argsort()[::-1][:MAX_MASKS]
            masks += list(found["masks"].cpu().numpy().astype(np.uint8)[best])
            scores += list(found["scores"].float().cpu().numpy()[best])
    order = np.argsort(scores)[::-1]
    if not masks:
        return np.zeros((0, h, w), np.uint8), np.zeros(0)
    return np.stack(masks)[order], np.asarray(scores)[order]


def ball_masks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Candidate ball masks in a 2-D uint8 image, as `(masks, scores)`, best first.

    `masks` is `(k, h, w)` uint8.
    """
    return _masks(image, PROMPTS)


def animal_mask(image: np.ndarray) -> tuple[np.ndarray, float] | None:
    """The animal on the ball in a 2-D uint8 image: its mask and score, or None."""
    masks, scores = _masks(image, ANIMAL_PROMPTS)
    if not len(masks):
        return None
    return masks[0].astype(bool), float(scores[0])
