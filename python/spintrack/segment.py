"""Where the ball is, from a promptable segmentation model (SAM 3).

The model only proposes the ball's silhouette; `spintrack.detect.ball_from_masks`
measures the rim on the image itself. Optional: needs the `sam` extra (torch,
transformers) and access to the gated `facebook/sam3` checkpoint on Hugging Face.
"""

from __future__ import annotations

import functools
import importlib.util
import logging

import numpy as np

log = logging.getLogger("spintrack")

# Meta's checkpoint, at a fixed commit. The repository is gated: request access on its
# page, then log in with `hf auth login`.
MODEL_ID = "facebook/sam3"
REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
ACCESS = (
    f"request access at https://huggingface.co/{MODEL_ID}, then log in with "
    "`hf auth login`"
)
# Masks of both prompts are pooled: on the lab test set "ball" alone missed one rig's
# ball that "sphere" found, and neither proposed a ball where there was none.
PROMPTS = ("ball", "sphere")
MIN_SCORE = 0.1
MAX_MASKS = 5  # per prompt


class SegmenterUnavailable(RuntimeError):
    """Raised when the model cannot be loaded (extra not installed, no access)."""


def installed() -> bool:
    """Whether the `sam` extra is installed (the checkpoint may still be missing)."""
    packages = ("PIL", "torch", "torchvision", "transformers")
    return all(importlib.util.find_spec(name) is not None for name in packages)


@functools.cache
def _model():
    try:
        import torch
        from transformers import Sam3Model, Sam3Processor
        from transformers.utils import logging as hf_logging
    except ImportError as exc:
        raise SegmenterUnavailable(
            "model-based detection needs the sam extra: pip install 'spintrack[sam]'"
        ) from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # Keep the checkpoint download's progress, not the loading and HTTP chatter.
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log.info("loading SAM 3 on %s (the first use downloads 3.4 GB)", device)
    try:
        processor = Sam3Processor.from_pretrained(MODEL_ID, revision=REVISION)
        model = Sam3Model.from_pretrained(MODEL_ID, revision=REVISION, dtype=dtype)
    except ImportError as exc:  # a dependency of the processor is missing
        raise SegmenterUnavailable(f"{exc}".strip().splitlines()[0]) from exc
    except OSError as exc:  # no access, offline and not cached, ...
        first = str(exc).splitlines()[0]
        raise SegmenterUnavailable(
            f"cannot load {MODEL_ID} ({first}): {ACCESS}"
        ) from exc
    return model.to(device).eval(), processor, device, dtype


def ball_masks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Candidate ball masks in a 2-D uint8 image, as `(masks, scores)`, best first.

    `masks` is `(k, h, w)` uint8. The image is encoded once and every prompt decoded
    against it.
    """
    import torch

    model, processor, device, dtype = _model()
    h, w = image.shape
    rgb = np.repeat(np.asarray(image, np.uint8)[..., None], 3, axis=2)
    pixels = processor(images=rgb, return_tensors="pt").pixel_values
    masks, scores = [], []
    with torch.inference_mode():
        vision = model.get_vision_features(pixels.to(device, dtype))
        for prompt in PROMPTS:
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
