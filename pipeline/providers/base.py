"""The generation provider contract. A provider is any object with these methods; return values
are plain bytes/str so no framework leaks into the pipeline.

ImageProvider (composite needs only gen_object + gen_bg; dress is optional = the ControlNet path):
    gen_object(name: str, seed: int = 0) -> bytes   # PNG, RGBA cutout of one object
    gen_bg(name: str, seed: int = 0)     -> bytes   # PNG, a plain background surface
    dress(sil_png: bytes, prompt: str, scale=0.7, seed=0) -> bytes   # PNG (OPTIONAL)
Proposer:
    propose(illusion, inducer_shape, natural=True, n=12) -> str      # JSON-array text

To plug in your own model: copy template.py, implement the two gen_* methods (return PNG bytes),
and set `models.image.provider: "yourmodule:YourClass"` in the config.
"""
from typing import Protocol, runtime_checkable

# Default model ids (shared by hf_api + modal_hf so behaviour matches).
SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
CN_MODEL = "diffusers/controlnet-canny-sdxl-1.0"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def object_prompt(name):
    return (f"top-down product photo of one {name}, centered on a pure flat white seamless "
            f"background, evenly lit, realistic, sharp")


def object_negative():
    return "table, plate, shadow, text, multiple"


def background_prompt(name):
    return (f"close-up flat-lay photograph of a {name} surface, seamless even texture filling the whole "
            f"frame, subtle fine natural surface detail, soft even diffuse studio lighting, modern "
            f"minimalist background for design, nothing placed on it")


def background_negative():
    return ("objects, container, plate, bowl, basket, tray, dish, slab, tile, grid, checkerboard, "
            "geometric pattern, rectangle, square, edge, border, frame, seam, fold, folds, crease, "
            "wrinkle, drapery, cast shadow, harsh shadow, vignette, depth of field, bokeh, blur, "
            "tilted angle, 3d render, cgi, watermark, text")


def propose_prompt(illusion, inducer_shape, natural, n):
    match = (
        "Prefer a surface the object plausibly rests on (e.g. poker chip on green felt, "
        "seashell on sand), but the surface must still be uniform and fill the whole frame."
        if natural else
        "The surface can be any uniform material, plausible or not -- be creative -- as long as "
        "contrast is high and the object reads as a disc."
    )
    return (
        f"Invent {n} DIFFERENT scenes for a '{illusion}' visual illusion. Each scene = a real "
        f"DISC-SHAPED object placed on a BACKGROUND SURFACE. Requirements:\n"
        f"1. Viewed straight from above the object is clearly round / circular, approximating the "
        f"inducer: {inducer_shape}. (avoid elongated / thin / multi-part things.)\n"
        f"2. The object stands out with HIGH CONTRAST against the surface (dark on light or light on dark).\n"
        f"3. The BACKGROUND is a full-frame natural material SURFACE with rich all-over texture (e.g. "
        f"marble, granite, terrazzo, travertine, slate, sandstone, limestone, basalt, quartz, cork, "
        f"concrete, pebbled stone, mottled stone, wood, sand, felt, velvet, moss). Speckled, grainy, "
        f"mottled or finely veined textures are all welcome. Pick from a WIDE RANGE of materials and "
        f"colours so the set is varied and visually interesting. It must FILL the whole frame and must "
        f"NOT be an object or container (no plate, bowl, basket, tray, slab, tile, board), and must "
        f"have NO geometric grid/tile pattern and no single dominant crease or straight line. Vary the "
        f"material and colour widely across the {n} scenes.\n"
        f"4. {match}\n"
        f"Return ONLY a JSON array with keys: object (noun phrase), background (a uniform surface/"
        f"material phrase), tone (dark|light|colorful)."
    )


@runtime_checkable
class ImageProvider(Protocol):
    def gen_object(self, name: str, seed: int = 0) -> bytes: ...
    def gen_bg(self, name: str, seed: int = 0) -> bytes: ...


@runtime_checkable
class Proposer(Protocol):
    def propose(self, illusion: str, inducer_shape: str, natural: bool = True,
                n: int = 12, seed: int = 0) -> str: ...
