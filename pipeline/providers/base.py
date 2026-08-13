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
    return f"top-down flat-lay photo of a {name}, even soft light, plain, nothing on it"


def background_negative():
    return "objects, food, clutter, text"


def propose_prompt(illusion, inducer_shape, natural, n):
    match = (
        "The object must PLAUSIBLY belong on that surface (ecologically/contextually natural) -- "
        "e.g. hay bales on a wheat field, lily pads on a pond, poker chips on green felt."
        if natural else
        "The surface can be ANYTHING, plausible or not -- be creative -- as long as contrast is "
        "high and the object reads as a disc."
    )
    return (
        f"Invent {n} DIFFERENT scenes for a '{illusion}' visual illusion. Each scene = a real "
        f"DISC-SHAPED object placed on a BACKGROUND surface. Requirements:\n"
        f"1. Viewed straight from above the object is clearly round / circular, approximating the "
        f"inducer: {inducer_shape}. (avoid elongated / thin / multi-part things.)\n"
        f"2. The object stands out with HIGH CONTRAST against the surface (dark on light or light on dark).\n"
        f"3. {match}\n"
        f"Return ONLY a JSON array with keys: object (noun phrase), background (short scene/surface "
        f"phrase), tone (dark|light|colorful)."
    )


@runtime_checkable
class ImageProvider(Protocol):
    def gen_object(self, name: str, seed: int = 0) -> bytes: ...
    def gen_bg(self, name: str, seed: int = 0) -> bytes: ...


@runtime_checkable
class Proposer(Protocol):
    def propose(self, illusion: str, inducer_shape: str, natural: bool = True,
                n: int = 12, seed: int = 0) -> str: ...
