"""Deployable Modal app for the `modal_hf` provider (maintainer's access point).

Deploy once:   modal deploy app.py
Then set provider: modal_hf in your config; pipeline/providers/modal_hf.py calls these classes by
name via modal.Cls.from_name. The rest of the repo never imports modal.

Model ids are class parameters (swappable per config); weights lazy-load in @modal.enter into a
Volume (any model id works, no image rebuild). Gated models: attach a Secret holding HF_TOKEN.
"""
import io, os, modal

# Defaults (kept in sync with pipeline/providers/base.py). Self-contained so the container needs
# only this file — the `pipeline` package is NOT shipped into the Modal image.
SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
CN_MODEL = "diffusers/controlnet-canny-sdxl-1.0"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CACHE = "/cache"
sdxl_vol = modal.Volume.from_name("sdxl-cache", create_if_missing=True)
u2 = modal.Volume.from_name("u2net-cache", create_if_missing=True)
llm_vol = modal.Volume.from_name("qwen-cache", create_if_missing=True)

sdxl_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.1", "diffusers==0.31.0", "transformers==4.44.2", "accelerate==0.34.2",
        "safetensors==0.8.0", "huggingface_hub==0.36.2", "invisible-watermark==0.2.0",
        "opencv-python-headless==5.0.0.93", "rembg==2.0.78", "onnxruntime==1.28.0",
        "Pillow==12.3.0", "numpy==2.4.6",
    )
    .env({"U2NET_HOME": "/models"})
)
llm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "transformers==4.44.2", "accelerate==0.34.2",
                 "safetensors==0.8.0", "huggingface_hub==0.36.2", "numpy==2.4.6")
)

app = modal.App("perceptual-grouping")


@app.cls(gpu="L4", image=sdxl_image, volumes={CACHE: sdxl_vol, "/models": u2},
         timeout=1800, max_containers=8)
class SDXL:
    model_id: str = modal.parameter(default=SDXL_MODEL)
    cn_id: str = modal.parameter(default=CN_MODEL)

    @modal.enter()
    def load(self):
        import torch
        from huggingface_hub import snapshot_download
        from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
        tok = os.environ.get("HF_TOKEN")                 # present only if a Secret is attached
        # lazy, cached in Volume; fp16 patterns keep the default (SDXL) download lean
        snapshot_download(self.model_id, cache_dir=CACHE, token=tok,
                          allow_patterns=["*.json", "*.txt", "*fp16*", "*.model"])
        snapshot_download(self.cn_id, cache_dir=CACHE, token=tok,
                          allow_patterns=["*.json", "*.safetensors"])
        self.torch = torch
        cn = ControlNetModel.from_pretrained(self.cn_id, torch_dtype=torch.float16,
                                             use_safetensors=True, cache_dir=CACHE)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.model_id, controlnet=cn, torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True, cache_dir=CACHE).to("cuda")
        self.pipe.watermark = None

    def _run(self, prompt, neg, ctrl, scale, steps, seed):
        return self.pipe(
            prompt, negative_prompt=neg, image=ctrl, controlnet_conditioning_scale=float(scale),
            width=1024, height=1024, num_inference_steps=steps, guidance_scale=6.5,
            generator=self.torch.Generator("cuda").manual_seed(seed),
        ).images[0]

    @modal.method()
    def gen_object(self, name, seed=0):
        from PIL import Image
        from rembg import remove
        blank = Image.new("RGB", (1024, 1024), "white")
        img = self._run(
            f"top-down product photo of one {name}, centered on a pure flat white seamless "
            f"background, evenly lit, realistic, sharp",
            "table, plate, shadow, text, multiple", blank, 0.0, 28, seed)
        b = io.BytesIO(); remove(img).save(b, format="PNG"); return b.getvalue()

    @modal.method()
    def gen_bg(self, name, seed=0):
        from PIL import Image
        blank = Image.new("RGB", (1024, 1024), "white")
        img = self._run(
            f"top-down flat-lay photo of a {name}, even soft light, plain, nothing on it",
            "objects, food, clutter, text", blank, 0.0, 28, seed)
        b = io.BytesIO(); img.save(b, format="PNG"); return b.getvalue()

    @modal.method()
    def dress_controlnet(self, sil_png, prompt, scale=0.7, seed=0):
        import numpy as np, cv2
        from PIL import Image
        sil = Image.open(io.BytesIO(sil_png)).convert("L").resize((1024, 1024))
        edges = cv2.Canny(np.array(sil), 80, 160)
        ctrl = Image.fromarray(np.stack([edges] * 3, -1))
        img = self._run(prompt, "blurry, deformed, text, watermark, extra objects", ctrl, scale, 32, seed)
        b = io.BytesIO(); img.save(b, format="PNG"); return b.getvalue()


@app.cls(gpu="L4", image=llm_image, volumes={CACHE: llm_vol}, timeout=1800)
class LLM:
    model_id: str = modal.parameter(default=LLM_MODEL)

    @modal.enter()
    def load(self):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = os.environ.get("HF_TOKEN")
        snapshot_download(self.model_id, cache_dir=CACHE, token=tok,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.model_id, cache_dir=CACHE)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, cache_dir=CACHE).to("cuda")

    @modal.method()
    def propose(self, illusion, inducer_shape, natural=True, n=12, seed=0):
        self.torch.manual_seed(seed)                     # reproducible sampling (same GPU/model)
        match = (
            "The object must PLAUSIBLY belong on that surface (ecologically/contextually natural) -- "
            "e.g. hay bales on a wheat field, lily pads on a pond, poker chips on green felt."
            if natural else
            "The surface can be ANYTHING, plausible or not -- be creative -- as long as contrast is "
            "high and the object reads as a disc."
        )
        prompt = (
            f"Invent {n} DIFFERENT scenes for a '{illusion}' visual illusion. Each scene = a real "
            f"DISC-SHAPED object placed on a BACKGROUND surface. Requirements:\n"
            f"1. Viewed straight from above the object is clearly round / circular, approximating the "
            f"inducer: {inducer_shape}. (avoid elongated / thin / multi-part things.)\n"
            f"2. The object stands out with HIGH CONTRAST against the surface (dark on light or light on dark).\n"
            f"3. {match}\n"
            f"Return ONLY a JSON array with keys: object (noun phrase), background (short scene/surface "
            f"phrase), tone (dark|light|colorful)."
        )
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        ids = self.tok([text], return_tensors="pt").to("cuda")
        out = self.model.generate(**ids, max_new_tokens=900, do_sample=True, temperature=0.9, top_p=0.9)
        return self.tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
