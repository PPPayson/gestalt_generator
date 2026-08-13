"""One Modal app for the perceptual-grouping stimulus pipeline.
  LLM  cls  -> Qwen2.5-7B proposes object/background pairs (source=propose only)
  SDXL cls  -> SDXL + canny-ControlNet + rembg: object cutouts, backgrounds, ControlNet renders
  run()     -> local entrypoint that orchestrates every stage from a YAML config.

Usage:  modal run app.py::run --config configs/example.yaml      (or: python run.py configs/example.yaml)
"""
import io, modal

SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
CN = "diffusers/controlnet-canny-sdxl-1.0"
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
def _dl_sdxl():
    from huggingface_hub import snapshot_download
    snapshot_download(SDXL_MODEL, cache_dir=CACHE, allow_patterns=["*.json", "*.txt", "*fp16*", "*.model"])
    snapshot_download(CN, cache_dir=CACHE, allow_patterns=["*.json", "*.safetensors"])
    from rembg import new_session; new_session("u2net")
sdxl_image = sdxl_image.run_function(_dl_sdxl, volumes={CACHE: sdxl_vol, "/models": u2})

llm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "transformers==4.44.2", "accelerate==0.34.2",
                 "safetensors==0.8.0", "huggingface_hub==0.36.2", "numpy==2.4.6")
)
def _dl_llm():
    from huggingface_hub import snapshot_download
    snapshot_download(LLM_MODEL, cache_dir=CACHE, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
llm_image = llm_image.run_function(_dl_llm, volumes={CACHE: llm_vol})

app = modal.App("perceptual-grouping")


@app.cls(gpu="L4", image=sdxl_image, volumes={CACHE: sdxl_vol, "/models": u2},
         timeout=1800, max_containers=8)
class SDXL:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
        self.torch = torch
        cn = ControlNetModel.from_pretrained(CN, torch_dtype=torch.float16,
                                             use_safetensors=True, cache_dir=CACHE)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_MODEL, controlnet=cn, torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True, cache_dir=CACHE).to("cuda")
        self.pipe.watermark = None

    def _run(self, prompt, neg, ctrl, scale, steps, seed):
        return self.pipe(
            prompt, negative_prompt=neg, image=ctrl, controlnet_conditioning_scale=float(scale),
            width=1024, height=1024, num_inference_steps=steps, guidance_scale=6.5,
            generator=self.torch.Generator("cuda").manual_seed(seed),
        ).images[0]

    @modal.method()
    def gen_object(self, prompt, seed=0):
        from PIL import Image
        from rembg import remove
        blank = Image.new("RGB", (1024, 1024), "white")
        img = self._run(
            f"top-down product photo of one {prompt}, centered on a pure flat white seamless "
            f"background, evenly lit, realistic, sharp",
            "table, plate, shadow, text, multiple", blank, 0.0, 28, seed)
        b = io.BytesIO(); remove(img).save(b, format="PNG"); return b.getvalue()

    @modal.method()
    def gen_bg(self, prompt, seed=0):
        from PIL import Image
        blank = Image.new("RGB", (1024, 1024), "white")
        img = self._run(
            f"top-down flat-lay photo of a {prompt}, even soft light, plain, nothing on it",
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
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(LLM_MODEL, cache_dir=CACHE)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL, torch_dtype=torch.bfloat16, cache_dir=CACHE).to("cuda")

    @modal.method()
    def propose(self, illusion, inducer_shape, natural=True, n=12):
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


@app.local_entrypoint()
def run(config: str = "configs/example.yaml"):
    from pipeline import (config as C, silhouettes, propose, assets, validate,
                          pairs as P, dress_composite, dress_controlnet, contact)
    from pipeline.utils import load_all_candidates, DATA
    cfg = C.load(config)
    print(f"== {cfg['name']} : pairs.source={cfg['pairs']['source']} ==")
    metas = silhouettes.build(cfg)
    sdxl = SDXL()
    if cfg["pairs"]["source"] == "propose":
        propose.run(LLM(), cfg, metas)
        cand = load_all_candidates()
        objs = {it["object"] for v in cand.values() for it in v}
        bgs = {it["background"] for v in cand.values() for it in v}
        assets.ensure(sdxl, objs, bgs, cfg["assets"]["best_of"])
        prs = validate.run(cfg, metas)
    else:
        prs = P.resolve(cfg, metas)
        objs, bgs = P.needed(prs)
        assets.ensure(sdxl, objs, bgs, cfg["assets"]["best_of"])
    if "composite" in cfg["dress"]["backends"]:
        dress_composite.run(cfg, metas, prs)
    if "controlnet" in cfg["dress"]["backends"]:
        dress_controlnet.run(sdxl, cfg, metas, prs)
    if cfg["output"]["contact_sheet"]:
        contact.run(cfg)
    print(f"DONE -> {DATA / 'out'} (stimuli), {DATA / 'gt'} (ground truth), {DATA / 'manifest.csv'}")
