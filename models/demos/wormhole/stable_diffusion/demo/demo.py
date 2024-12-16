# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import json
import torch
import pytest
import numpy as np
from PIL import Image
from loguru import logger
from tqdm.auto import tqdm
from datasets import load_dataset
import os
import time
import math
import sd3_impls

from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection, T5EncoderModel
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionPipeline,
    StableDiffusion3Pipeline,
)

from models.utility_functions import skip_for_grayskull
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.stable_diffusion.sd_pndm_scheduler import TtPNDMScheduler
from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
from models.demos.wormhole.stable_diffusion.sd35_preprocessing import sd35_preprocessor
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_unet_2d_condition_model_new_conv import (
    UNet2DConditionModel as UNet2D,
)
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import round_up_to_tile_dim
from torchvision.transforms import ToTensor
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance
from scipy import integrate

import datetime
from other_impls import SD3Tokenizer, SDClipModel, SDXLClipG, T5XXLModel
from safetensors import safe_open
from sd3_impls import (
    SDVAE,
    BaseModel,
    CFGDenoiser,
    SD3LatentFormat,
    SkipLayerCFGDenoiser,
)


def load_inputs(input_path):
    with open(input_path) as f:
        input_data = json.load(f)
        assert input_data, "Input data is empty."
        prompt = [item["prompt"] for item in input_data]
        return prompt


def constant_prop_time_embeddings(timesteps, sample, time_proj):
    timesteps = timesteps[None]
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = time_proj(timesteps)
    return t_emb


def tt_guide(noise_pred, guidance_scale):  # will return latents
    noise_pred_uncond = noise_pred[:1, :, :, :]
    noise_pred_text = ttnn.slice(
        noise_pred,
        [1, 0, 0, 0],
        [
            noise_pred.shape[0],
            noise_pred.shape[1],
            noise_pred.shape[2],
            noise_pred.shape[3],
        ],
    )
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


def _save_image_and_latents(latents, iter, vae, pre_fix="", pre_fix2=""):
    latents = ttnn.to_torch(latents).to(torch.float32)
    pre_fix = "" if pre_fix == "" else f"{pre_fix}_"
    pre_fix2 = "" if pre_fix2 == "" else f"{pre_fix2}_"
    _latents = 1 / 0.18215 * latents

    with torch.no_grad():
        image = vae.decode(_latents).sample
    # Image post-processing
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images][0]
    pil_images.save(f"{pre_fix}{pre_fix2}image_iter_{iter}.png")


def calculate_fid_score(imgs_path1, imgs_path2):
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(imgs_path1, real=False)
    fid.update(imgs_path2, real=True)
    return fid.compute()


def preprocess_images(image_paths):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = image.resize((299, 299))
        image = ToTensor()(image)
        images.append(image)
    return torch.stack(images)


# Note: Sigma shift value, publicly released models use 3.0
SHIFT = 3.0
# Naturally, adjust to the width/height of the model you have
WIDTH = 1024
HEIGHT = 1024
# Pick your prompt
PROMPT = "a photo of a cat"
# Most models prefer the range of 4-5, but still work well around 7
CFG_SCALE = 4.5
# Different models want different step counts but most will be good at 50, albeit that's slow to run
# sd3_medium is quite decent at 28 steps
STEPS = 40
# Seed
SEED = 23
# SEEDTYPE = "fixed"
SEEDTYPE = "rand"
# SEEDTYPE = "roll"
# Actual model file path
MODEL = "models/sd3.5_medium.safetensors"
# MODEL = "models/sd3.5_large_turbo.safetensors"
# MODEL = "models/sd3.5_large.safetensors"
# VAE model file path, or set None to use the same model file
VAEFile = None  # "models/sd3_vae.safetensors"
# Optional init image file path
INIT_IMAGE = None
# If init_image is given, this is the percentage of denoising steps to run (1.0 = full denoise, 0.0 = no denoise at all)
DENOISE = 0.6
# Output file path
OUTDIR = "outputs"
# SAMPLER
# SAMPLER = "euler"
SAMPLER = "dpmpp_2m"
CONFIGS = {
    "sd3_medium": {
        "shift": 1.0,
        "cfg": 5.0,
        "steps": 50,
        "sampler": "dpmpp_2m",
    },
    "sd3.5_medium": {
        "shift": 3.0,
        "cfg": 5.0,
        "steps": 50,
        "sampler": "dpmpp_2m",
        "skip_layer_config": {
            "scale": 2.5,
            "start": 0.01,
            "end": 0.20,
            "layers": [7, 8, 9],
            "cfg": 4.0,
        },
    },
}
CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}


def load_into(f, model, prefix, device, dtype=None):
    """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module."""
    for key in f.keys():
        if key.startswith(prefix) and not key.startswith("loss."):
            path = key[len(prefix) :].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(f"Skipping key '{key}' in safetensors file as '{p}' does not exist in python model")
                        break
            if obj is None:
                continue
            try:
                tensor = f.get_tensor(key).to(device=device)
                if dtype is not None:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e


class ClipL:
    def __init__(self):
        with safe_open("models/clip_l.safetensors", framework="pt", device="cpu") as f:
            self.model = SDClipModel(
                layer="hidden",
                layer_idx=-2,
                device="cpu",
                dtype=torch.float32,
                layer_norm_hidden_state=False,
                return_projected_pooled=False,
                textmodel_json_config=CLIPL_CONFIG,
            )
            load_into(f, self.model.transformer, "", "cpu", torch.float32)
        # self.model = CLIPTextModelWithProjection.from_pretrained("models/clip_l.safetensors", use_safetensors=True)


T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}
CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}


class ClipG:
    def __init__(self):
        with safe_open("models/clip_g.safetensors", framework="pt", device="cpu") as f:
            self.model = SDXLClipG(CLIPG_CONFIG, device="cpu", dtype=torch.float32)
            load_into(f, self.model.transformer, "", "cpu", torch.float32)


class T5XXL:
    def __init__(self):
        with safe_open("models/t5xxl.safetensors", framework="pt", device="cpu") as f:
            self.model = T5XXLModel(T5_CONFIG, device="cpu", dtype=torch.float32)
            load_into(f, self.model.transformer, "", "cpu", torch.float32)


class SD3:
    def __init__(self, model, shift, verbose=False):
        with safe_open(model, framework="pt", device="cpu") as f:
            self.model = BaseModel(
                shift=shift,
                file=f,
                prefix="model.diffusion_model.",
                device="cpu",
                dtype=torch.float16,
                verbose=verbose,
            ).eval()
            load_into(f, self.model, "model.", "cpu", torch.float16)


class VAE:
    def __init__(self, model):
        with safe_open(model, framework="pt", device="cpu") as f:
            self.model = SDVAE(device="cpu", dtype=torch.float16).eval().cpu()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, self.model, prefix, "cpu", torch.float16)


class SD3Inferencer:
    def print(self, txt):
        if self.verbose:
            print(txt)

    def load(self, model=MODEL, vae=VAEFile, shift=SHIFT, verbose=False):
        self.verbose = verbose
        print("Loading tokenizers...")
        # NOTE: if you need a reference impl for a high performance CLIP tokenizer instead of just using the HF transformers one,
        # check https://github.com/Stability-AI/StableSwarmUI/blob/master/src/Utils/CliplikeTokenizer.cs
        # (T5 tokenizer is different though)
        self.tokenizer = SD3Tokenizer()
        print("Loading OpenAI CLIP L...")
        # self.clip_l = ClipL()
        # print("Loading OpenCLIP bigG...")
        # self.clip_g = ClipG()
        # print("Loading Google T5-v1-XXL...")
        # self.t5xxl = T5XXL()
        # print(f"Loading SD3 model {os.path.basename(model)}...")
        self.sd3 = SD3(model, shift, verbose)
        print("Loading VAE model...")
        self.vae = VAE(vae or model)
        print("Models loaded.")

    def get_empty_latent(self, width, height):
        self.print("Prep an empty latent...")
        return torch.ones(1, 16, height // 8, width // 8, device="cpu") * 0.0609

    def get_empty_latent(self, width, height):
        self.print("Prep an empty latent...")
        return torch.ones(1, 16, height // 8, width // 8, device="cpu") * 0.0609

    def get_sigmas(self, sampling, steps):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(sampling.sigma(ts))
        sigs += [0.0]
        return torch.FloatTensor(sigs)

    def get_noise(self, seed, latent):
        generator = torch.manual_seed(seed)
        self.print(f"dtype = {latent.dtype}, layout = {latent.layout}, device = {latent.device}")
        return torch.randn(
            latent.size(),
            dtype=torch.float32,
            layout=latent.layout,
            generator=generator,
            device="cpu",
        ).to(latent.dtype)

    def get_cond(self, prompt):
        self.print("Encode prompt...")
        tokens = self.tokenizer.tokenize_with_weights(prompt)
        l_out, l_pooled = self.clip_l.model.encode_token_weights(tokens["l"])
        g_out, g_pooled = self.clip_g.model.encode_token_weights(tokens["g"])
        t5_out, t5_pooled = self.t5xxl.model.encode_token_weights(tokens["t5xxl"])
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        return torch.cat([lg_out, t5_out], dim=-2), torch.cat((l_pooled, g_pooled), dim=-1)

    def max_denoise(self, sigmas):
        max_sigma = float(self.sd3.model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

    def fix_cond(self, cond):
        cond, pooled = (cond[0].half().cuda(), cond[1].half().cuda())
        return {"c_crossattn": cond, "y": pooled}

    def do_sampling(
        self,
        latent,
        seed,
        conditioning,
        neg_cond,
        steps,
        cfg_scale,
        sampler="dpmpp_2m",
        denoise=1.0,
        skip_layer_config={},
    ) -> torch.Tensor:
        self.print("Sampling...")
        latent = latent.half().cuda()
        self.sd3.model = self.sd3.model.cuda()
        noise = self.get_noise(seed, latent).cuda()
        sigmas = self.get_sigmas(self.sd3.model.model_sampling, steps).cuda()
        sigmas = sigmas[int(steps * (1 - denoise)) :]
        conditioning = self.fix_cond(conditioning)  # cuda
        neg_cond = self.fix_cond(neg_cond)  # cuda
        extra_args = {"cond": conditioning, "uncond": neg_cond, "cond_scale": cfg_scale}
        noise_scaled = self.sd3.model.model_sampling.noise_scaling(sigmas[0], noise, latent, self.max_denoise(sigmas))
        sample_fn = getattr(sd3_impls, f"sample_{sampler}")
        denoiser = SkipLayerCFGDenoiser if skip_layer_config.get("scale", 0) > 0 else CFGDenoiser
        latent = sample_fn(
            denoiser(self.sd3.model, steps, skip_layer_config),
            noise_scaled,
            sigmas,
            extra_args=extra_args,
        )
        latent = SD3LatentFormat().process_out(latent)
        self.sd3.model = self.sd3.model.cpu()
        self.print("Sampling done")
        return latent

    def gen_image(
        self,
        prompts=[PROMPT],
        width=WIDTH,
        height=HEIGHT,
        steps=STEPS,
        cfg_scale=CFG_SCALE,
        sampler=SAMPLER,
        seed=SEED,
        seed_type=SEEDTYPE,
        out_dir=OUTDIR,
        init_image=INIT_IMAGE,
        denoise=DENOISE,
        skip_layer_config={},
    ):
        latent = self.get_empty_latent(width, height)
        if init_image:
            image_data = Image.open(init_image)
            image_data = image_data.resize((width, height), Image.LANCZOS)
            latent = self.vae_encode(image_data)
            latent = SD3LatentFormat().process_in(latent)
        neg_cond = self.get_cond("")
        seed_num = None
        pbar = tqdm(enumerate(prompts), total=len(prompts), position=0, leave=True)
        for i, prompt in pbar:
            if seed_type == "roll":
                seed_num = seed if seed_num is None else seed_num + 1
            elif seed_type == "rand":
                seed_num = torch.randint(0, 100000, (1,)).item()
            else:  # fixed
                seed_num = seed
            conditioning = self.get_cond(prompt)
            sampled_latent = self.do_sampling(
                latent,
                seed_num,
                conditioning,
                neg_cond,
                steps,
                cfg_scale,
                sampler,
                denoise if init_image else 1.0,
                skip_layer_config,
            )
            image = self.vae_decode(sampled_latent)
            save_path = os.path.join(out_dir, f"{i:06d}.png")
            self.print(f"Will save to {save_path}")
            image.save(save_path)
            self.print("Done")

    def vae_encode(self, image) -> torch.Tensor:
        self.print("Encoding image to latent...")
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        batch_images = np.expand_dims(image_np, axis=0).repeat(1, axis=0)
        image_torch = torch.from_numpy(batch_images)
        image_torch = 2.0 * image_torch - 1.0
        image_torch = image_torch.cuda()
        self.vae.model = self.vae.model.cuda()
        latent = self.vae.model.encode(image_torch).cpu()
        self.vae.model = self.vae.model.cpu()
        self.print("Encoded")
        return latent

    def vae_decode(self, latent) -> Image.Image:
        self.print("Decoding latent to image...")
        latent = latent.cuda()
        self.vae.model = self.vae.model.cuda()
        image = self.vae.model.decode(latent)
        image = image.float()
        self.vae.model = self.vae.model.cpu()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
        decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
        decoded_np = decoded_np.astype(np.uint8)
        out_image = Image.fromarray(decoded_np)
        self.print("Decoded")
        return out_image


def run_demo_inference_35(
    device,
    prompt=PROMPT,
    model=MODEL,
    out_dir=OUTDIR,
    postfix=None,
    seed=SEED,
    seed_type=SEEDTYPE,
    sampler=None,
    steps=None,
    cfg=None,
    shift=None,
    width=WIDTH,
    height=HEIGHT,
    vae=VAEFile,
    init_image=INIT_IMAGE,
    denoise=DENOISE,
    skip_layer_cfg=False,
    verbose=False,
):
    disable_persistent_kernel_cache()
    device.enable_program_cache()

    steps = steps or CONFIGS.get(os.path.splitext(os.path.basename(model))[0], {}).get("steps", 50)
    cfg = cfg or CONFIGS.get(os.path.splitext(os.path.basename(model))[0], {}).get("cfg", 5)
    shift = shift or CONFIGS.get(os.path.splitext(os.path.basename(model))[0], {}).get("shift", 3)
    sampler = sampler or CONFIGS.get(os.path.splitext(os.path.basename(model))[0], {}).get("sampler", "dpmpp_2m")
    if skip_layer_cfg:
        skip_layer_config = CONFIGS.get(os.path.splitext(os.path.basename(model))[0], {}).get("skip_layer_config", {})
        cfg = skip_layer_config.get("cfg", cfg)
    else:
        skip_layer_config = {}

    with torch.no_grad():
        inferencer = SD3Inferencer()
        inferencer.load(model, vae, shift, verbose)
        ttnn.CONFIG.enable_model_cache = True
        # inferencer.clip_l.model = preprocess_model_parameters(
        #     model_name='sd3.5_ClipL', initialize_model=lambda: inferencer.clip_l.model, custom_preprocessor=sd35_preprocessor, device=device
        # )
        # print('Parsing ClipL done')
        # inferencer.clip_g.model = preprocess_model_parameters(
        #     model_name='sd3.5_ClipG', initialize_model=lambda: inferencer.clip_g.model, custom_preprocessor=sd35_preprocessor, device=device
        # )
        # print('Parsing ClipG done')
        # inferencer.t5xxl.model = preprocess_model_parameters(
        #     model_name='sd3.5_T5XXL', initialize_model=lambda: inferencer.t5xxl.model, custom_preprocessor=sd35_preprocessor, device=device
        # )
        # print('Parsing T5XXL done')
        # inferencer.vae.model = preprocess_model_parameters(
        #    model_name='sd3.5_VAE', initialize_model=lambda: inferencer.vae.model, custom_preprocessor=sd35_preprocessor, device=device
        # )
        # Above models load fine. Error in loading below model
        print("Parsing SD3 done")
        inferencer.sd3.model = preprocess_model_parameters(
            model_name="sd3.5_DiT",
            initialize_model=lambda: inferencer.sd3.model,
            custom_preprocessor=sd35_preprocessor,
            device=device,
        )
        print("Parsing SD3 done")

        if isinstance(prompt, str):
            if os.path.splitext(prompt)[-1] == ".txt":
                with open(prompt, "r") as f:
                    prompts = [l.strip() for l in f.readlines()]
            else:
                prompts = [prompt]

        out_dir = os.path.join(
            out_dir,
            os.path.splitext(os.path.basename(model))[0],
            os.path.splitext(os.path.basename(prompt))[0][:50]
            + (postfix or datetime.datetime.now().strftime("_%Y-%m-%dT%H-%M-%S")),
        )
        print(f"Saving images to {out_dir}")
        os.makedirs(out_dir, exist_ok=False)
        inferencer.gen_image(
            prompts,
            width,
            height,
            steps,
            cfg,
            sampler,
            seed,
            seed_type,
            out_dir,
            init_image,
            denoise,
            skip_layer_config,
        )
    print("done")


def run_demo_inference(device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size=(256, 256)):
    disable_persistent_kernel_cache()
    device.enable_program_cache()

    # Until di/dt issues are resolved
    os.environ["SLOW_MATMULS"] = "1"
    assert (
        num_inference_steps >= 4
    ), f"PNDMScheduler only supports num_inference_steps >= 4. Found num_inference_steps={num_inference_steps}"

    height, width = image_size

    torch_device = "cpu"
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.to(torch_device)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    # 4. load the K-LMS scheduler with some fitting parameters.
    ttnn_scheduler = TtPNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        skip_prk_steps=True,
        steps_offset=1,
        device=device,
    )

    text_encoder.to(torch_device)
    unet.to(torch_device)

    config = unet.config
    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    input_height = 64
    input_width = 64
    reader_patterns_cache = {} if height == 512 and width == 512 else None
    model = UNet2D(device, parameters, 2, input_height, input_width, reader_patterns_cache)

    guidance_scale = 7.5  # Scale for classifier-free guidance
    generator = torch.manual_seed(174)  # 10233 Seed generator to create the inital latent noise
    batch_size = 1

    # Initial random noise
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // vae_scale_factor, width // vae_scale_factor),
        generator=generator,
    )
    latents = latents.to(torch_device)

    ttnn_scheduler.set_timesteps(num_inference_steps)

    latents = latents * ttnn_scheduler.init_noise_sigma
    rand_latents = torch.tensor(latents)
    rand_latents = ttnn.from_torch(rand_latents, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # ttnn_latents = ttnn.from_torch(ttnn_latents, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_latent_model_input = ttnn.concat([rand_latents, rand_latents], dim=0)
    _tlist = []
    for t in ttnn_scheduler.timesteps:
        _t = constant_prop_time_embeddings(t, ttnn_latent_model_input, unet.time_proj)
        _t = _t.unsqueeze(0).unsqueeze(0)
        _t = _t.permute(2, 0, 1, 3)  # pre-permute temb
        _t = ttnn.from_torch(_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _tlist.append(_t)

    time_step = ttnn_scheduler.timesteps.tolist()
    i = 0

    inputs = load_inputs(input_path)
    input_prompts = inputs[:num_prompts]

    while i < num_prompts:
        ttnn_scheduler.set_timesteps(num_inference_steps)
        input_prompt = [input_prompts[i]]
        i = i + 1

        experiment_name = f"input_data_{i}_{height}x{width}"
        logger.info(f"input prompt : {input_prompt}")
        batch_size = len(input_prompt)

        ## First, we get the text_embeddings for the prompt. These embeddings will be used to condition the UNet model.
        # Tokenizer and Text Encoder
        text_input = tokenizer(
            input_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

        # For classifier-free guidance, we need to do two forward passes: one with the conditioned input (text_embeddings),
        # and another with the unconditional embeddings (uncond_embeddings).
        # In practice, we can concatenate both into a single batch to avoid doing two forward passes.
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        ttnn_text_embeddings = torch.nn.functional.pad(text_embeddings, (0, 0, 0, 19))
        ttnn_text_embeddings = ttnn.from_torch(
            ttnn_text_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        iter = 0
        ttnn_latents = rand_latents
        # # Denoising loop
        for index in tqdm(range(len(time_step))):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            ttnn_latent_model_input = ttnn.concat([ttnn_latents, ttnn_latents], dim=0)
            _t = _tlist[index]
            t = time_step[index]
            # predict the noise residual
            with torch.no_grad():
                ttnn_output = model(
                    ttnn_latent_model_input,  # input
                    timestep=_t,
                    encoder_hidden_states=ttnn_text_embeddings,
                    class_labels=None,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                    return_dict=True,
                    config=config,
                )
            # perform guidance
            noise_pred = tt_guide(ttnn_output, guidance_scale)

            ttnn_latents = ttnn_scheduler.step(noise_pred, t, ttnn_latents).prev_sample
            _save_image_and_latents(ttnn_latents, iter, vae, pre_fix=f"{experiment_name}_tt", pre_fix2="")

            iter += 1

        latents = ttnn.to_torch(ttnn_latents).to(torch.float32)

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        # Image post-processing
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images][0]
        ttnn_output_path = f"{experiment_name}_ttnn.png"
        pil_images.save(ttnn_output_path)


def run_interactive_demo_inference(device, num_inference_steps, image_size=(256, 256)):
    disable_persistent_kernel_cache()
    device.enable_program_cache()

    # Until di/dt issues are resolved
    os.environ["SLOW_MATMULS"] = "1"
    assert (
        num_inference_steps >= 4
    ), f"PNDMScheduler only supports num_inference_steps >= 4. Found num_inference_steps={num_inference_steps}"

    height, width = image_size

    torch_device = "cpu"
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.to(torch_device)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    # 4. load the K-LMS scheduler with some fitting parameters.
    ttnn_scheduler = TtPNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        skip_prk_steps=True,
        steps_offset=1,
        device=device,
    )

    text_encoder.to(torch_device)
    unet.to(torch_device)

    config = unet.config
    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    input_height = 64
    input_width = 64
    reader_patterns_cache = {} if height == 512 and width == 512 else None
    model = UNet2D(device, parameters, 2, input_height, input_width, reader_patterns_cache)

    guidance_scale = 7.5  # Scale for classifier-free guidance
    generator = torch.manual_seed(174)  # 10233 Seed generator to create the inital latent noise
    batch_size = 1

    # Initial random noise
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // vae_scale_factor, width // vae_scale_factor),
        generator=generator,
    )
    latents = latents.to(torch_device)

    ttnn_scheduler.set_timesteps(num_inference_steps)

    latents = latents * ttnn_scheduler.init_noise_sigma
    rand_latents = torch.tensor(latents)
    rand_latents = ttnn.from_torch(rand_latents, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # ttnn_latents = ttnn.from_torch(ttnn_latents, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_latent_model_input = ttnn.concat([rand_latents, rand_latents], dim=0)
    _tlist = []
    for t in ttnn_scheduler.timesteps:
        _t = constant_prop_time_embeddings(t, ttnn_latent_model_input, unet.time_proj)
        _t = _t.unsqueeze(0).unsqueeze(0)
        _t = _t.permute(2, 0, 1, 3)  # pre-permute temb
        _t = ttnn.from_torch(_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _tlist.append(_t)

    time_step = ttnn_scheduler.timesteps.tolist()

    while 1:
        ttnn_scheduler.set_timesteps(num_inference_steps)
        print("Enter the input promt, or q to exit:")
        new_prompt = input()
        if len(new_prompt) > 0:
            input_prompt = [new_prompt]
        if input_prompt[0] == "q":
            break

        experiment_name = f"interactive_{height}x{width}"
        logger.info(f"input prompt : {input_prompt}")
        batch_size = len(input_prompt)

        ## First, we get the text_embeddings for the prompt. These embeddings will be used to condition the UNet model.
        # Tokenizer and Text Encoder
        text_input = tokenizer(
            input_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

        # For classifier-free guidance, we need to do two forward passes: one with the conditioned input (text_embeddings),
        # and another with the unconditional embeddings (uncond_embeddings).
        # In practice, we can concatenate both into a single batch to avoid doing two forward passes.
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        ttnn_text_embeddings = torch.nn.functional.pad(text_embeddings, (0, 0, 0, 19))
        ttnn_text_embeddings = ttnn.from_torch(
            ttnn_text_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        iter = 0
        ttnn_latents = rand_latents
        # # Denoising loop
        total_accum = 0
        for index in tqdm(range(len(time_step))):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            t0 = time.time()
            ttnn_latent_model_input = ttnn.concat([ttnn_latents, ttnn_latents], dim=0)
            _t = _tlist[index]
            t = time_step[index]
            # predict the noise residual
            with torch.no_grad():
                ttnn_output = model(
                    ttnn_latent_model_input,  # input
                    timestep=_t,
                    encoder_hidden_states=ttnn_text_embeddings,
                    class_labels=None,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                    return_dict=True,
                    config=config,
                )
            # perform guidance
            noise_pred = tt_guide(ttnn_output, guidance_scale)

            ttnn_latents = ttnn_scheduler.step(noise_pred, t, ttnn_latents).prev_sample
            total_accum += time.time() - t0
            iter += 1
        print(f"Time taken for {iter} iterations: total: {total_accum:.3f}")

        latents = ttnn.to_torch(ttnn_latents).to(torch.float32)

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        # Image post-processing
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images][0]
        ttnn_output_path = f"{experiment_name}_ttnn.png"
        pil_images.save(ttnn_output_path)


def run_demo_inference_diffusiondb(
    device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size=(256, 256)
):
    disable_persistent_kernel_cache()
    device.enable_program_cache()

    # Until di/dt issues are resolved
    os.environ["SLOW_MATMULS"] = "1"

    assert (
        num_inference_steps >= 4
    ), f"PNDMScheduler only supports num_inference_steps >= 4. Found num_inference_steps={num_inference_steps}"
    # 0. Load a sample prompt from the dataset
    dataset = load_dataset("poloclub/diffusiondb", "2m_random_1k")
    data_1k = dataset["train"]

    height, width = image_size

    torch_device = "cpu"
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.to(torch_device)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    # 4. load the K-LMS scheduler with some fitting parameters.
    ttnn_scheduler = TtPNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        skip_prk_steps=True,
        steps_offset=1,
        device=device,
    )

    text_encoder.to(torch_device)
    unet.to(torch_device)

    config = unet.config
    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    input_height = 64
    input_width = 64
    reader_patterns_cache = {} if height == 512 and width == 512 else None
    model = UNet2D(device, parameters, 2, input_height, input_width, reader_patterns_cache)

    guidance_scale = 7.5  # Scale for classifier-free guidance
    generator = torch.manual_seed(174)  # 10233 Seed generator to create the inital latent noise
    batch_size = 1

    # Initial random noise
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // vae_scale_factor, width // vae_scale_factor),
        generator=generator,
    )
    latents = latents.to(torch_device)

    ttnn_scheduler.set_timesteps(num_inference_steps)

    latents = latents * ttnn_scheduler.init_noise_sigma
    rand_latents = torch.tensor(latents)
    rand_latents = ttnn.from_torch(rand_latents, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # ttnn_latents = ttnn.from_torch(ttnn_latents, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_latent_model_input = ttnn.concat([rand_latents, rand_latents], dim=0)
    _tlist = []
    for t in ttnn_scheduler.timesteps:
        _t = constant_prop_time_embeddings(t, ttnn_latent_model_input, unet.time_proj)
        _t = _t.unsqueeze(0).unsqueeze(0)
        _t = _t.permute(2, 0, 1, 3)  # pre-permute temb
        _t = ttnn.from_torch(_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _tlist.append(_t)

    time_step = ttnn_scheduler.timesteps.tolist()

    i = 0
    while i < num_prompts:
        experiment_name = f"diffusiondb_{i}__{height}x{width}"
        ttnn_scheduler.set_timesteps(num_inference_steps)
        input_prompt = [f"{data_1k['prompt'][i]}"]

        image = np.array(data_1k["image"][i])
        ref_images = Image.fromarray(image)
        ref_img_path = f"{experiment_name}_ref.png"
        ref_images.save(ref_img_path)
        i = i + 1

        logger.info(f"input_prompts: {input_prompt}")

        ## First, we get the text_embeddings for the prompt. These embeddings will be used to condition the UNet model.
        # Tokenizer and Text Encoder
        text_input = tokenizer(
            input_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

        # For classifier-free guidance, we need to do two forward passes: one with the conditioned input (text_embeddings),
        # and another with the unconditional embeddings (uncond_embeddings).
        # In practice, we can concatenate both into a single batch to avoid doing two forward passes.
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        ttnn_text_embeddings = torch.nn.functional.pad(text_embeddings, (0, 0, 0, 19))
        ttnn_text_embeddings = ttnn.from_torch(
            ttnn_text_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        iter = 0
        ttnn_latents = rand_latents
        # # Denoising loop
        for index in tqdm(range(len(time_step))):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            ttnn_latent_model_input = ttnn.concat([ttnn_latents, ttnn_latents], dim=0)
            _t = _tlist[index]
            t = time_step[index]
            # predict the noise residual
            with torch.no_grad():
                ttnn_output = model(
                    ttnn_latent_model_input,  # input
                    timestep=_t,
                    encoder_hidden_states=ttnn_text_embeddings,
                    class_labels=None,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                    return_dict=True,
                    config=config,
                )

            # perform guidance
            noise_pred = tt_guide(ttnn_output, guidance_scale)
            ttnn_latents = ttnn_scheduler.step(noise_pred, t, ttnn_latents).prev_sample
            _save_image_and_latents(ttnn_latents, iter, vae, pre_fix=f"{experiment_name}_tt", pre_fix2="")

            iter += 1
            enable_persistent_kernel_cache()

        latents = ttnn.to_torch(ttnn_latents).to(torch.float32)
        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        # Image post-processing
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images][0]
        ttnn_output_path = f"{experiment_name}_ttnn.png"
        pil_images.save(ttnn_output_path)

        ttnn_paths = [ttnn_output_path, ttnn_output_path]
        ttnn_images = preprocess_images(ttnn_paths)
        ref_paths = [ref_img_path, ref_img_path]
        ref_images = preprocess_images(ref_paths)

        # Calculate FID scores
        fid_score_ref_ttnn = calculate_fid_score(ref_images, ttnn_images)
        logger.info(f"FID Score (Reference vs TTNN): {fid_score_ref_ttnn}")

        # calculate Clip score
        clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

        clip_score_ttnn = clip_score(ttnn_images[0], input_prompt)
        clip_score_ttnn = clip_score_ttnn.detach()
        logger.info(f"CLIP Score (TTNN): {clip_score_ttnn}")


# @pytest.mark.skip(reason="#9945: Skip for now since this breaks on WH because of di/dt")
@skip_for_grayskull()
@torch.no_grad()
def test_demo_35(device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size):
    return run_demo_inference_35(device)


# @pytest.mark.skip(reason="#9945: Skip for now since this breaks on WH because of di/dt")
@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "num_prompts",
    ((1),),
)
@pytest.mark.parametrize(
    "num_inference_steps",
    ((5),),
)
@pytest.mark.parametrize(
    "image_size",
    ((512, 512),),
)
def test_demo(device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size):
    # return run_demo_inference(device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size)
    return run_demo_inference_35(device)


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "num_prompts",
    ((1),),
)
@pytest.mark.parametrize(
    "num_inference_steps",
    ((50),),
)
@pytest.mark.parametrize(
    "image_size",
    ((512, 512),),
)
def test_demo_diffusiondb(device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size):
    return run_demo_inference_diffusiondb(device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size)


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "num_inference_steps",
    ((50),),
)
@pytest.mark.parametrize(
    "image_size",
    ((512, 512),),
)
def test_interactive_demo(device, num_inference_steps, image_size):
    return run_interactive_demo_inference(device, num_inference_steps, image_size)
