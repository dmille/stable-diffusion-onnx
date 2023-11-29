import gc
import os
import json

from typing import Optional

import onnx
import typer
import torch
import numpy as np
from transformers import CLIPTokenizer
from ggml.contrib.onnx import GgmlRuntimeBackend
from tqdm import tqdm
from diffusers import PNDMScheduler
from PIL import Image
import IPython

app = typer.Typer()


class TextEncoder:
    def __init__(self, base_path):
        config_path = os.path.join(base_path, "text_encoder/config.json")
        model_path = os.path.join(base_path, "text_encoder/model.onnx")
        model = onnx.load(model_path)
        self.sess = GgmlRuntimeBackend.prepare(model)
        self.config = json.load(open(config_path))

    def __call__(self, **kwargs):
        return self.sess.run(kwargs)


class UNet:
    def __init__(self, base_path):
        model_path = os.path.join(base_path, "unet/model.onnx")
        config_path = os.path.join(base_path, "unet/config.json")
        model = onnx.load(model_path)
        self.sess = GgmlRuntimeBackend.prepare(model)
        self.config = json.load(open(config_path))

    def __call__(self, **kwargs):
        return self.sess.run(kwargs)


class VAEDecoder:
    def __init__(self, base_path):
        model_path = os.path.join(base_path, "vae_decoder/model.onnx")
        config_path = os.path.join(base_path, "vae_decoder/config.json")
        model = onnx.load(model_path)
        self.sess = GgmlRuntimeBackend.prepare(model)
        self.config = json.load(open(config_path))

        self.scale_factor = 2 ** (
            len(self.config.get("block_out_channels", [-1] * 4)) - 1
        )

    def __call__(self, **kwargs):
        return self.sess.run(kwargs)


class StableDiffusionPipeline:
    def __init__(self, model_path):
        tokenizer_vocab_path = os.path.join(model_path, "tokenizer/vocab.json")
        tokenizer_merges_path = os.path.join(model_path, "tokenizer/merges.txt")
        tokenizer_config_path = os.path.join(
            model_path, "tokenizer/tokenizer_config.json"
        )
        tokenizer_config = json.load(open(tokenizer_config_path))
        self.tokenizer = CLIPTokenizer(
            tokenizer_vocab_path,
            tokenizer_merges_path,
            errors=tokenizer_config.get("errors", "replace"),
            unk_token=tokenizer_config["unk_token"].get("content", "<|endoftext|>"),
        )
        self.tokenizer.model_max_length = tokenizer_config.get("model_max_length", 77)

        self.text_encoder = TextEncoder(model_path)
        self.unet = UNet(model_path)
        self.vae_decoder = VAEDecoder(model_path)

        scheduler_config_path = os.path.join(
            model_path, "scheduler/scheduler_config.json"
        )
        scheduler_config = json.load(open(scheduler_config_path))

        self.scheduler = PNDMScheduler(
            num_train_timesteps=scheduler_config["num_train_timesteps"],
            beta_start=scheduler_config["beta_start"],
            beta_end=scheduler_config["beta_end"],
            beta_schedule=scheduler_config["beta_schedule"],
            trained_betas=scheduler_config["trained_betas"],
            skip_prk_steps=scheduler_config["skip_prk_steps"],
            set_alpha_to_one=scheduler_config["set_alpha_to_one"],
            prediction_type=scheduler_config["prediction_type"],
            timestep_spacing=scheduler_config["timestep_spacing"],
            steps_offset=scheduler_config["steps_offset"],
        )

    def _encode_prompt(self, prompt, negative_prompt=""):
        text_inputs = self.tokenizer(
            prompt,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )

        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder(input_ids=text_input_ids.astype(np.int32))[0]

        max_length = prompt_embeds.shape[1]
        uncond_input = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="np",
        )
        negative_prompt_embeds = self.text_encoder(
            input_ids=uncond_input.input_ids.astype(np.int32)
        )[0]
        return np.concatenate([negative_prompt_embeds, prompt_embeds])

    @classmethod
    def postprocess_image(cls, image):
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))
        image = (image * 255).round().astype("uint8")
        return Image.fromarray(image[0])

    def _prepare_latents(
        self, n_channels, height, width, dtype=np.float32, init_latents=None
    ):
        shape = (
            1,
            n_channels,
            height // self.vae_decoder.scale_factor,
            width // self.vae_decoder.scale_factor,
        )
        latents = init_latents
        if latents is None:
            latents = np.random.randn(*shape).astype(dtype)

        # TODO: scale the initial noise by the standard deviation required by the scheduler
        latents = latents * np.float64(self.scheduler.init_noise_sigma)
        return latents

    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        timesteps: int = 50,
        init_latents=None,
        only_positive_prompt=True,
    ):
        height = self.unet.config.get("sample_size", 64) * self.vae_decoder.scale_factor
        width = self.unet.config.get("sample_size", 64) * self.vae_decoder.scale_factor

        prompt_embeds = self._encode_prompt(prompt)
        if negative_prompt is None:
            prompt_embeds = prompt_embeds[0:1]

        self.scheduler.set_timesteps(timesteps)
        timesteps = self.scheduler.timesteps
        latents = self._prepare_latents(
            self.unet.config.get("in_channels", 4),
            height,
            width,
            dtype=prompt_embeds.dtype,
            init_latents=init_latents,
        )

        for t in tqdm(timesteps):
            if negative_prompt is None:
                latent_model_input = latents
            else:
                latent_model_input = np.concatenate([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(
                torch.from_numpy(latent_model_input), t
            )
            latent_model_input = latent_model_input.cpu().numpy()
            timestep = np.array([t])

            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
            )[0]
            IPython.embed()
            scheduler_output = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents)
            )
            latents = scheduler_output.prev_sample.numpy()
            gc.collect()
        latents = latents.copy()
        gc.collect()
        latents /= self.vae_decoder.config.get("scaling_factor", 0.18215)
        image = self.vae_decoder(latent_sample=latents)[0]
        return self.postprocess_image(image)


@app.command()
def main(
    prompt: str,
    timesteps: int = 50,
    model_path: str = "./models/sd_v15_onnx",
    output_path: str = "./output.png",
):
    pipeline = StableDiffusionPipeline(model_path)
    image = pipeline(prompt, timesteps)
    image.save(output_path)


if __name__ == "__main__":
    app()
