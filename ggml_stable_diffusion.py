import numpy as np
from stable_diffusion import StableDiffusionPipeline

prompt = "An astronaut riding a unicorn"
timesteps = 10
model_path = "models/sd_v15_onnx"

np.random.seed(42)
latents = np.random.random([1, 4, 64, 64]).astype(np.float32)

ggml_pipeline = StableDiffusionPipeline(model_path)
ggml_image = ggml_pipeline(prompt, timesteps=timesteps, init_latents=latents)
ggml_image.save(f"outputs/output_t{timesteps}_ggml.png")
