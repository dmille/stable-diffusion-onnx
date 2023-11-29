import numpy as np
from optimum.onnxruntime import ORTStableDiffusionPipeline

prompt = "An astronaut riding a unicorn"
timesteps = 10
model_path = "models/sd_v15_onnx"

np.random.seed(42)
latents = np.random.random([1, 4, 64, 64]).astype(np.float32)

ort_pipeline = ORTStableDiffusionPipeline.from_pretrained(model_path)
ort_image = ort_pipeline(prompt, num_inference_steps=timesteps, latents=latents).images[
    0
]
ort_image.save(f"outputs/output_t{timesteps}_ort.png")
