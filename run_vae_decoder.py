import onnx
import numpy as np

from utils import make_vae_decoder_input_dict, run_onnx_model

# from stable_diffusion import StableDiffusionPipeline

model = onnx.load("./models/sd_v15_onnx/vae_decoder/model.onnx")
input_dict = make_vae_decoder_input_dict()

ort_result = run_onnx_model(model, input_dict, backend="ort")
ggml_result = run_onnx_model(model, input_dict, backend="ggml")

# decoded = ort_result[0]
# decoded = ggml_result[0]

# image = StableDiffusionPipeline.postprocess_image(decoded)
# image.save("test.png")
# breakpoint()
assert np.allclose(ort_result[0], ggml_result[0], atol=1e-2)
