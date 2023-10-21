import onnx
import numpy as np

from utils import make_vae_encoder_input_dict, run_onnx_model


model = onnx.load("./sd_v15_onnx/vae_encoder/model.onnx")
input_dict = make_vae_encoder_input_dict()

ort_result = run_onnx_model(model, input_dict, backend="ort")
ggml_result = run_onnx_model(model, input_dict, backend="ggml")

assert np.allclose(ort_result[0], ggml_result[0], atol=1e-2)
