import onnx
import numpy as np

from utils import make_text_encoder_input_dict, run_onnx_model


model = onnx.load("./sd_v15_onnx/text_encoder/model.onnx")
input_dict = make_text_encoder_input_dict()

ort_result = run_onnx_model(model, input_dict, backend="ort")
ggml_result = run_onnx_model(model, input_dict, backend="ggml")

assert np.allclose(ort_result[0], ggml_result[0], atol=1e-2)
