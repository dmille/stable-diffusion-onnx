import onnx
import numpy as np

from utils import make_unet_input_dict, run_onnx_model
from onnx.external_data_helper import load_external_data_for_model

model = onnx.load("./models/sd_v15_onnx/unet/model.onnx", load_external_data=False)
load_external_data_for_model(model, "./models/sd_v15_onnx/unet/")
input_dict = make_unet_input_dict()

ort_result = run_onnx_model(model, input_dict, backend="ort")
breakpoint()
ggml_result = run_onnx_model(model, input_dict, backend="ggml")
breakpoint()
# assert np.allclose(ort_result[0], ggml_result[0], atol=1e-2)
