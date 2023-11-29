import onnx
import numpy as np

from utils import make_unet_input_dict, run_onnx_model
from onnx.external_data_helper import load_external_data_for_model
import onnxruntime as ort
import IPython
from ggml.contrib.onnx import GgmlRuntimeBackend

input_dict = make_unet_input_dict()
model_path = "./models/sd_v15_onnx/unet/model.onnx"

# sess = ort.InferenceSession(model_path)
# ort_result = sess.run(None, input_dict)
# assert ort_result[0].shape == (1, 4, 64, 64)


model = onnx.load(model_path)
sess = GgmlRuntimeBackend.prepare(model)

node_types = set()
for node in sess.graph.node:
    node_types.add(node.op_type)
print(node_types)

ggml_result = sess.run(input_dict)
assert ggml_result[0].shape == (1, 4, 64, 64)
