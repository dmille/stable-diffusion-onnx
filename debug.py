import onnx
from onnx.onnx_pb import TensorProto
from transformers import CLIPTokenizer
from utils import (
    print_all_outputs,
    make_text_encoder_input_dict,
    make_vae_encoder_input_dict,
    trace_output,
    compare_outputs,
)

model_path = "./models/sd_v15_onnx/vae_encoder/model.onnx"
model = onnx.load(model_path)


input_dict = make_vae_encoder_input_dict()
outputs = [
    (input_dict, "/Gather_3_output_0", TensorProto.INT64),
    (input_dict, "/Gather_3_output_0", TensorProto.INT64),
    (input_dict, "/Concat_output_0", TensorProto.FLOAT),
    (input_dict, "/ConstantOfShape_output_0", TensorProto.FLOAT),
    (input_dict, "/Concat_output_0", TensorProto.FLOAT),
]

# compare_outputs(outputs[0], model, n_layers=3)
# print("\n\nORT")
trace_output(outputs[0], model, "ort", n_layers=3)
# print("\n\nGGML")
# trace_output(outputs[0], model, "ggml", n_layers=3)

# print("ORT")
# print_all_outputs(outputs, model, "ort")

# print("GGML")
# print_all_outputs(outputs, model, "ggml")
