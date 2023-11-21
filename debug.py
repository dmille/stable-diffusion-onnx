import onnx
import IPython
from onnx.onnx_pb import TensorProto
from transformers import CLIPTokenizer
from utils import (
    print_all_outputs,
    make_text_encoder_input_dict,
    make_vae_encoder_input_dict,
    trace_output,
    compare_outputs,
    draw_graph,
)

# model_path = "./models/sd_v15_onnx/text_encoder/model.onnx"
# input_dict = make_text_encoder_input_dict()

input_dict = make_vae_encoder_input_dict()
model_path = "./models/sd_v15_onnx/vae_encoder/model.onnx"


model = onnx.load(model_path)

outputs = [
    (
        input_dict,
        "/text_model/Unsqueeze_3",
        TensorProto.FLOAT,
    ),
]

# compare_outputs(outputs[0], model, n_layers=3)


print("\n\nORT")
G_sub = trace_output(outputs[0], model, "ort", n_layers=2)
draw_graph(G_sub, "graph_ort.png")


print("\n\nGGML")
G_sub = trace_output(outputs[0], model, "ggml", n_layers=4)
draw_graph(G_sub, "graph_ggml.png")
# print("ORT")
# print_all_outputs(outputs, model, "ort")

# print("GGML")
# print_all_outputs(outputs, model, "ggml")
