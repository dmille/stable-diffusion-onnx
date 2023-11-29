import os
import re
import onnx
from onnx import helper
from onnx.onnx_pb import TensorProto
from onnx.tools.net_drawer import GetOpNodeProducer, GetPydotGraph
from onnxruntime.capi.onnxruntime_pybind11_state import Fail as ONNXFailException
from onnxruntime.capi.onnxruntime_pybind11_state import (
    InvalidArgument as ONNXInvalidArgumentException,
)
import onnxruntime as ort

import networkx as nx
import matplotlib.pyplot as plt

import numpy as np

from ggml.contrib.onnx import GgmlRuntimeBackend
from collections import namedtuple
from transformers import CLIPTokenizer
import IPython
import ctypes
from signal import SIGABRT
import traceback

c_globals = ctypes.CDLL(None)  # POSIX


@ctypes.CFUNCTYPE(None, ctypes.c_int)
def sigabrt_handler(sig):
    traceback.print_stack()
    raise Exception("GGML SIGABRT")


c_globals.signal(SIGABRT, sigabrt_handler)

ort.set_default_logger_severity(3)

numpy_type_to_onnx_type = {
    "int64": TensorProto.INT64,
    "int32": TensorProto.INT32,
    "float": TensorProto.FLOAT,
    "float64": TensorProto.FLOAT,
    "float32": TensorProto.FLOAT,
    "bool": TensorProto.BOOL,
    "string": TensorProto.STRING,
}


def printv(string, xoff=10):
    print(" " * xoff + ("\n" + " " * xoff).join(list(string)))


def run_onnx_model(model, input_dict, backend="ggml"):
    if backend == "ort":
        sess = ort.InferenceSession(model.SerializeToString())
        result = sess.run(None, input_dict)
    if backend == "ggml":
        sess = GgmlRuntimeBackend.prepare(model)
        result = sess.run(input_dict)
    return result


def make_vae_encoder_input_dict():
    return {"sample": np.random.random([1, 3, 512, 512]).astype(np.float32)}


def make_vae_decoder_input_dict():
    if os.path.exists("./data/latents.npy"):
        latents = np.load("./data/latents.npy") / 0.18215
    else:
        latents = np.random.random([1, 4, 64, 64]).astype(np.float32)

    return {"latent_sample": latents}


def make_unet_input_dict():
    if os.path.exists("./data/latents.npy"):
        latents = np.load("./data/latents.npy") / 0.18215
    else:
        latents = np.random.random([1, 4, 64, 64]).astype(np.float32)

    return {
        "sample": latents,
        "timestep": np.array([801]).astype(np.int64),
        "encoder_hidden_states": np.random.random([1, 77, 768]).astype(np.float32),
    }


def make_text_encoder_input_dict():
    tokenizer_vocab_path = "./models/sd_v15_onnx/tokenizer/vocab.json"
    tokenizer_merges_path = "./models/sd_v15_onnx/tokenizer/merges.txt"

    tokenizer = CLIPTokenizer(tokenizer_vocab_path, tokenizer_merges_path)

    prompt = "Sample Prompt"

    text_inputs = tokenizer(
        prompt,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=77,
    )
    input_ids = text_inputs.input_ids
    return {"input_ids": np.array(input_ids, dtype=np.int32)}


def export_graph(graph, filename="new_graph.dot"):
    pydot_graph = GetPydotGraph(
        graph,
        name=graph.name,
        rankdir="LR",
        node_producer=GetOpNodeProducer("docstring"),
    )

    pydot_graph.write_dot(filename)


def onnx_nodes_to_nx_graph(nodes, custom_labels=None):
    G = nx.DiGraph()
    input_names = set()
    output_names = set()

    for idx, node in enumerate(nodes):
        if custom_labels is not None:
            custom_label = custom_labels[idx]
        else:
            custom_label = ""

        G.add_node(node.name, type="Op", node=node)
        for output in node.output:
            output_names.add(output)
            G.add_node(output, type="Tensor", label=custom_label)
            G.add_edge(node.name, output)
        for input in node.input:
            input_names.add(input)
            G.add_node(input, type="Tensor")
            G.add_edge(input, node.name)
    return G


def onnx_to_nx_graph(model, reversed=False):
    model_graph = model.graph
    G = onnx_nodes_to_nx_graph(model_graph.node)

    return G


def onnx_subgraph(
    model,
    subgraph_head,
    output_type=TensorProto.FLOAT,
    input_shape=[1, 77],
    output_shape=[
        2,
    ],
):
    G = onnx_to_nx_graph(model)

    ancestors = nx.ancestors(G, subgraph_head) | {subgraph_head}
    G_sub = G.subgraph(ancestors)
    inputs = []
    initializers = []

    all_inputs = {input.name: input for input in model.graph.input}
    all_initializers = {init.name: init for init in model.graph.initializer}

    for node_name, in_degree in G_sub.in_degree:
        if G_sub.nodes[node_name]["type"] == "Tensor" and in_degree == 0:
            if node_name in all_inputs.keys():
                inputs.append(all_inputs[node_name])
            else:
                initializers.append(all_initializers[node_name])

    outputs = []
    for node_name, out_degree in G_sub.out_degree:
        if G_sub.nodes[node_name]["type"] == "Tensor" and out_degree == 0:
            value_info = helper.make_tensor_value_info(
                node_name, output_type, output_shape
            )
            outputs.append(value_info)

    nodes = []
    # Iterate through in correct order
    for node in model.graph.node:
        if node.name in ancestors:
            nx_node = G_sub.nodes[node.name]
            if nx_node["type"] == "Op":
                nodes.append(node)

    subgraph = helper.make_graph(nodes, "subgraph", inputs, outputs, initializers)
    subgraph_model = helper.make_model(subgraph, producer_name="subgraph")
    subgraph_model.opset_import[0].version = model.opset_import[0].version
    subgraph_model.ir_version = model.ir_version
    onnx.checker.check_model(subgraph_model)

    return subgraph_model


# Tools


def compute_output(output, model, backend="ort"):
    input_dict, output_name, output_type = output
    if input_dict == None:
        input_dict = {}

    subgraph_model = onnx_subgraph(model, output_name, output_type)

    if backend == "ort":
        result = run_onnx_model(subgraph_model, input_dict, backend="ort")
    if backend == "ggml":
        result = run_onnx_model(subgraph_model, input_dict, backend="ggml")
    return result


def print_all_outputs(outputs, model, backend="ort"):
    for output in outputs:
        result = compute_output(output, model, backend=backend)
        print(f"{result} {result[0].shape}")


def failsafe_compute_output(output, model, backend="ort"):
    input_dict, output_name, output_type = output

    try:
        result = compute_output((input_dict, output_name, output_type), model, backend)
    except (KeyError, ONNXInvalidArgumentException):
        result = failsafe_compute_output(
            (None, output_name, output_type), model, backend
        )
    except ONNXFailException as e:
        # regex match the last matching parenthesis in the string
        string = e.args[0]
        result = re.findall(r"\(([^)]+)\)", string)[-1]
        numpy_typestr = result.split("(")[1]
        output_type = numpy_type_to_onnx_type[numpy_typestr]

        result = failsafe_compute_output(
            (input_dict, output_name, output_type), model, backend
        )
    return result


def bfs_layer_list(G, output_name, downstream=False, reverse=True, n_layers=3):
    if not downstream:
        G = nx.reverse(G)
    layers = []

    hops = 0
    for layer in nx.bfs_layers(G, output_name):
        if hops > n_layers:
            break

        if G.nodes[layer[0]]["type"] == "Tensor":
            hops += 1
            continue

        output_names = []
        for node_name in layer:
            node = G.nodes[node_name]

            if node["type"] == "Op":
                output_names.append(node["node"].name)
        layers.append(output_names)

    if reverse:
        layers.reverse()

    return layers


def trace_output(output, model, backend="ort", n_layers=3):
    input_dict, output_name, output_type = output
    nodes = []
    G = onnx_to_nx_graph(model)

    layers = bfs_layer_list(
        G, output_name, downstream=False, reverse=True, n_layers=n_layers
    )
    results = []
    nodes = []
    for layer in layers:
        print("-" * 50 + "\n")
        for idx, output_name in enumerate(layer):
            if idx != 0:
                print("\n" + "-" * 10 + "\n")

            node = G.nodes[output_name]["node"]
            print(node)

            for output_name in node.output:
                result = failsafe_compute_output(
                    (input_dict, output_name, output_type), model, backend
                )
                results.append(result)
                nodes.append(node)
                print([result.shape for result in result])

        print("-" * 50)
        if layer != layers[-1]:
            printv("|||||v", xoff=20)
        # subgraph_model = onnx_subgraph(model, output_name, output_type)

    # Convert the result to separate lists
    custom_labels = []
    for node, result in zip(nodes, results):
        repr = node.output[0]
        for output in result:
            repr += "\n" + output.shape.__str__()
            if output.size < 10:
                repr += "\n" + output.__repr__()
            else:
                repr += "\n" + output.flatten().__repr__() + "..."
        repr += "\n"
        custom_labels.append(repr)
    G_sub = onnx_nodes_to_nx_graph(nodes, custom_labels)

    return G_sub


def flatten(l):
    return [item for sublist in l for item in sublist]


def compare_outputs(output, model, n_layers=3):
    input_dict, output_name, output_type = output

    G = onnx_to_nx_graph(model)

    layers = bfs_layer_list(
        G, output_name, downstream=False, reverse=False, n_layers=n_layers
    )

    for node_output_name in flatten(layers):
        node = G.nodes[node_output_name]["node"]
        for node_output_name in node.output:
            try:
                result_onnx = failsafe_compute_output(
                    (input_dict, node_output_name, output_type), model, "ort"
                )

                result_ggml = failsafe_compute_output(
                    (input_dict, node_output_name, output_type), model, "ggml"
                )
            except Exception as e:
                print(e)
            if not np.allclose(result_onnx, result_ggml):
                print("-" * 50)
                print("Output differs for node:\n")
                print(node)
                print("ONNX output: ")
                print(result_onnx)
                print("\nGGML output: ")
                print(result_ggml)
                print("\n")


def draw_graph(G, output_path):
    A = nx.nx_agraph.to_agraph(G)
    A.layout(prog="dot", args="-Grankdir=LR")
    A.draw(output_path)
