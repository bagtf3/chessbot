"""Explicit Q/DQ ONNX export for a frozen int8 model.

TRT keys int8 kernel selection off QuantizeLinear / DequantizeLinear node
types, so the graph is built by hand instead of traced. Per layer:

    q_x --DQ(1)--> x --Gemm(DQ(q_w, s), b)--> relu --Clip--Q(1)--> q_y

Activations carry scale 1 (the float value is the int8 value). Weights are
int8 initializers with per-output-channel scale s on axis 0, matching the
[out, in] layout before Gemm's transB.
"""
import numpy as np

from chessbot.int8.export import Int8L1Norm, Int8Linear
from chessbot.int8.layers import EPS

OPSET = 13


class GraphBuilder:
    def __init__(self):
        from onnx import helper, numpy_helper
        self.helper = helper
        self.numpy_helper = numpy_helper
        self.nodes = []
        self.inits = []
        self.one = self.init("one", np.float32(1.0))
        self.zero = self.init("zero", np.int8(0))
        self.lo = self.init("lo", np.float32(-127.0))
        self.hi = self.init("hi", np.float32(127.0))

    def init(self, name, value):
        self.inits.append(self.numpy_helper.from_array(np.asarray(value), name))
        return name

    def node(self, op, inputs, out, **attrs):
        self.nodes.append(self.helper.make_node(op, inputs, [out], name=out, **attrs))
        return out

    def quant(self, x, stem):
        clipped = self.node("Clip", [x, self.lo, self.hi], f"{stem}_clip")
        return self.node("QuantizeLinear", [clipped, self.one, self.zero], f"{stem}_q")

    def dequant(self, q, stem):
        return self.node("DequantizeLinear", [q, self.one, self.zero], f"{stem}_dq")

    def linear(self, layer, x, stem):
        q_w = self.init(f"{stem}_qw", layer.Q.cpu().numpy())
        s = self.init(f"{stem}_s", layer.s.cpu().numpy())
        b = self.init(f"{stem}_b", layer.b.cpu().numpy())
        w = self.node("DequantizeLinear", [q_w, s, self.zero], f"{stem}_w", axis=0)
        y = self.node("Gemm", [x, w, b], f"{stem}_gemm", transB=1)
        if layer.act:
            y = self.node("Relu", [y], f"{stem}_relu")
        return y

    def l1norm(self, layer, x, stem):
        s = self.init(f"{stem}_s", layer.s.cpu().numpy())
        b = self.init(f"{stem}_b", layer.b.cpu().numpy())
        eps = self.init(f"{stem}_eps", np.float32(EPS))
        a = self.node("Abs", [x], f"{stem}_abs")
        m = self.node("ReduceMean", [a], f"{stem}_mean", axes=[-1], keepdims=1)
        d = self.node("Add", [m, eps], f"{stem}_denom")
        n = self.node("Div", [x, d], f"{stem}_norm")
        y = self.node("Mul", [n, s], f"{stem}_scaled")
        return self.node("Add", [y, b], f"{stem}_out")

    def model(self, name, inputs, outputs):
        graph = self.helper.make_graph(self.nodes, name, inputs, outputs,
                                       initializer=self.inits)
        return self.helper.make_model(
            graph, producer_name="chessbot.int8",
            opset_imports=[self.helper.make_opsetid("", OPSET)])


def export_frozen_sequential_qdq(model, path, input_name="input", output_name="output"):
    """Flat frozen Sequential of Int8Linear / Int8L1Norm -> Q/DQ ONNX file.
    Float input on the scale-1 grid, int8 output."""
    import onnx
    from onnx import TensorProto

    layers = list(model.children())
    bad = [type(m).__name__ for m in layers if not isinstance(m, (Int8Linear, Int8L1Norm))]
    if bad or not isinstance(layers[0], Int8Linear):
        raise TypeError(f"expected Int8Linear/Int8L1Norm Sequential, got {bad}")

    g = GraphBuilder()
    q = g.quant(input_name, "input")
    for i, layer in enumerate(layers):
        stem = f"l{i}"
        x = g.dequant(q, stem)
        if isinstance(layer, Int8Linear):
            y = g.linear(layer, x, stem)
        else:
            y = g.l1norm(layer, x, stem)
        q = g.quant(y, f"{stem}_out")
    g.node("Identity", [q], output_name)

    din = int(layers[0].Q.shape[1])
    vi = g.helper.make_tensor_value_info
    proto = g.model(
        "chessbot_int8_qdq",
        [vi(input_name, TensorProto.FLOAT, ["batch", din])],
        [vi(output_name, TensorProto.INT8, ["batch", None])])
    onnx.checker.check_model(proto)
    onnx.save(proto, str(path))
    return proto
