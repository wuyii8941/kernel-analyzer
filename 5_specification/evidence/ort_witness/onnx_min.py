"""Minimal ONNX ModelProto encoder (protobuf wire format by hand) so that ONNX Runtime can be used as an independent
witness without the `onnx` package. Field numbers follow onnx.proto."""
import struct
import numpy as np
import onnxruntime as ort

def _varint(n):
    if n < 0: n += 1 << 64
    out = bytearray()
    while True:
        b = n & 0x7F; n >>= 7
        if n: out.append(b | 0x80)
        else: out.append(b); return bytes(out)
def _key(f, wt): return _varint((f << 3) | wt)
def f_int(f, v): return _key(f, 0) + _varint(int(v))
def f_bytes(f, b): return _key(f, 2) + _varint(len(b)) + b
def f_str(f, s): return f_bytes(f, s.encode())
def f_float(f, v): return _key(f, 5) + struct.pack('<f', v)

T = dict(FLOAT=1, INT64=7, DOUBLE=11, FLOAT16=10, BFLOAT16=16, FLOAT8E4M3FN=17, FLOAT8E5M2=19, BOOL=9, INT32=6)
NP = {np.float32: 1, np.float64: 11, np.int64: 7, np.bool_: 9, np.int32: 6}

def attr(name, v):
    b = f_str(1, name)
    if isinstance(v, float): return f_bytes(5, b + f_float(2, v) + f_int(20, 1))       # attribute field of NodeProto = 5
    if isinstance(v, bool) or isinstance(v, int): return f_bytes(5, b + f_int(3, int(v)) + f_int(20, 2))
    if isinstance(v, str): return f_bytes(5, b + f_bytes(4, v.encode()) + f_int(20, 3))
    if isinstance(v, (list, tuple)) and all(isinstance(x, int) for x in v):
        return f_bytes(5, b + b''.join(f_int(8, x) for x in v) + f_int(20, 7))
    if isinstance(v, (list, tuple)): return f_bytes(5, b + b''.join(f_float(7, x) for x in v) + f_int(20, 6))
    raise TypeError(v)
def node(op, ins, outs, domain='', **attrs):
    b = b''.join(f_str(1, i) for i in ins) + b''.join(f_str(2, o) for o in outs) + f_str(4, op)
    b += b''.join(attr(k, v) for k, v in attrs.items())
    if domain: b += f_str(7, domain)
    return f_bytes(1, b)                                                                   # node field of GraphProto = 1
def vinfo(name, elem, shape, field):
    dims = b''.join(f_bytes(1, f_int(1, d) if isinstance(d, int) else f_str(2, d)) for d in shape)
    tt = f_int(1, elem) + f_bytes(2, dims)
    return f_bytes(field, f_str(1, name) + f_bytes(2, f_bytes(1, tt)))
def model(nodes, inputs, outputs, opset=21, ir=10):
    g = b''.join(nodes) + f_str(2, 'g') + b''.join(vinfo(n, e, s, 11) for n, e, s in inputs) \
        + b''.join(vinfo(n, e, s, 12) for n, e, s in outputs)
    return f_int(1, ir) + f_str(2, 'witness') + f_bytes(7, g) + f_bytes(8, f_str(1, '') + f_int(2, opset))

def run(nodes, feeds, outputs, opset=21, ir=10):
    """feeds: dict name -> np.ndarray; outputs: list of (name, elem_type, shape)"""
    ins = [(k, NP[v.dtype.type], list(v.shape)) for k, v in feeds.items()]
    m = model(nodes, ins, outputs, opset, ir)
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(m, so, providers=['CPUExecutionProvider'])
    return sess.run([o[0] for o in outputs], feeds)
