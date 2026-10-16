import os, sys
from fractions import Fraction as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'specs'))
from _base import I, iv, sigmoid, tanh, softmax_row, isum, SpecNotEstablished, SpecInputError  # noqa
import mpmath as mp
def Q(x): return x if isinstance(x, F) else F(str(x)) if isinstance(x, str) else F(x)
def QL(xs): return [QL(x) if isinstance(x, list) else Q(x) for x in xs]
