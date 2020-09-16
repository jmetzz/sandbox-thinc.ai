import numpy as np
from thinc.api import Relu
from thinc.api import chain
from thinc.layers import list2ragged, reduce_sum, ParametricAttention, Linear, with_array

model = with_array(Linear())
model.initialize(X=np.zeros((1, 1, 1), dtype="f"))

X = [np.zeros((4, 75), dtype="f")]
Y = np.zeros((1,), dtype="f")
model = chain(
    list2ragged(),
    reduce_sum(),
    Relu(12, dropout=0.5),  # -> Floats2d
    ParametricAttention(12)
)
model.initialize(X=X, Y=Y)
