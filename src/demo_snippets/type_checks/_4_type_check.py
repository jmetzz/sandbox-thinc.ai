from typing import List

from thinc.layers import Relu, Softmax, chain, reduce_max, concatenate
from thinc.model import Model

# Define Custom X/Y types
MyModelX = List[List[float]]
MyModelY = List[List[float]]
model: Model[MyModelX, MyModelY] = chain(Relu(12), Relu(12, dropout=0.2), Softmax(), )

# ERROR: incompatible type "bool", expected "List[List[float]]"
model(False)

# ERROR: List item 0 has incompatible type "str"; expected "float"
model.begin_update([["0"]])

# ERROR: incompatible type "bool", expected "List[List[float]]"
model.predict(True)

# This example should be run with mypy. This is an example of type-level checking
# for network validity.
#
# We first define an invalid network.
# It's invalid because reduce_max expects Array3d as input, while Relu produces
# Array2d as output. chain has type-logic to verify input and output types
# line up.
#
# You should see the error an error,
# _4_type_check.py:32: error: Cannot infer type argument 2 of "chain"
# _4_type_check.py:32: error: Layer outputs type (thinc.types.Floats2d) but the next layer expects (thinc.types.Ragged) as an input
# _4_type_check.py:32: error: Layer input type (thinc.types.Ragged) is not compatible with output (thinc.types.Floats2d) from previous layer
bad_model = chain(Relu(10), reduce_max(), Softmax())

concatenate_model = concatenate(Relu(10), reduce_max(), Relu(10), Relu(10)), reduce_max()

concatenate_chain_model = chain(
    concatenate(Relu(10), reduce_max(), Relu(10), Relu(10)), reduce_max()
)

# Now let's try it with a network that does work, just to be sure.
good_model = chain(Relu(10), Relu(10), Softmax())

# Finally we can reveal_type on the good model, to see what it thinks.
reveal_type(good_model)
