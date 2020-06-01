from thinc.api import Model, Linear, reduce_sum, chain

from thinc.types import Ragged, Floats2d

layer1: Model[Floats2d, Floats2d] = Linear(10, 10)
layer2: Model[Ragged, Floats2d] = reduce_sum()

model = chain(layer1, layer2)
