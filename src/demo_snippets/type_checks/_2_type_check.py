from thinc.api import Model, Linear, reduce_sum, chain
from thinc.types import Array2d, Array3d

layer1: Model[Array2d, Array2d] = Linear(10, 10)
layer2: Model[Array3d, Array2d] = reduce_sum()

model = chain(layer1, layer2)
