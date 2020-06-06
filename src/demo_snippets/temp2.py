import numpy as np
from thinc.api import with_array, Linear

input_data = np.zeros((1, 1, 1), dtype="f")
model = with_array(Linear())
model.initialize(X=input_data)

from thinc.types import Ints2d, Array2d
from thinc.model import Model

input_data = np.zeros((1, 1, 1), dtype="f")
model2: Model[Array2d, Ints2d] = with_array(Linear())
model2.initialize(X=input_data)
