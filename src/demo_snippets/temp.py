import numpy as np
from thinc.api import Model, with_array, Linear
from thinc.types import Ints2d, Array2d, Floats2d

# @registry.layers("with_array.v1")
# def with_array(layer: Model[ValT, ValT], pad: int = 0) -> Model[SeqT, SeqT]:
#
# Thinc's types
# Array2d = Union["Floats2d", "Ints2d"]
# ValT = TypeVar("ValT", bound=Array2d)
# SeqT = TypeVar("SeqT", bound=Union[Padded, Ragged, List2d, Array2d])#

input_data: np.ndarray = np.zeros((1, 1, 1), dtype="f")
model: Model[Array2d, Ints2d] = with_array(Linear())
model.initialize(X=input_data)

# ragged array, a padded array, a 2d array or a list of 2d arrays
