from typing import Tuple, Callable

from thinc.model import Model, OutT, InT
from thinc.types import Floats2d, Floats1d


def _do_whatever_backprop(derivative_y, features):
    return derivative_y


def forward(model: Model, features: Floats2d) -> Tuple[Floats1d, Callable[[OutT], InT]]:
    # do whatever computation. Ex:
    pred = model.ops.reshape3f(features, 1, 1, -1)

    def backward(derivative_y: OutT) -> InT:
        derivative_x: InT = _do_whatever_backprop(derivative_y, features)
        return derivative_x

    return pred, backward
