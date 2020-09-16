from thinc.api import Model
from typing import Tuple, Any, Callable
import random

def random_chain(child_layer1: Model, child_layer2: Model, prob: float = 0.2) -> Model:
    """Randomly invert the order of two layers during training."""

    def random_chain_forward(model: Model, X, is_train: bool):
        child_layer1 = model.layers[0]
        child_layer2 = model.layers[1]
        prob = model.get_attr("prob")
        is_reversed = is_train and prob >= random.random()
        if is_reversed:
            Y, get_dX = child_layer2(X, is_train)
            Z, get_dY = child_layer1(Y, is_train)
        else:
            Y, get_dX = child_layer1(X, is_train)
            Z, get_dY = child_layer2(Y, is_train)

        def backprop(dZ):
            dY = get_dY(dZ)
            dX = get_dX(dY)
            return dX

        return Z, backprop

    return Model(
        "random_order",
        random_chain_forward,
        init=init,
        attrs={"prob": prob},
        layers=[child_layer1, child_layer2],
    )

def random_chain_init(model, X=None, Y=None):
    if X is not None and model.has_dim("nI") is None:
        model.set_dim("nI", X.shape[1])
    if Y is not None and model.has_dim("nO") is None:
        model.set_dim("nO", Y.shape[1])
    for child_layer in model.layers:
        child_layer.initialize(X=X, Y=Y)
        

def times_two(layer: Model, init=None) -> Model:
    """Randomly invert the order of two layers during training."""
    return Model(
        "times_two",
        multiply_by_two_forward,
        init=init,
        attrs=None,
        layers=[layer],
    )

def multiply_by_two_forward(model: Model, X, is_train)-> Tuple[Any, Callable]:
    Y = X * 2
    W = model.get_param("W")
    def backprop(dY):
        model.inc_grad("b", dY.sum(axis=0))
        model.inc_grad("W", dY.T @ X)
        return dY @ W

    return Y, backprop