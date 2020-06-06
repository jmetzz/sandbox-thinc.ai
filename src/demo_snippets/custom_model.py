from thinc.api import Model


def random_chain(child_layer1: Model, child_layer2: Model, prob: float = 0.2) -> Model:
    """Randomly invert the order of two layers during training."""
    return Model(
        "random_order",
        random_chain_forward,
        init=init,
        attrs={"prob": prob},
        layers=[child_layer1, child_layer2],
    )

def times_two(layer: Model) -> Model:
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

    def backprop(dY):
        model.inc_grad("b", dY.sum(axis=0))
        model.inc_grad("W", dY.T @ X)
        return dY @ W

    return Y, backprop