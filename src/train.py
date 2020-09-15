from typing import Tuple

from thinc.model import Model
from thinc.optimizers import Optimizer
from thinc.types import Floats2d, Floats1d
from tqdm import tqdm


def train_model(data: Tuple, model: Model, optimizer: Optimizer, n_iter: int, batch_size: int) -> None:
    (train_X, train_Y), (test_X, test_Y) = data
    model.initialize(X=train_X[:5], Y=train_Y[:5])
    for i in range(n_iter):
        loss = 0.0
        batches = model.ops.multibatch(batch_size, train_X, train_Y, shuffle=True)
        for X, Y in tqdm(batches, leave=False):
            Yh, backprop = model.begin_update(X)
            d_loss = Yh - Y
            loss += (d_loss ** 2).sum()
            backprop(d_loss)
            model.finish_update(optimizer)
        # Evaluate and print progress
        score = evaluate(model, test_X, test_Y, batch_size)
        print(f"{i}\t{loss:.2f}\t{score:.3f}")


def evaluate(model: Model, test_features: Floats2d, test_target: Floats1d, batch_size: int) -> float:
    correct = 0
    total = 0
    for X, Y in model.ops.multibatch(batch_size, test_features, test_target):
        prediction = model.predict(X)
        correct += (prediction.argmax(axis=1) == Y.argmax(axis=1)).sum()
        total += prediction.shape[0]
    score = correct / total
    return score
