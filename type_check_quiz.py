from thinc.types import Floats3d, Ints1d

def numpy_shapes_pop_quiz(arr1: Floats3d, indices: Ints1d):
    # How many dimensions do each of these arrays have?
    q1 = arr1[0]
    q2 = arr1.mean()
    q3 = arr1[1:, 0]
    q4 = arr1[1:, :-1]
    q5 = arr1.sum(axis=0)
    q6 = arr1[1:, ..., :-1]
    q7 = arr1.sum(axis=(0, 1), keepdims=True)
    q8 = arr1[indices].cumsum()
    q9 = arr1[indices[indices]].ptp(axis=(-2, -1))
    # Run mypy over the snippet to find out your score!
    reveal_type(q1)
    reveal_type(q2)
    reveal_type(q3)
    reveal_type(q4)
    reveal_type(q5)
    reveal_type(q6)
    reveal_type(q7)
    reveal_type(q8)
    reveal_type(q9)