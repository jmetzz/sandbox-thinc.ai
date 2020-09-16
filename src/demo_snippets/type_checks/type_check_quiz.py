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
    reveal_type(q1)  # note: Revealed type is 'thinc.types.Floats2d'
    reveal_type(q2)  # error: "Floats3d" has no attribute "mean"
    reveal_type(q3)  # note: Revealed type is 'thinc.types.Floats2d'
    reveal_type(q4)  # note: Revealed type is 'thinc.types.Floats3d'
    reveal_type(q5)  # error: No overload variant of "sum" of "Floats3d" matches argument type "int"
    # note: Possible overload variants:
    # type_check_quiz.py:10: note:     def sum(self, *, keepdims: Literal[True], axis: Union[Tuple[int, int, int], Tuple[int, int], int, Tuple[int], None] = ..., out: Optional[Floats3d] = ...) -> Floats3d
    # type_check_quiz.py:10: note:     def sum(self, *, keepdims: Literal[False], axis: Union[int, Tuple[int]], out: Optional[Floats2d] = ...) -> Floats2d
    # type_check_quiz.py:10: note:     <2 more similar overloads not shown, out of 4 total overloads>
    reveal_type(q6)  # note: Revealed type is 'builtins.float'
    reveal_type(q7)  # note: Revealed type is 'thinc.types.Floats3d'
    reveal_type(q8)  # error: "Floats3d" has no attribute "cumsum"
    reveal_type(q9)  # error: "Floats3d" has no attribute "ptp"
