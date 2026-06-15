import typing

import numpy as np
import numpy.typing as npt
import pytest

from cmenpy.decoder import (
    BaseSpaceDecoder,
    FloatSpaceDecoder,
    IntegerSpaceDecoder,
    StringSpaceDecoder,
    CategoricalSpaceDecoder,
    SequenceSpaceDecoder,
    PermutationSpaceDecoder,
    BinarySpaceDecoder,
    BoolSpaceDecoder,
    as_bound_pair,
    as_label_set,
    as_label_sets,
)


class TestHelpers:
    def test_bound_pair_scalar(self):
        lb, ub = as_bound_pair(-5, 5)
        assert lb.tolist() == [-5.0] and ub.tolist() == [5.0]

    def test_bound_pair_sequence(self):
        lb, ub = as_bound_pair([0.0, 1.0], [2.0, 3.0])
        assert lb.tolist() == [0.0, 1.0] and ub.tolist() == [2.0, 3.0]

    def test_bound_pair_length_mismatch(self):
        with pytest.raises(ValueError):
            as_bound_pair([0.0, 1.0], [2.0])

    def test_bound_pair_mixed_types(self):
        with pytest.raises(TypeError):
            as_bound_pair(0.0, [2.0, 3.0])

    def test_label_set_sorted(self):
        arr, table = as_label_set(["c", "a", "b"])
        assert arr.tolist() == ["a", "b", "c"]
        assert table == {"a": 0, "b": 1, "c": 2}

    def test_label_sets_flat(self):
        labels, index = as_label_sets(("relu", "tanh"))
        assert len(labels) == 1 and labels[0].tolist() == ["relu", "tanh"]

    def test_label_sets_rejects_singleton_group(self):
        with pytest.raises(ValueError):
            as_label_sets((("a", "b"), ("only",)))


class TestFloatSpace:
    def test_bounds(self):
        sp = FloatSpaceDecoder(lb=[-5.0, 0.0], ub=[5.0, 1.0])
        assert sp.n_vars == 2
        assert sp.lb.tolist() == [-5.0, 0.0]
        assert sp.ub.tolist() == [5.0, 1.0]

    def test_scalar_bounds(self):
        sp = FloatSpaceDecoder(lb=-3.0, ub=3.0)
        assert sp.n_vars == 1

    def test_clip_high_and_low(self):
        sp = FloatSpaceDecoder(lb=[-5.0, 0.0], ub=[5.0, 1.0])
        np.testing.assert_array_equal(sp.decode(np.array([9.9, -3.0])), [5.0, 0.0])

    def test_generate_in_bounds(self):
        sp = FloatSpaceDecoder(lb=0.0, ub=1.0)
        sp.set_seed(0)
        for _ in range(100):
            v = sp.generate()
            assert (v >= sp.lb).all() and (v <= sp.ub).all()

    def test_encode_identity(self):
        sp = FloatSpaceDecoder(lb=0.0, ub=10.0)
        np.testing.assert_array_equal(sp.encode([3.5]), [3.5])


class TestIntegerSpace:
    def test_shifted_bounds(self):
        sp = IntegerSpaceDecoder(lb=16, ub=256)
        assert sp.lb[0] == pytest.approx(15.5)
        assert sp.ub[0] == pytest.approx(256.5 - sp.EPS)

    def test_round_half_up(self):
        sp = IntegerSpaceDecoder(lb=0, ub=1000)
        assert sp.decode(np.array([128.4]))[0] == 128
        assert sp.decode(np.array([128.6]))[0] == 129
        assert sp.decode(np.array([128.5]))[0] == 129

    def test_clip(self):
        sp = IntegerSpaceDecoder(lb=16, ub=256)
        assert sp.decode(np.array([9999.0]))[0] == 256
        assert sp.decode(np.array([-9999.0]))[0] == 16

    def test_generate_range(self):
        sp = IntegerSpaceDecoder(lb=1, ub=6)
        sp.set_seed(0)
        seen = {int(sp.generate()[0]) for _ in range(500)}
        assert seen == {1, 2, 3, 4, 5, 6, 7}


class TestStringSpace:
    def test_alphabetical_index(self):
        sp = StringSpaceDecoder(valid_sets=("relu", "tanh", "elu"))
        assert sp.labels[0].tolist() == ["elu", "relu", "tanh"]

    def test_multivar(self):
        sp = StringSpaceDecoder(
            valid_sets=(("relu", "tanh", "sigmoid"), ("adam", "sgd"))
        )
        assert sp.n_vars == 2
        assert sp.decode(np.array([2.3, 1.8])) == ["tanh", "sgd"]

    def test_encode_decode_roundtrip(self):
        sp = StringSpaceDecoder(
            valid_sets=(("relu", "tanh", "sigmoid"), ("adam", "sgd"))
        )
        encoded = sp.encode(["sigmoid", "sgd"])
        assert sp.decode(encoded) == ["sigmoid", "sgd"]

    def test_truncation_boundary(self):
        sp = StringSpaceDecoder(valid_sets=("a", "b", "c"))
        assert sp.decode(np.array([0.99]))[0] == "a"
        assert sp.decode(np.array([1.0]))[0] == "b"

    def test_ub_never_overflows(self):
        sp = StringSpaceDecoder(valid_sets=("a", "b", "c", "d"))
        assert sp.decode(sp.ub)[0] == "d"


class TestCategoricalSpace:
    def test_mixed_types(self):
        sp = CategoricalSpaceDecoder(valid_sets=((8, 16, 32), ("l1", "l2", "none")))
        assert sp.decode(np.array([2.1, 0.9])) == [8, "l1"]

    def test_preserves_int_type(self):
        sp = CategoricalSpaceDecoder(valid_sets=((8, 16, 32),))
        assert sp.labels[0].tolist() == [16, 32, 8]
        result = sp.decode(np.array([1.0]))
        assert result[0] == 32
        assert isinstance(result[0], int)


class TestSequenceSpace:
    def test_pick_whole_sequence(self):
        sp = SequenceSpaceDecoder(valid_sets=([1, 2, 3], [10, 20], [5, 5, 5, 5]))
        assert sp.decode(np.array([0.4])) == (1, 2, 3)
        assert sp.decode(np.array([2.9])) == (5, 5, 5, 5)

    def test_return_type_list(self):
        sp = SequenceSpaceDecoder(valid_sets=([1, 2, 3], [10, 20]), return_type=list)
        out = sp.decode(np.array([0.0]))
        assert out == [1, 2, 3] and isinstance(out, list)

    def test_return_type_tuple_default(self):
        sp = SequenceSpaceDecoder(valid_sets=([1, 2, 3], [10, 20]))
        assert isinstance(sp.decode(np.array([1.5])), tuple)

    def test_encode_roundtrip(self):
        sp = SequenceSpaceDecoder(valid_sets=([1, 2, 3], [10, 20]))
        enc = sp.encode([10, 20])
        assert sp.decode(enc) == (10, 20)


class TestPermutationSpace:
    def test_validity_arbitrary_floats(self):
        sp = PermutationSpaceDecoder(valid_set=["A", "B", "C", "D"])
        out = sp.decode(np.array([5.5, -2.0, 100.0, 0.0]))
        assert sorted(out.tolist()) == ["A", "B", "C", "D"]

    def test_argsort_order(self):
        sp = PermutationSpaceDecoder(valid_set=["A", "B", "C", "D"])
        out = sp.decode(np.array([0.9, 0.1, 0.7, 0.3]))
        assert out.tolist() == ["B", "D", "C", "A"]

    def test_generate_is_permutation(self):
        sp = PermutationSpaceDecoder(valid_set=[1, 2, 3, 4, 5])
        sp.set_seed(0)
        out = sp.generate()
        assert sorted(out.tolist()) == [1, 2, 3, 4, 5]

    def test_rejects_singleton(self):
        with pytest.raises(ValueError):
            PermutationSpaceDecoder(valid_set=["only"])

    def test_encode_maps_to_sorted_indices(self):
        sp = PermutationSpaceDecoder(valid_set=["A", "B", "C"])
        np.testing.assert_array_equal(sp.encode(["B", "C", "A"]), [1.0, 2.0, 0.0])

    def test_decode_always_valid_permutation(self):
        sp = PermutationSpaceDecoder(valid_set=["A", "B", "C"])
        enc = sp.encode(["B", "C", "A"])
        assert sorted(sp.decode(enc).tolist()) == ["A", "B", "C"]

    def test_identity_input_recovers_label_order(self):
        sp = PermutationSpaceDecoder(valid_set=["A", "B", "C"])
        np.testing.assert_array_equal(
            sp.decode(np.array([0.0, 1.0, 2.0])), ["A", "B", "C"]
        )


class TestBinarySpace:
    def test_truncation(self):
        sp = BinarySpaceDecoder(n_vars=6)
        out = sp.decode(np.array([0.0, 0.5, 0.99, 1.0, 1.5, 9.0]))
        np.testing.assert_array_equal(out, [0, 0, 0, 1, 1, 1])

    def test_generate_only_binary(self):
        sp = BinarySpaceDecoder(n_vars=10)
        sp.set_seed(0)
        out = sp.generate()
        assert set(out.tolist()) <= {0, 1}

    def test_rejects_zero_vars(self):
        with pytest.raises(ValueError):
            BinarySpaceDecoder(n_vars=0)


class TestBoolSpace:
    def test_dtype_is_bool(self):
        sp = BoolSpaceDecoder(n_vars=4)
        out = sp.decode(np.array([0.2, 1.1, 0.99, 1.8]))
        assert out.dtype == np.bool_
        np.testing.assert_array_equal(out, [False, True, False, True])

    def test_generate_bool(self):
        sp = BoolSpaceDecoder(n_vars=5)
        sp.set_seed(0)
        assert sp.generate().dtype == np.bool_


ALL_SPACES = [
    lambda: FloatSpaceDecoder(lb=0.0, ub=1.0),
    lambda: IntegerSpaceDecoder(lb=1, ub=10),
    lambda: StringSpaceDecoder(valid_sets=("a", "b", "c")),
    lambda: CategoricalSpaceDecoder(valid_sets=((1, 2, 3),)),
    lambda: SequenceSpaceDecoder(valid_sets=([1, 2], [3, 4])),
    lambda: PermutationSpaceDecoder(valid_set=["a", "b", "c"]),
    lambda: BinarySpaceDecoder(n_vars=4),
    lambda: BoolSpaceDecoder(n_vars=4),
]


class TestReproducibility:
    @pytest.mark.parametrize("factory", ALL_SPACES)
    def test_seed_determinism(self, factory):
        sp1 = factory()
        sp1.set_seed(123)
        r1 = sp1.generate()
        sp2 = factory()
        sp2.set_seed(123)
        r2 = sp2.generate()
        np.testing.assert_array_equal(np.asarray(r1), np.asarray(r2))

    @pytest.mark.parametrize("factory", ALL_SPACES)
    def test_different_seeds_differ_eventually(self, factory):
        sp = factory()
        sp.set_seed(1)
        a = [np.asarray(sp.generate()).tolist() for _ in range(20)]
        sp.set_seed(2)
        b = [np.asarray(sp.generate()).tolist() for _ in range(20)]
        assert a != b


class TestContract:
    @pytest.mark.parametrize("factory", ALL_SPACES)
    def test_bounds_shape(self, factory):
        sp = factory()
        assert sp.lb.shape == sp.ub.shape == (sp.n_vars,)
        assert (sp.lb <= sp.ub).all()

    @pytest.mark.parametrize("factory", ALL_SPACES)
    def test_decode_accepts_lb_and_ub(self, factory):
        sp = factory()
        sp.decode(sp.lb.copy())
        sp.decode(sp.ub.copy())
