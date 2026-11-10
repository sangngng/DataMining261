"""Unit tests for the metric (official definition: gain (y-1)/4, weight 1/log2(r+1), harmonic mean of Q and R)."""
import numpy as np
from src.evaluation import metric_fast
from src.data import str_rank
from src.metric import compute_competition_metric as ref


def run(y, p, ids=None):
    ids = ids if ids is not None else list(range(len(y)))
    return metric_fast(y, p, str_rank(ids))


def test_worked_example_q08_r06():
    assert abs(2 * 0.8 * 0.6 / 1.4 - 0.685714) < 1e-6


def test_empty_list_zero():
    assert run([1, 2, 3], [0, 0, 0]) == (0.0, 0.0, 0.0)


def test_split_without_worthwhile_papers_is_zero():
    assert run([1, 1, 1], [3, 2, 1])[0] == 0.0


def test_priority_zero_excluded_but_stays_in_coverage_denominator():
    assert abs(run([5, 5, 1], [3, 0, 2])[2] - 0.5) < 1e-12


def test_string_tiebreak_1000_before_999():
    s1 = run([5, 1], [1.0, 1.0], ids=[1000, 999]); s2 = run([1, 5], [1.0, 1.0], ids=[999, 1000])
    assert s1[1] == s2[1] and s1[1] > 0.4


def test_matches_reference_implementation():
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = 120; y = rng.integers(1, 6, n); p = rng.random(n) * (rng.random(n) > .3); ids = rng.permutation(5000)[:n]
        assert np.allclose(run(y, p, ids), ref(ids, y, p))


def test_perfect_order_has_q_below_one_when_labels_below_five():
    s, Q, R = run([5, 4, 3], [3, 2, 1]); assert Q < 1 and R == 1
