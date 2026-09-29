"""KID estimator properties (UPGRADES #58, increment_campaign.md §2).

These pin the properties that make KID usable where FID is not: unbiasedness at
small n, and a standard error that says whether two runs are resolvable.

No TensorFlow here -- the estimator is pure numpy, so its correctness is
testable without touching InceptionV3.
"""

import numpy as np
import pytest

from snowgan.kid import kid_score, polynomial_mmd2_unbiased


def _gaussian(n, d, loc=0.0, scale=1.0, seed=0):
    return np.random.default_rng(seed).normal(loc, scale, size=(n, d))


def test_identical_distributions_score_near_zero():
    """The defining property: same distribution -> MMD^2 ~ 0."""
    x = _gaussian(200, 64, seed=1)
    y = _gaussian(200, 64, seed=2)
    assert abs(polynomial_mmd2_unbiased(x, y)) < 0.05


def test_separated_distributions_score_clearly_positive():
    x = _gaussian(200, 64, loc=0.0, seed=1)
    y = _gaussian(200, 64, loc=3.0, seed=2)
    assert polynomial_mmd2_unbiased(x, y) > 1.0


def test_score_grows_with_separation():
    x = _gaussian(200, 64, seed=1)
    scores = [polynomial_mmd2_unbiased(x, _gaussian(200, 64, loc=shift, seed=2))
              for shift in (0.0, 0.5, 1.0, 2.0)]
    assert scores == sorted(scores), f"not monotone in separation: {scores}"


def test_estimator_is_unbiased_across_sample_sizes():
    """The reason KID replaces FID here.

    FID's bias depends on n, so two runs scored with different sample counts are
    not comparable. An unbiased estimator's expectation must not drift with n.
    """
    means = []
    for n in (25, 50, 100, 200):
        vals = [polynomial_mmd2_unbiased(_gaussian(n, 64, seed=s),
                                         _gaussian(n, 64, seed=s + 500))
                for s in range(12)]
        means.append(float(np.mean(vals)))
    # Every sample size must agree on ~0; no systematic drift with n.
    assert max(abs(m) for m in means) < 0.05, f"bias drifts with n: {means}"


def test_unbiased_estimator_may_go_negative():
    """Expected, not a bug: clipping to zero would restore the bias the
    unbiased estimator exists to remove."""
    negatives = [polynomial_mmd2_unbiased(_gaussian(40, 64, seed=s),
                                          _gaussian(40, 64, seed=s + 900))
                 for s in range(30)]
    assert any(v < 0 for v in negatives), "no negative estimates across 30 trials"


def test_kid_score_reports_a_usable_standard_error():
    x = _gaussian(400, 64, seed=1)
    y = _gaussian(400, 64, loc=1.0, seed=2)
    result = kid_score(x, y, subsets=10, subset_size=100)

    assert result["subsets"] == 10 and result["subset_size"] == 100
    assert result["kid_se"] > 0
    # SE is the spread of the mean, so it must be smaller than the spread of
    # individual subset estimates.
    assert result["kid_se"] < result["kid_std"]


def test_kid_resolves_a_difference_its_error_bars_support():
    """A metric is only useful if the gap it reports exceeds its own noise."""
    real = _gaussian(400, 64, seed=1)
    near = _gaussian(400, 64, loc=0.0, seed=2)
    far = _gaussian(400, 64, loc=1.5, seed=3)

    a = kid_score(real, near, subsets=10, subset_size=100)
    b = kid_score(real, far, subsets=10, subset_size=100)

    gap = b["kid_mean"] - a["kid_mean"]
    pooled_se = np.sqrt(a["kid_se"] ** 2 + b["kid_se"] ** 2)
    assert gap > 2 * pooled_se, f"gap {gap:.4f} not resolvable at 2 SE {2*pooled_se:.4f}"


def test_subset_size_is_capped_by_the_smaller_corpus():
    """We have 344 held-out images; asking for 1000 must not explode."""
    result = kid_score(_gaussian(50, 32, seed=1), _gaussian(400, 32, seed=2),
                       subsets=4, subset_size=1000)
    assert result["subset_size"] == 50


def test_too_few_samples_raises():
    with pytest.raises(ValueError):
        polynomial_mmd2_unbiased(_gaussian(1, 8), _gaussian(5, 8))
