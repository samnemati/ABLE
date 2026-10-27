import numpy as np
import pytest

from able.evaluate import fold_ids, holm, paired_bootstrap, permutation_pvalue, repeated_cv

RNG = np.random.default_rng(0)
N, P = 120, 50
X = RNG.normal(size=(N, P))
SIGNAL = X[:, 0] + 0.1 * RNG.normal(size=N)
NOISE = RNG.normal(size=N)
GROUPS = np.repeat(np.arange(30), 4)


def test_signal_is_recovered():
    assert repeated_cv(X, SIGNAL, None, n_repeats=3)["rho_mean"] > 0.8


def test_noise_gives_no_positive_signal():
    # If scaling or alpha selection saw the test fold, noise would look
    # predictable (positive rho). It must not. On pure noise, cross-validation
    # is known to give slightly NEGATIVE rho: a model that learns nothing
    # predicts about the training-fold mean, and holding out a high value
    # lowers that mean. So we bound rho from above, not around zero.
    for groups in (None, GROUPS):
        assert repeated_cv(X, NOISE, groups, n_repeats=3)["rho_mean"] < 0.1


def test_groups_never_split():
    folds = fold_ids(N, GROUPS, seed=0)
    for fold in np.unique(folds):
        test_groups = set(GROUPS[folds == fold])
        train_groups = set(GROUPS[folds != fold])
        assert not test_groups & train_groups


def test_permutation_pvalue_small_for_signal_large_for_noise():
    observed_signal = repeated_cv(X, SIGNAL, GROUPS, n_repeats=1)["rho_mean"]
    observed_noise = repeated_cv(X, NOISE, GROUPS, n_repeats=1)["rho_mean"]
    assert permutation_pvalue(X, SIGNAL, GROUPS, observed_signal, n_perm=50) <= 0.05
    assert permutation_pvalue(X, NOISE, GROUPS, observed_noise, n_perm=50) > 0.1


def test_holm_known_values():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_paired_bootstrap_detects_better_predictor():
    good = SIGNAL + 0.1 * RNG.normal(size=N)
    bad = RNG.normal(size=N)
    delta, ci_low, ci_high, p = paired_bootstrap(SIGNAL, good, bad)
    assert delta > 0
    assert ci_low > 0
    assert p < 0.05


def test_ci_brackets_reported_rho():
    result = repeated_cv(X, SIGNAL + RNG.normal(size=N), GROUPS, n_repeats=5)
    assert result["ci_low"] <= result["rho_mean"] <= result["ci_high"]


def test_paired_delta_matches_reported_rhos():
    y = SIGNAL + RNG.normal(size=N)
    a = repeated_cv(X, y, GROUPS, n_repeats=3)
    b = repeated_cv(X[:, 1:], y, GROUPS, n_repeats=3)
    delta, _, _, _ = paired_bootstrap(y, a["oof"], b["oof"])
    assert delta == pytest.approx(a["rho_mean"] - b["rho_mean"])


def test_paired_p_never_reported_as_zero():
    good = SIGNAL + 0.01 * RNG.normal(size=N)
    _, _, _, p = paired_bootstrap(SIGNAL, good, RNG.normal(size=N), n_boot=500)
    assert p == pytest.approx(1 / 500)
