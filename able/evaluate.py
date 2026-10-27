"""Cross-validated ridge regression with honest uncertainty.

Every number this module reports comes from predictions on antibodies the
model never saw during fitting, including the choice of regularization
strength and the feature scaling.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-2, 5, 15)


def make_model() -> Pipeline:
    # The scaler sits inside the pipeline so its means and standard deviations
    # are learned from the training fold only, the same principle as per-fold
    # standardization in GraphBRAIN. RidgeCV picks alpha by leave-one-out
    # within the training fold, so alpha selection never sees the test fold
    # either: that is what makes the cross-validation nested.
    return Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=ALPHAS))])


def fold_ids(n: int, groups: np.ndarray | None, seed: int, n_splits: int = 5) -> np.ndarray:
    """Assign each sample a fold. With groups, a whole group lands in one fold."""
    folds = np.empty(n, dtype=int)
    if groups is None:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(n))
    else:
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
            np.zeros(n), groups=groups
        )
    for fold, (_, test) in enumerate(splitter):
        folds[test] = fold
    return folds


def oof_predict(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold predictions: each sample predicted by a model fit without it."""
    pred = np.empty(len(y))
    for fold in np.unique(folds):
        test = folds == fold
        model = make_model().fit(X[~test], y[~test])
        pred[test] = model.predict(X[test])
    return pred


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).statistic)


def _mean_rho(y: np.ndarray, preds: np.ndarray) -> float:
    """Spearman rho averaged over repeats; preds is (repeats, n) or (n,)."""
    preds = np.atleast_2d(preds)
    return float(np.mean([_spearman(y, p) for p in preds]))


def _bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def repeated_cv(X, y, groups, n_repeats: int = 10, n_boot: int = 2000) -> dict:
    """Repeat cross-validation with different fold assignments.

    The reported rho is the mean of the per-repeat Spearman correlations. The
    confidence interval bootstraps antibodies and recomputes that same
    quantity on each resample, so the interval and the point estimate measure
    the same thing. It keeps the fitted predictions fixed and resamples single
    antibodies, so it shows sampling noise in the evaluation set, not the full
    uncertainty of refitting on a new dataset.
    """
    y = np.asarray(y, dtype=float)
    oof = np.array([oof_predict(X, y, fold_ids(len(y), groups, seed)) for seed in range(n_repeats)])
    rhos = np.array([_spearman(y, p) for p in oof])
    boot = np.array([_mean_rho(y[i], oof[:, i]) for i in _bootstrap_indices(len(y), n_boot, seed=0)])
    return {
        "rho_per_repeat": rhos,
        "rho_mean": float(rhos.mean()),
        "rho_sd": float(rhos.std(ddof=1)) if n_repeats > 1 else 0.0,
        "oof": oof,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }


def permutation_pvalue(X, y, groups, observed: float, n_perm: int = 200, seed: int = 0) -> float:
    """How often shuffled labels score at least as well as the real ones.

    Shuffling breaks any real link between sequence and assay while keeping
    everything else (sample size, folds, model) the same.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    hits = 0
    for k in range(n_perm):
        shuffled = rng.permutation(y)
        pred = oof_predict(X, shuffled, fold_ids(len(y), groups, seed=k))
        hits += _spearman(shuffled, pred) >= observed
    return (1 + hits) / (1 + n_perm)


def paired_bootstrap(y, preds_a, preds_b, n_boot: int = 2000, seed: int = 0):
    """Compare two predictors on the same antibodies.

    preds_a and preds_b are out-of-fold predictions, (repeats, n) or (n,).
    Resampling the same antibodies for both keeps the comparison paired.
    Returns (rho_a - rho_b, ci_low, ci_high, p). The p-value inverts the
    bootstrap interval: twice the share of resampled differences on the
    other side of zero. With n_boot resamples it cannot be resolved below
    1/n_boot, so it is reported as at least that.
    """
    y = np.asarray(y, dtype=float)
    preds_a, preds_b = np.atleast_2d(preds_a), np.atleast_2d(preds_b)
    diffs = np.array(
        [_mean_rho(y[i], preds_a[:, i]) - _mean_rho(y[i], preds_b[:, i])
         for i in _bootstrap_indices(len(y), n_boot, seed)]
    )
    delta = _mean_rho(y, preds_a) - _mean_rho(y, preds_b)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p = min(1.0, max(p, 1 / n_boot))
    return delta, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(p)


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjustment for multiple comparisons."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()
