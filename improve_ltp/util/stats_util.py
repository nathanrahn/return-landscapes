import numpy as np

def purd_cvar(purd_samples: np.ndarray, cvar_target: float):
    quantiles_per_seed = np.quantile(purd_samples, cvar_target, axis=-1)
    cvars_per_seed = (quantiles_per_seed +
            np.mean(np.minimum(purd_samples - quantiles_per_seed[:, :, None], 0), axis=-1) / cvar_target)
    return cvars_per_seed

def purd_cvar_manystate(purd_samples: np.ndarray, cvar_target: float):
    # purd_samples: [ckpt, seed, sample idx, state idx]
    quantiles_per_seed = np.quantile(purd_samples, cvar_target, axis=2)
    cvars_per_seed = (quantiles_per_seed +
            np.mean(np.minimum(purd_samples - quantiles_per_seed[:, :, None, :], 0), axis=2) / cvar_target)
    return cvars_per_seed

def zero_normed_metric(window, upper_lim, bin_frac):
    return (window < upper_lim * bin_frac).sum() / len(window)

def get_perf_and_metric(returns, threshold=0.5, nbins=100):
    hist, bin_edges = np.histogram(returns, bins=nbins)
    big_bin = np.argmax(hist)
    mode = (bin_edges[big_bin] + bin_edges[big_bin + 1]) // 2

    metric = zero_normed_metric(returns, mode, threshold)

    return mode, metric
