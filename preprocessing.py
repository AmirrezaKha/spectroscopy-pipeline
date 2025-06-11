import numpy as np
from scipy.signal import savgol_filter

import numpy as np
from scipy.signal import savgol_filter

def apply_savgol_filter(
    spectra: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2,
    deriv: int = 0
) -> np.ndarray:
    """
    Applies Savitzky-Golay filter for smoothing or differentiation.

    Parameters
    ----------
    spectra : np.ndarray
        2D array of shape (samples x features), FTIR spectra data.
    window_length : int
        Length of the filter window (must be odd and >= polyorder + 2).
    polyorder : int
        Polynomial order to use in the filter.
    deriv : int
        Derivative order to compute (0 for smoothing).

    Returns
    -------
    np.ndarray
        Filtered spectra of the same shape.
    """
    return savgol_filter(
        spectra,
        window_length=window_length,
        polyorder=polyorder,
        deriv=deriv,
        axis=-1
    )


def standard_normal_variate(spectra: np.ndarray) -> np.ndarray:
    """
    Applies Standard Normal Variate (SNV) normalization to each spectrum.

    Parameters
    ----------
    spectra : np.ndarray
        2D array (samples x features).

    Returns
    -------
    np.ndarray
        SNV-normalized spectra.
    """
    mean = np.mean(spectra, axis=1, keepdims=True)
    std = np.std(spectra, axis=1, keepdims=True)
    return (spectra - mean) / std


def baseline_correction(spectra: np.ndarray) -> np.ndarray:
    """
    Applies a simple baseline correction by subtracting the minimum value
    in each spectrum.

    Parameters
    ----------
    spectra : np.ndarray
        2D array (samples x features).

    Returns
    -------
    np.ndarray
        Baseline-corrected spectra.
    """
    baseline = np.min(spectra, axis=1, keepdims=True)
    return spectra - baseline
