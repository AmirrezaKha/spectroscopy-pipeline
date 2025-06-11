import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import mahalanobis
from scipy.stats import zscore
from numpy.linalg import inv

def detect_isolation_forest(spectra, contamination=0.1, random_state=42):
    """
    Detects anomalies in spectral data using the Isolation Forest algorithm.

    Parameters:
    -----------
    spectra : np.ndarray
        A 2D NumPy array where each row represents one FTIR spectrum sample.
    
    contamination : float, optional (default=0.1)
        The proportion of anomalies in the data. Should be between 0.0 and 0.5.
    
    random_state : int, optional (default=42)
        Seed used by the random number generator for reproducibility.

    Returns:
    --------
    np.ndarray
        Array of indices corresponding to the detected anomalies in the input spectra.
    
    Notes:
    ------
    - Isolation Forest works well for high-dimensional data and is effective
      for unsupervised anomaly detection.
    - The detected outliers are those labeled as -1 by the model.
    """
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    preds = clf.fit_predict(spectra)
    return np.where(preds == -1)[0]

def detect_mahalanobis(X, threshold=3.0):
    """
    Detects anomalies in spectral data using Mahalanobis distance.

    Parameters
    ----------
    X : np.ndarray
        2D input array with shape (n_samples, n_features), where each row is a spectrum.
    
    threshold : float, optional (default=3.0)
        Threshold value for the Mahalanobis distance. Samples with distance
        greater than this threshold are considered outliers.

    Returns
    -------
    np.ndarray
        Indices of samples classified as anomalies based on Mahalanobis distance.

    Notes
    -----
    - Mahalanobis distance accounts for correlations between variables and
      is scale-invariant.
    - Assumes the input data is roughly normally distributed.
    - A threshold of ~3 is typically used for 99.7% confidence under normal distribution.
    """
    X_mean = np.mean(X, axis=0)
    X_cov_inv = inv(np.cov(X, rowvar=False))

    distances = []
    for row in X:
        d = mahalanobis(row, X_mean, X_cov_inv)
        distances.append(d)

    distances = np.array(distances)
    return np.where(distances > threshold)[0]

def detect_z_score_outliers(spectra, z_thresh=3):
    """
    Detects anomalies in spectral data using Z-score thresholding.

    Parameters
    ----------
    spectra : np.ndarray
        A 2D array of shape (n_samples, n_features), where each row is a spectrum.
    
    z_thresh : float, optional (default=3)
        Z-score threshold. Samples with any feature having an absolute z-score
        greater than this threshold are considered outliers.

    Returns
    -------
    np.ndarray
        Array of unique indices corresponding to outlier spectra.

    Notes
    -----
    - Z-score assumes the features follow a normal distribution.
    - A threshold of 3 corresponds to ~99.7% confidence under normal distribution.
    - This method flags a sample as an outlier if **any** of its features exceed the threshold.
    """
    z_scores = np.abs(zscore(spectra, axis=0))
    return np.unique(np.where(z_scores > z_thresh)[0])