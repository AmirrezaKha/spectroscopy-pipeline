import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

from preprocessing import apply_savgol_filter, standard_normal_variate, baseline_correction
from anomaly_detection import detect_isolation_forest, detect_mahalanobis, detect_z_score_outliers
from model import train_autoencoder


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads FTIR component spectra and a mixture spectrum.

    Returns
    -------
    tuple
        components: (3, n_features)
        mixture: (n_features,)
        all_spectra: (4, n_features)
    """
    components = np.load("data/components.npy")
    mixture = np.load("data/mixture_spectrum.npy")
    all_spectra = np.vstack([components, mixture])
    return components, mixture, all_spectra


def preprocess_spectra(spectra: np.ndarray) -> np.ndarray:
    """
    Applies a series of preprocessing steps to FTIR spectra.

    Parameters
    ----------
    spectra : np.ndarray
        Raw spectra data.

    Returns
    -------
    np.ndarray
        Preprocessed spectra.
    """
    spectra = apply_savgol_filter(spectra, deriv=1)
    spectra = standard_normal_variate(spectra)
    spectra = baseline_correction(spectra)
    return spectra


def estimate_concentrations(components: np.ndarray, mixture: np.ndarray) -> np.ndarray:
    """
    Solves the least squares problem to estimate component concentrations.

    Parameters
    ----------
    components : np.ndarray
        Known component spectra (3, n_features).
    mixture : np.ndarray
        Mixture spectrum (n_features,).

    Returns
    -------
    np.ndarray
        Estimated concentrations.
    """
    concentrations, *_ = lstsq(components.T, mixture, rcond=None)
    return concentrations


def plot_spectra(mixture: np.ndarray, components: np.ndarray):
    """
    Plots the mixture and component spectra.

    Parameters
    ----------
    mixture : np.ndarray
        Mixture spectrum.
    components : np.ndarray
        Array of individual component spectra.
    """
    plt.plot(mixture, label="Mixture", linewidth=2)
    for i, comp in enumerate(components):
        plt.plot(comp, label=f"Component {chr(65 + i)}")
    plt.legend()
    plt.title("FTIR Spectra (Mixture vs Components)")
    plt.xlabel("Wavenumber Index")
    plt.ylabel("Absorbance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # Load and preprocess data
    components, mixture, all_spectra = load_data()
    spectra = preprocess_spectra(all_spectra)

    # Anomaly detection
    iso_outliers = detect_isolation_forest(spectra)
    maha_outliers = detect_mahalanobis(spectra)
    z_outliers = detect_z_score_outliers(spectra)

    print("Isolation Forest anomalies:", iso_outliers)
    print("Mahalanobis anomalies:", maha_outliers)
    print("Z-score anomalies:", z_outliers)

    # Optional cleaning step: remove Isolation Forest outliers
    valid_indices = list(set(range(spectra.shape[0])) - set(iso_outliers))
    clean_spectra = spectra[valid_indices]

    # Optional: Train autoencoder on clean spectra
    # model = train_autoencoder(clean_spectra)

    # Estimate concentrations using least squares
    concentrations = estimate_concentrations(components, mixture)
    print("Estimated Concentrations [A, B, C]:", concentrations.round(1))

    # Plot spectra
    plot_spectra(mixture, components)


if __name__ == "__main__":
    main()
