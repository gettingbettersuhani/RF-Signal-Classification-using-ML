"""
Production Feature Extraction Module
RF Signal Classification
"""
from scipy.signal import welch
import numpy as np
from scipy.stats import kurtosis

import src.data.preprocessing as prep


EPSILON = 1e-12


def _haar_detail_coeffs(signal, levels=3):
    """
    Compute the final Haar detail coefficients without requiring PyWavelets.
    """

    coeffs = np.asarray(np.real(signal), dtype=float)

    if coeffs.size < 2:
        return coeffs

    for _ in range(levels):
        if coeffs.size < 2:
            break

        if coeffs.size % 2 == 1:
            coeffs = coeffs[:-1]

        coeffs = (coeffs[::2] - coeffs[1::2]) / np.sqrt(2.0)

        if coeffs.size == 0:
            break

    return coeffs

def get_complex_signal(iq_signal):
    """
    Convert IQ samples to complex baseband signal.
    """

    I, Q = prep.split_iq(iq_signal)

    return I + 1j * Q

def safe_divide(a, b):
    """
    Safe division.
    """

    return a / (b + EPSILON)


def normalize(signal):
    """
    Zero mean, unit variance normalization.
    """

    signal = signal - np.mean(signal)

    std = np.std(signal)

    if std < EPSILON:
        return signal

    return signal / std



def instantaneous_amplitude(signal):

    return np.abs(signal)



def instantaneous_phase(signal):

    return np.unwrap(np.angle(signal))

def instantaneous_frequency(signal):

    phase = instantaneous_phase(signal)

    return np.diff(phase)

def compute_moments(signal):
    """
    Compute complex moments.
    """

    r = signal
    rc = np.conj(r)

    moments = {}

    moments["M20"] = np.mean(r**2)
    moments["M21"] = np.mean(r * rc)
    moments["M22"] = np.mean(rc**2)

    moments["M40"] = np.mean(r**4)
    moments["M41"] = np.mean((r**3) * rc)
    moments["M42"] = np.mean((r**2) * (rc**2))
    moments["M43"] = np.mean(r * (rc**3))

    moments["M60"] = np.mean(r**6)
    moments["M61"] = np.mean((r**5) * rc)
    moments["M62"] = np.mean((r**4) * (rc**2))
    moments["M63"] = np.mean((r**3) * (rc**3))

    moments["M80"] = np.mean(r**8)
    moments["M84"] = np.mean((r**4) * (rc**4))

    return moments



def compute_cumulants(m):
    """
    Compute higher-order cumulants.
    """

    c = {}

    c["C20"] = m["M20"]

    c["C21"] = m["M21"]

    c["C40"] = m["M40"] - 3 * (m["M20"] ** 2)

    c["C41"] = m["M41"] - 3 * m["M20"] * m["M21"]

    c["C42"] = (
        m["M42"]
        - abs(m["M20"]) ** 2
        - 2 * (m["M21"] ** 2)
    )

    c["C60"] = (
        m["M60"]
        - 15 * m["M40"] * m["M20"]
        + 30 * (m["M20"] ** 3)
    )

    c["C61"] = (
        m["M61"]
        - 5 * m["M41"] * m["M20"]
        - 10 * m["M40"] * m["M21"]
        + 30 * (m["M20"] ** 2) * m["M21"]
    )

    c["C62"] = (
        m["M62"]
        - 6 * m["M42"] * m["M20"]
        - 8 * m["M41"] * m["M21"]
        + 6 * abs(m["M20"]) ** 2 * m["M20"]
        + 24 * (m["M21"] ** 2) * m["M20"]
    )

    c["C63"] = (
        m["M63"]
        - 9 * m["M42"] * m["M21"]
        + 12 * (m["M21"] ** 3)
    )

    c["C80"] = m["M80"]

    c["C84"] = m["M84"]

    return c



def compute_cumulant_ratios(c):
    """
    Normalized cumulant ratios.
    """

    ratios = {}

    ratios["R1"] = safe_divide(abs(c["C40"]), abs(c["C42"]))

    ratios["R2"] = safe_divide(abs(c["C41"]), abs(c["C42"]))

    ratios["R3"] = safe_divide(
        abs(c["C42"]),
        abs(c["C21"]) ** 2
    )

    ratios["R4"] = safe_divide(
        abs(c["C60"]),
        abs(c["C21"]) ** 3
    )

    ratios["R5"] = safe_divide(
        abs(c["C63"]),
        abs(c["C21"]) ** 3
    )

    ratios["R8"] = safe_divide(
        abs(c["C80"]),
        abs(c["C21"]) ** 2
    )

    ratios["R9"] = safe_divide(
        abs(c["C84"]),
        abs(c["C21"]) ** 2
    )

    return ratios

def compute_instantaneous_features(signal):
    """
    Instantaneous amplitude, phase and frequency features.
    """

    amp = instantaneous_amplitude(signal)

    phase = instantaneous_phase(signal)

    freq = instantaneous_frequency(signal)

    features = {}

    amp_norm = normalize(amp)

    phase_norm = normalize(phase)

    freq_norm = normalize(freq)

    features["AmplitudeMean"] = np.mean(amp)

    features["AmplitudeStd"] = np.std(amp)

    features["AmplitudeKurtosis"] = np.nan_to_num(
      kurtosis(
          amp,
          fisher=False,
          bias=False
      )
    )

    features["PhaseMean"] = np.mean(phase)

    features["PhaseStd"] = np.std(phase)

    features["FrequencyMean"] = np.mean(freq)

    features["FrequencyStd"] = np.std(freq)

    features["SigmaAA"] = np.std(amp_norm)

    features["SigmaAP"] = np.std(phase_norm)

    features["SigmaAF"] = np.std(freq_norm)

    return features



def compute_psd_features(signal):
    """
    Power Spectral Density Features
    """

    amp = instantaneous_amplitude(signal)

    amp = normalize(amp)

    freqs, psd = welch(
        amp,
        nperseg=min(256, len(amp))
    )

    psd = psd + EPSILON

    features = {}

    features["PSDMean"] = np.mean(psd)

    features["PSDMax"] = np.max(psd)

    features["PSDStd"] = np.std(psd)

    features["GammaMean"] = np.mean(psd)

    features["GammaMax"] = np.max(psd)

    features["SpectralVariance"] = np.var(psd)

    return features



def spectral_symmetry(signal):
    """
    Spectral symmetry parameter.
    """

    spectrum = np.abs(np.fft.fft(signal))

    half = len(spectrum) // 2

    left = spectrum[:half]

    right = spectrum[-half:]

    length = min(len(left), len(right))

    left = left[:length]

    right = right[:length]

    value = np.mean(
        np.abs(left - right[::-1])
    )

    return value



def zero_crossing_rate(x):
    """
    Zero Crossing Rate
    """

    return np.mean(
        np.abs(np.diff(np.sign(x)))
    ) / 2



def compute_zero_crossing_features(signal):

    I = np.real(signal)

    Q = np.imag(signal)

    freq = instantaneous_frequency(signal)

    features = {}

    features["IZCR"] = zero_crossing_rate(I)

    features["QZCR"] = zero_crossing_rate(Q)

    features["FrequencyZCR"] = zero_crossing_rate(freq)

    return features

def wavelet_template_features(signal):
    """
    Haar Wavelet Template Correlation Features
    """

    detail = _haar_detail_coeffs(signal, levels=3)

    if detail.size == 0:
        return {
           "WaveletASKCorrelation": 0.0,
            "WaveletFSKCorrelation": 0.0
         }

    detail = normalize(detail)

    ask_template = np.ones_like(detail)

    fsk_template = np.sign(
        np.sin(
            np.linspace(
                0,
                8 * np.pi,
                len(detail)
            )
        )
    )

    ask_corr = np.corrcoef(
        detail,
        ask_template
    )[0, 1]

    fsk_corr = np.corrcoef(
        detail,
        fsk_template
    )[0, 1]

    ask_corr = np.nan_to_num(ask_corr)

    fsk_corr = np.nan_to_num(fsk_corr)

    return {

        "WaveletASKCorrelation": ask_corr,

        "WaveletFSKCorrelation": fsk_corr

    }



def estimate_snr(signal):
    """
    Robust Blind M2M4 SNR Estimator
    Never returns NaN or Inf.
    """

    signal = np.asarray(signal)

    power = np.mean(np.abs(signal) ** 2)

    fourth = np.mean(np.abs(signal) ** 4)

    if (
        np.isnan(power)
        or np.isnan(fourth)
        or power <= EPSILON
        or fourth <= EPSILON
    ):
        return 0.0

    denominator = fourth - power**2

    if denominator <= EPSILON:
        return 0.0

    numerator = 2 * power**2 - fourth

    if numerator <= EPSILON:
        return 0.0

    gamma = numerator / denominator

    if gamma <= EPSILON:
        return 0.0

    snr = 10 * np.log10(gamma)

    if not np.isfinite(snr):
        return 0.0

    return float(snr)

def compute_misc_features(signal):
    """
    Miscellaneous Features
    """

    features = {}

    features["SpectralSymmetry"] = spectral_symmetry(signal)

    features["BlindSNR"] = estimate_snr(signal)

    return features

def compute_advanced_features(signal):
    """
    Advanced Signal Features
    ----------------------------------------
    Spectral Entropy
    Spectral Flatness
    Spectral RollOff
    Spectral Bandwidth
    MφNL
    SigmaDP
    SigmaZ2
    Exact M2M4 SNR Estimator
    """

    features = {}

    amp = instantaneous_amplitude(signal)
    phase = instantaneous_phase(signal)
    freq = instantaneous_frequency(signal)

    # =====================================================
    # FFT
    # =====================================================

    fft_mag = np.abs(np.fft.fft(signal))

    fft_mag = fft_mag[: len(fft_mag) // 2]

    fft_mag = fft_mag + EPSILON

    freqs = np.arange(len(fft_mag))

    # =====================================================
    # Spectral Entropy
    # =====================================================

    ps = fft_mag / np.sum(fft_mag)

    features["SpectralEntropy"] = -np.sum(
        ps * np.log2(ps)
    )

    # =====================================================
    # Spectral Flatness
    # =====================================================

    gm = np.exp(
        np.mean(np.log(fft_mag))
    )

    am = np.mean(fft_mag)

    features["SpectralFlatness"] = safe_divide(
        gm,
        am
    )

    # =====================================================
    # Spectral Roll-Off (85%)
    # =====================================================

    cumulative = np.cumsum(fft_mag)

    idx = np.where(
        cumulative >= 0.85 * cumulative[-1]
    )[0]

    if len(idx):

        features["SpectralRollOff"] = idx[0]

    else:

        features["SpectralRollOff"] = 0

    # =====================================================
    # Spectral Bandwidth
    # =====================================================

    centroid = np.sum(
        freqs * fft_mag
    ) / np.sum(fft_mag)

    bandwidth = np.sqrt(

        np.sum(

            ((freqs - centroid) ** 2)

            * fft_mag

        )

        / np.sum(fft_mag)

    )

    features["SpectralBandwidth"] = bandwidth

    # =====================================================
    # MphiNL
    # =====================================================

    phase_center = phase - np.mean(phase)

    phase_norm = normalize(phase_center)

    features["MphiNL"] = np.mean(
        phase_norm ** 2
    )

    # =====================================================
    # SigmaDP
    # =====================================================

    median = np.median(phase)

    features["SigmaDP"] = np.std(
        phase - median
    )

    # =====================================================
    # SigmaZ2
    # =====================================================

    analytic_fft = np.abs(
        np.fft.fft(signal)
    )

    features["SigmaZ2"] = np.var(
        analytic_fft
    )

    # =====================================================
    # Exact M2M4 Blind SNR
    # =====================================================

    M2 = np.mean(
        np.abs(signal) ** 2
    )

    M4 = np.mean(
        np.abs(signal) ** 4
    )

    denominator = M4 - (M2 ** 2)

    if denominator <= EPSILON:

        features["M2M4SNR"] = 0.0

    else:

        gamma = np.sqrt(
            safe_divide(
                2 * (M2 ** 2) - M4,
                denominator
            )
        )

        gamma = np.maximum(
            gamma,
            EPSILON
        )

        features["M2M4SNR"] = (
            10 * np.log10(gamma)
        )

    return features

def extract_features(iq_signal):
    """
    Production Feature Extractor
    """

    signal = get_complex_signal(iq_signal)

    features = {}

    # ==========================================================
    # Higher Order Moments
    # ==========================================================

    moments = compute_moments(signal)

    for key, value in moments.items():
        features[key] = np.abs(value)

    # ==========================================================
    # Higher Order Cumulants
    # ==========================================================

    cumulants = compute_cumulants(moments)

    for key, value in cumulants.items():
        features[key] = np.abs(value)

    # ==========================================================
    # Normalized Cumulant Ratios
    # ==========================================================

    ratios = compute_cumulant_ratios(cumulants)

    features.update(ratios)

    # ==========================================================
    # Instantaneous Features
    # ==========================================================

    instant = compute_instantaneous_features(signal)

    features.update(instant)

    # ==========================================================
    # PSD Features
    # ==========================================================

    psd = compute_psd_features(signal)

    features.update(psd)

    # ==========================================================
    # Zero Crossing Features
    # ==========================================================

    zcr = compute_zero_crossing_features(signal)

    features.update(zcr)

    # ==========================================================
    # Wavelet Features
    # ==========================================================

    wavelet = wavelet_template_features(signal)

    features.update(wavelet)

    # ==========================================================
    # Miscellaneous Features
    # ==========================================================

    misc = compute_misc_features(signal)

    features.update(misc)

    # ==========================================================
    # Advanced Features
    # ==========================================================

    advanced = compute_advanced_features(signal)

    features.update(advanced)
    return features

