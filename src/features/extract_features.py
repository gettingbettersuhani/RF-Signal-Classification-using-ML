from pyexpat import features

import numpy as np
import src.data.preprocessing as prep

def extract_time_features(iq_signal):
    """
    Extract time-domain features.
    """

    amplitude = prep.calculate_amplitude(iq_signal)

    mean = np.mean(amplitude)

    std = np.std(amplitude)

    variance = np.var(amplitude)

    maximum = np.max(amplitude)

    minimum = np.min(amplitude)

    median = np.median(amplitude)

    rms = np.sqrt(np.mean(amplitude ** 2))

    peak_to_peak = np.ptp(amplitude)

    energy = np.sum(amplitude ** 2)

    mean_abs = np.mean(np.abs(amplitude))

    abs_peak = np.max(np.abs(amplitude))

    crest_factor = abs_peak / rms

    shape_factor = rms / mean_abs

    impulse_factor = abs_peak / mean_abs

    return {

        "Mean": mean,

        "Std": std,

        "Variance": variance,

        "Maximum": maximum,

        "Minimum": minimum,

        "Median": median,

        "RMS": rms,

        "PeakToPeak": peak_to_peak,

        "Energy": energy,

        "MeanAbsolute": mean_abs,

        "AbsolutePeak": abs_peak,

        "CrestFactor": crest_factor,

        "ShapeFactor": shape_factor,

        "ImpulseFactor": impulse_factor

    }

def extract_frequency_features(iq_signal):
    """
    Extract frequency-domain features.
    """

    fft = prep.calculate_fft_magnitude(iq_signal)

    psd = prep.calculate_psd(iq_signal)

    # Basic FFT Statistics
    fft_mean = np.mean(fft)

    fft_std = np.std(fft)

    fft_max = np.max(fft)

    fft_energy = np.sum(fft ** 2)

    # Dominant Frequency Index
    dominant_frequency = np.argmax(fft)

    # Spectral Centroid
    frequencies = np.arange(len(fft))

    spectral_centroid = np.sum(frequencies * fft) / np.sum(fft)

    # Spectral Spread
    spectral_spread = np.sqrt(
        np.sum(
            ((frequencies - spectral_centroid) ** 2) * fft
        ) / np.sum(fft)
    )

    # Spectral Entropy
    normalized_psd = psd / np.sum(psd)

    spectral_entropy = -np.sum(
        normalized_psd * np.log2(normalized_psd + 1e-12)
    )

    # PSD Statistics
    psd_mean = np.mean(psd)

    psd_max = np.max(psd)

    return {

        "FFTMean": fft_mean,

        "FFTStd": fft_std,

        "FFTMaximum": fft_max,

        "FFTEnergy": fft_energy,

        "DominantFrequency": dominant_frequency,

        "SpectralCentroid": spectral_centroid,

        "SpectralSpread": spectral_spread,

        "SpectralEntropy": spectral_entropy,

        "PSDMean": psd_mean,

        "PSDMaximum": psd_max

    }

def extract_iq_features(iq_signal):
    """
    Extract IQ-domain features.
    """

    I, Q = prep.split_iq(iq_signal)

    # Mean
    i_mean = np.mean(I)
    q_mean = np.mean(Q)

    # Standard deviation
    i_std = np.std(I)
    q_std = np.std(Q)

    # Correlation
    if np.std(I) == 0 or np.std(Q) == 0:
        correlation = 0.0
    else:
        correlation = np.corrcoef(I, Q)[0, 1]

    # Phase
    phase = prep.calculate_phase(iq_signal)

    phase_std = np.std(phase)

    return {

        "IMean": i_mean,

        "QMean": q_mean,

        "IStd": i_std,

        "QStd": q_std,

        "IQCorrelation": correlation,

        "PhaseStd": phase_std

    }

def extract_features(iq_signal):
    """
    Extract complete feature vector.
    """

    features = {}

    features.update(
        extract_time_features(iq_signal)
    )

    features.update(
        extract_frequency_features(iq_signal)
    )

    features.update(
        extract_iq_features(iq_signal)
    )

    for key in features:
        if np.isnan(features[key]):
           features[key] = 0.0

        if np.isinf(features[key]):
           features[key] = 0.0

    return features