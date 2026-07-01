import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def split_iq(iq_signal):
    """
    Split IQ signal into I and Q components.
    """

    I = iq_signal[0]
    Q = iq_signal[1]

    return I, Q

def normalize_signal(signal):
    """
    Normalize signal between -1 and 1.
    """

    max_value = np.max(np.abs(signal))

    if max_value == 0:
        return signal

    return signal / max_value

def normalize_iq(iq_signal):
    """
    Normalize I and Q separately.
    """

    I,Q = split_iq(iq_signal)

    I = normalize_signal(I)

    Q = normalize_signal(Q)

    return np.array([I,Q])

def calculate_amplitude(iq_signal):
    """
    Calculate amplitude from IQ signal.
    """

    I, Q = split_iq(iq_signal)

    amplitude = np.sqrt(I**2 + Q**2)

    return amplitude

def calculate_phase(iq_signal):
    """
    Calculate phase from IQ signal.
    """

    I, Q = split_iq(iq_signal)

    phase = np.arctan2(Q, I)

    return phase

def calculate_fft(iq_signal):
    """
    Calculate FFT magnitude of IQ signal.
    """

    I, Q = split_iq(iq_signal)

    complex_signal = I + 1j * Q

    fft = np.fft.fft(complex_signal)

    magnitude = np.abs(fft)

    return magnitude

def calculate_fft_shifted(iq_signal):
    """
    Calculate centered FFT magnitude.
    """

    fft = calculate_fft(iq_signal)

    shifted = np.fft.fftshift(fft)

    return shifted

def calculate_psd(iq_signal):
    """
    Calculate Power Spectral Density (PSD)
    """

    fft = calculate_fft(iq_signal)

    psd = (fft ** 2) / len(fft)

    return psd

def add_awgn_noise(iq_signal, snr_db):
    """
    Add AWGN noise to an IQ signal.
    """

    # Convert IQ to complex signal
    I, Q = split_iq(iq_signal)

    complex_signal = I + 1j * Q

    # Signal Power
    signal_power = np.mean(np.abs(complex_signal) ** 2)

    # Convert SNR dB → Linear
    snr_linear = 10 ** (snr_db / 10)

    # Noise Power
    noise_power = signal_power / snr_linear

    # Gaussian Noise
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(complex_signal))
        + 1j * np.random.randn(len(complex_signal))
    )

    noisy_signal = complex_signal + noise

    return np.array([
        noisy_signal.real,
        noisy_signal.imag
    ])

def calculate_fft_magnitude(iq_signal):
    """
    Compute FFT magnitude of IQ signal.
    """

    I, Q = split_iq(iq_signal)

    complex_signal = I + 1j * Q

    fft = np.fft.fft(complex_signal)

    fft = np.abs(fft)

    return fft

def calculate_psd(iq_signal):
    """
    Compute Power Spectral Density.
    """

    fft = calculate_fft_magnitude(iq_signal)

    psd = (fft ** 2) / len(fft)

    return psd