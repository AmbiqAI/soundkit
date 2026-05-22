"""
Complex number utilities for NumPy.
"""
import numpy as np
from numpy.typing import NDArray


def complex_magnitude(
        complex_array: NDArray[np.complexfloating],
        eps: float = 1e-8) -> NDArray[np.floating]:
    """
    Compute the magnitude of a complex array.

    Args:
        complex_array: Complex-valued array
        eps: Small value to avoid sqrt(0)

    Returns:
        Real-valued array representing the magnitude of the complex array
    """
    mag = np.abs(complex_array)
    return np.maximum(mag, eps)


def complex_angle(
        complex_array: NDArray[np.complexfloating],
        eps: float = 1e-8) -> NDArray[np.floating]:
    """
    Compute the angle (phase) of a complex array.

    Args:
        complex_array: Complex-valued array
        eps: Small value to avoid division by zero

    Returns:
        Real-valued array representing the angle (phase) of the complex array
    """
    return np.arctan2(
        np.imag(complex_array),
        np.real(complex_array) + eps
    )


def polar_to_complex(
        magnitude: NDArray[np.floating],
        phase: NDArray[np.floating]) -> NDArray[np.complexfloating]:
    """
    Convert magnitude and phase to complex representation.

    Args:
        magnitude: Real-valued magnitude array
        phase: Real-valued phase array (radians)

    Returns:
        Complex array reconstructed from magnitude and phase
    """
    real = magnitude * np.cos(phase)
    imag = magnitude * np.sin(phase)
    return real + 1j * imag


def get_compressed_complex(
        spec: NDArray[np.complexfloating],
        exponent: float = 0.3,
        eps: float = 1e-7) -> NDArray[np.complexfloating]:
    """Apply power-law compression to a complex spectrogram while preserving phase.

    Args:
        spec: Complex-valued spectrogram array of shape [B, T, F]
        exponent: Compression exponent (e.g., 0.3 for power-law compression)
        eps: Small value to avoid division by zero in phase calculation

    Returns:
        Complex array with power-law compressed magnitude and original phase
    """
    mag = np.abs(spec)

    # Layer 1: Floor the magnitude for the power operation
    safe_mag = np.maximum(mag, eps)

    # Layer 2: Calculate the scaling ratio (mag^(exponent-1))
    ratio = np.power(safe_mag, exponent - 1.0)

    # Layer 3: Hard-mask the ratio to 0 where the input was 0
    ratio = np.where(mag > 0, ratio, 0.0)

    # Apply to the original complex components
    return (np.real(spec) * ratio + 1j * np.imag(spec) * ratio)


def complex_to_realarray(
        complex_array: NDArray[np.complexfloating]) -> NDArray[np.floating]:
    """
    Convert complex array to real-valued array.

    Args:
        complex_array: Complex-valued array

    Returns:
        Real-valued array with shape [B, T, 2*F]
    """
    real_part = np.real(complex_array)
    imag_part = np.imag(complex_array)
    return np.stack([real_part, imag_part], axis=-1)
