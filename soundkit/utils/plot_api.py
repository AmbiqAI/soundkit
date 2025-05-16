"""Plotting API for visualizing audio data."""
import matplotlib.pyplot as plt
import numpy as np

def draw_spectrogram(
        spectrogram: np.ndarray,
        title: str = "Spectrogram",
        vmin: float = -80.0,
        vmax: float = 10.0,
        show_colorbar: bool = True,
        ) -> None:
    """ 
    Draw a spectrogram using matplotlib.
    Args:
        spectrogram (np.ndarray): Spectrogram to be plotted.
        title (str): Title of the plot.
        vmin (float): Minimum value for color scaling.
        vmax (float): Maximum value for color scaling.
        show_colorbar (bool): Whether to show the colorbar.
    """
    plt.imshow(
            spectrogram,
            aspect='auto',
            cmap='pink_r',
            origin='lower',
            vmin=vmin,
            vmax=vmax)
    plt.title(title)
    if show_colorbar:
        plt.colorbar()
