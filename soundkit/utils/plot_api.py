"""Plotting API for visualizing audio data."""
import io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
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


def plot_spectrograms(images, titles=None, vmin_vmax=None, figsize=(10, 2.5),
                      show_colorbar=True, cmap="pink_r", save_path=None, show_fig=False):
    """
    Plot a list of spectrogram images vertically and optionally save to a file.

    Args:
        images (list of 2D np.ndarray): Spectrogram images (transposed).
        titles (list of str): Titles for each subplot.
        vmin_vmax (list of tuple): (vmin, vmax) per subplot.
        figsize (tuple): Size of the whole figure (width, height per subplot).
        show_colorbar (bool): Show colorbars.
        cmap (str): Colormap for imshow.
        save_path (str): If given, saves the figure to this path.
    """
    n = len(images)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1]*n))

    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for i, ax in enumerate(axes):
        image = images[i]
        title = titles[i] if titles else None
        vmin, vmax = vmin_vmax[i] if vmin_vmax else (None, None)

        im = ax.imshow(image, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
        if title:
            ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        if show_colorbar:
            plt.colorbar(im, ax=ax, format="%.1f", fraction=0.046, pad=0.01)

    plt.tight_layout(pad=1.0)

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    if show_fig:
        plt.show()
    return fig

import io
import tensorflow as tf

def fig_to_image(fig):
    """Convert a Matplotlib figure to a TensorFlow image tensor."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)  # Add batch dimension
    buf.close()
    plt.close(fig)
    return image
