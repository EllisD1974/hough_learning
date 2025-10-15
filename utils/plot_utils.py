import numpy as np
import matplotlib.pyplot as plt


def plot_image(img, title: str = None, cmap: str = None):
    title = title or ""
    cmap = cmap or "grap"

    # Plot the image
    # plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap=cmap)
    plt.title(title)

def plot_lines(rhos, thetas, rho_idx, theta_idx, img):
    # Plot each line
    for r, t in zip(rho_idx, theta_idx):
        rho = rhos[r]
        theta = thetas[t]

        # Convert (rho, theta) to line
        if np.sin(theta) != 0:
            y = np.arange(img.shape[0])
            x = (rho - y * np.sin(theta)) / np.cos(theta)
            plt.plot(x, y, linewidth=2)
        else:  # vertical line
            x = rho / np.cos(theta)
            plt.axvline(x=x, linewidth=2)

def plot_clamed_lines(lines):
    """
    Plot clamped line segments
    :param lines: Tuple(x1, y1, x2, y2)
    :return:
    """
    for (x1, y1, x2, y2) in lines:
        plt.plot([x1, x2], [y1, y2], linewidth=2)
