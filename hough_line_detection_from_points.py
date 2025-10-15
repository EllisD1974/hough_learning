import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from utils.plot_utils import (
    plot_image,
    plot_lines,
    plot_clamed_lines,
)
from utils.np_img_utils import (
    rotate_points_ccw,
    load_img_as_np,
    detect_edge_points,
    get_edge_marked_img,
)


# Random points with a definate line of points included
# all_points = np.load("points.npy")

img = load_img_as_np(Path("images/test_img_road.jpg"))
edge_points = detect_edge_points(img)

edge_points_rotated = rotate_points_ccw(edge_points, img.shape)

img = get_edge_marked_img(edge_points_rotated)

plot_image(img, title="Edge points image", cmap="gray")


def hough_transform(points, theta_steps=180, rho_resolution=1):
    # Extract x,y
    x = points[:, 0]
    y = points[:, 1]

    # Theta values between -90 and +90 degrees (converted to radians)
    thetas = np.deg2rad(np.linspace(-90, 90, theta_steps))

    # Max rho possible
    max_rho = int(np.hypot(x.max(), y.max()))
    rhos = np.arange(-max_rho, max_rho + 1, rho_resolution)

    # Create accumulator array
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=int)

    # Vote in parameter space
    for xi, yi in points:
        for t_idx, theta in enumerate(thetas):
            rho = int(xi * np.cos(theta) + yi * np.sin(theta))
            rho_idx = rho + max_rho  # shift index to avoid negative
            accumulator[rho_idx, t_idx] += 1

    return accumulator, thetas, rhos

# Run the hough transform to define the "accumulator array" A(m, c)
accumulator, thetas, rhos = hough_transform(edge_points_rotated)


# Find top N lines
N = 25
indices = np.argpartition(accumulator.flatten(), -N)[-N:]
rho_idx, theta_idx = np.unravel_index(indices, accumulator.shape)

def get_hough_lines_clamped(points, rhos, thetas, rho_idx, theta_idx):
    """
    Convert Hough peak detections into clamped line segments based on data extent.

    Parameters
    ----------
    points : np.ndarray
        Nx2 array of (x, y) point coordinates.
    rhos : np.ndarray
        Array of rho values from Hough transform.
    thetas : np.ndarray
        Array of theta values (in radians) from Hough transform.
    rho_idx : list or np.ndarray
        Indices of rho peaks (from accumulator).
    theta_idx : list or np.ndarray
        Indices of theta peaks (from accumulator).

    Returns
    -------
    list of tuples
        [(x1, y1, x2, y2), ...] line segments clamped to bounding box of points.
    """
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()

    line_segments = []

    for r, t in zip(rho_idx, theta_idx):
        rho = rhos[r]
        theta = thetas[t]

        pts = []

        # Vertical line: x = rho / cos(theta)
        if np.isclose(np.cos(theta), 0):
            x_line = rho / np.cos(theta)
            if min_x <= x_line <= max_x:
                pts = [(x_line, min_y), (x_line, max_y)]

        else:
            # Solve for intersections with bounding box
            # y for x bounds
            y1 = (rho - min_x * np.cos(theta)) / np.sin(theta)
            y2 = (rho - max_x * np.cos(theta)) / np.sin(theta)
            if min_y <= y1 <= max_y:
                pts.append((min_x, y1))
            if min_y <= y2 <= max_y:
                pts.append((max_x, y2))

            # x for y bounds
            x1 = (rho - min_y * np.sin(theta)) / np.cos(theta)
            x2 = (rho - max_y * np.sin(theta)) / np.cos(theta)
            if min_x <= x1 <= max_x:
                pts.append((x1, min_y))
            if min_x <= x2 <= max_x:
                pts.append((x2, max_y))

        if len(pts) >= 2:
            # Use first two valid intersections
            line_segments.append((pts[0][0], pts[0][1], pts[1][0], pts[1][1]))

    return line_segments

clamped_hough_lines = get_hough_lines_clamped(edge_points_rotated, rhos, thetas, rho_idx, theta_idx)


# plot_lines(rhos, thetas, rho_idx, theta_idx, img)
plot_clamed_lines(clamped_hough_lines)
plt.show()

pause = 1




# # Existing 50 random points
# points = np.random.rand(50, 2)
#
# # Generate 10 points in a straight line with random spacing
# num_line_points = 10
# start_point = np.random.rand(2)  # random start
# direction = np.random.rand(2) - 0.5  # random direction
# direction = direction / np.linalg.norm(direction)  # normalize direction
#
# # Random distances between 0.05 and 0.15 for each new point
# spacings = np.random.uniform(0.05, 0.15, size=num_line_points)
#
# # Build line points
# line_points = np.array([start_point + direction * np.sum(spacings[:i]) for i in range(num_line_points)])
#
# # Combine with original points
# all_points = np.vstack((points, line_points))
# print(all_points)