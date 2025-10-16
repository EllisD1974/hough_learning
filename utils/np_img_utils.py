import numpy as np
from pathlib import Path
from PIL import Image


def rotate_points_cw(points, image_shape):
    """
    Rotate 2D points 90 degrees clockwise around the origin based on image size.

    :param points: numpy array of shape (N, 2)
    :param image_shape: (height, width) of image
    """
    h, w = image_shape
    x, y = points[:, 0], points[:, 1]
    new_x = y
    new_y = w - 1 - x
    return np.column_stack((new_x, new_y))

def rotate_points_ccw(points, image_shape):
    """
    Rotate 2D points 90 degrees counter-clockwise based on image size.

    :param points: numpy array of shape (N, 2) where each row is (x, y)
    :param image_shape: (height, width) of the image the points came from
    :return: rotated Nx2 array
    """
    h, w = image_shape[:2]  # height, width
    x = points[:, 0]
    y = points[:, 1]

    new_x = h - 1 - y
    new_y = x
    return np.column_stack((new_x, new_y))

def load_img_as_np(img_path: Path) -> np.array:
    image = Image.open(img_path).convert("L")
    img = np.array(image)

    return img

def detect_edge_points(img: np.array, threshold=50) -> np.array:
    # Simple edge detection (threshold gradient)
    edges = np.zeros_like(img)
    gx, gy = np.gradient(img.astype(float))
    grad = np.hypot(gx, gy)
    edges[grad > threshold] = 1  # threshold

    # Extract edge points (y, x)
    edge_points = np.column_stack(np.nonzero(edges))

    return edge_points

def get_edge_marked_img(edge_points: np.array) -> np.array:
    # Suppose your points are within a known width/height
    width = np.max(edge_points[:, 0]) + 10
    height = np.max(edge_points[:, 1]) + 10

    # Create a blank image
    img = np.zeros((height, width), dtype=np.uint8)

    # Mark point locations
    for x, y in edge_points:
        img[int(y), int(x)] = 255  # white point

    return img
