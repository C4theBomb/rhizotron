import cv2
from skimage.morphology import skeletonize
import numpy as np


def find_root_count(image: np.ndarray) -> int:
    """
    Counts the number of roots in the given image.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        int: The number of roots in the image.
    """

    image_contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    return len(image_contours)


def find_total_root_length(image: np.ndarray, scaling_factor: float) -> float:
    """
    Calculates the total length of roots in an image.

    Parameters:
        image (np.ndarray): The root image.
        scaling_factor (float): The scaling factor to apply to the total length.

    Returns:
        float: The calculated root length.
    """

    skeleton = skeletonize(image)

    return np.sum(skeleton) * scaling_factor


def find_total_root_area(image: np.ndarray, scaling_factor: float) -> float:
    """
    Calculates the total area of roots in an image.

    Parameters:
        image (np.ndarray): The root image.
        scaling_factor (float): The scaling factor to apply to the total area.

    Returns:
        float: The calculated root area.
    """

    return np.sum(image / 255) * (scaling_factor ** 2)


def find_root_diameter(image: np.ndarray, scaling_factor: float) -> float:
    """
    Calculates the average diameter of roots in an image.

    Parameters:
        image (np.ndarray): The root image.
        scaling_factor (float): The scaling factor to apply to the total diameter.

    Returns:
        float: The calculated root diameter.
    """

    skeleton = skeletonize(image).astype(np.uint8)

    image_contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    y, x = np.where(skeleton == 1)

    if len(x) == 0 or len(y) == 0:
        return 0

    image_contour_points = np.vstack(image_contours).squeeze()

    diameters = []
    for point in zip(x, y):
        distances = np.linalg.norm(image_contour_points - point, axis=1)
        diameters.append(2 * np.min(distances))

    return np.mean(diameters) * scaling_factor


def find_total_root_volume(image: np.ndarray, scaling_factor: float) -> float:
    """
    Calculates the total volume of roots in an image.

    Parameters:
        image (np.ndarray): The root image.
        scaling_factor (float): The scaling factor to apply to the total volume.

    Returns:
        float: The calculated root volume.
    """

    skeleton = skeletonize(image).astype(np.uint8)

    image_contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    y, x = np.where(skeleton == 1)

    if len(x) == 0 or len(y) == 0:
        return 0

    image_contour_points = np.vstack(image_contours).squeeze()

    radii = []
    for point in zip(x, y):
        distances = np.linalg.norm(image_contour_points - point, axis=1)
        radii.append(np.min(distances) * scaling_factor)

    return np.sum(np.pi * (np.array(radii) ** 2))
