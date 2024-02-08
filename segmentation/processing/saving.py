import json
import numpy as np
import cv2

from PIL import Image


def labelme(image_filename: str, image: Image.Image) -> str:
    """
    Convert an image with contours to a LabelMe JSON string.

    Parameters:
        image_filename (str): The filename of the image.
        image (np.ndarray): The image with contours.

    Returns:
        str: The LabelMe JSON string.
    """
    image = np.array(image)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    shapes = []
    for contour in contours:
        points = contour.squeeze(1).tolist()

        if len(points) < 3:
            continue

        shapes.append({
            'label': 'root',
            'points': points,
            'group_id': None,
            'shape_type': 'polygon',
            'flags': {}
        })

    labelme_json = json.dumps({
        'version': '4.6.0',
        'flags': {},
        'shapes': shapes,
        'imagePath': image_filename,
        'imageData': None,
        'imageHeight': image.shape[0],
        'imageWidth': image.shape[1]
    })

    return labelme_json


def save_new_mask(image: Image.Image, mask_json: str) -> Image.Image:
    """
    Save a new mask from a LabelMe JSON string.

    Parameters:
        json (str): The LabelMe JSON string.

    Returns:
        np.ndarray: The new mask.
    """

    image = np.array(image)

    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    polygons = [shape['points'] for shape in mask_json['shapes']]
    for polygon in polygons:
        points = np.array(polygon, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], (255, 255, 255))

    mask = Image.fromarray(mask)

    return mask
