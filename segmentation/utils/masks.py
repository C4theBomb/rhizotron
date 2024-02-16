import json
import numpy as np
import cv2


def to_comparison(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create a comparison image from an image and its mask.

    Parameters:
        image (np.ndarray): The original image.
        mask (np.ndarray): The mask image.

    Returns:
        np.ndarray: The comparison image.
    """

    image = np.array(image)
    mask = np.array(mask)

    comparison = cv2.addWeighted(image, 0.5, mask, 0.5, 0)

    return comparison


def to_labelme(image_filename: str, image: np.ndarray) -> str:
    """
    Convert an image with contours to a LabelMe JSON string.

    Parameters:
        image_filename (str): The filename of the image.
        image (np.ndarray): The image with contours.

    Returns:
        str: The LabelMe JSON string.
    """

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


def from_labelme(image: np.ndarray, mask_json: str) -> np.ndarray:
    """
    Save a new mask from a LabelMe JSON string.

    Parameters:
        image (np.ndarray): The original image.
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

    return mask


def threshold(mask: np.ndarray, threshold_area: int = 50) -> np.ndarray:
    """
    Apply thresholding to a binary mask based on contour area.

    Parameters:
        mask (np.ndarray): Binary mask image.
        threshold_area (int, optional): Minimum contour area threshold. Defaults to 50.

    Returns:
        np.ndarray: Thresholded mask image.
    """

    output_contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    if len(output_contours) == 0:
        return mask

    hierarchy = hierarchy.squeeze(0)

    threshold_contours = []
    threshold_heirarchy = []
    for i in range(len(output_contours)):
        if hierarchy[i][3] != -1:
            continue

        current_index = hierarchy[i][2]
        contour_area = cv2.contourArea(output_contours[i])
        while current_index != -1:
            contour_area -= cv2.contourArea(output_contours[current_index])
            current_index = hierarchy[current_index][0]

        if contour_area < threshold_area:
            continue

        threshold_contours.append(output_contours[i])
        threshold_heirarchy.append(hierarchy[i])

        current_index = hierarchy[i][2]
        while current_index != -1:
            threshold_contours.append(output_contours[current_index])
            threshold_heirarchy.append(hierarchy[current_index])
            current_index = hierarchy[current_index][0]

    thresholded_mask = np.zeros(mask.shape, dtype=np.uint8)

    for i in range(len(threshold_contours)):
        if threshold_heirarchy[i][3] != -1:
            continue

        cv2.drawContours(thresholded_mask, threshold_contours, i, 255, cv2.FILLED)

    for i in range(len(threshold_contours)):
        if threshold_heirarchy[i][3] == -1:
            continue

        cv2.drawContours(thresholded_mask, threshold_contours, i, 0, cv2.FILLED)

    return thresholded_mask
