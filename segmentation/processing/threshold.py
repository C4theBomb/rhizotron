import numpy as np
import cv2


def threshold_mask(mask: np.ndarray, threshold_area: int = 50) -> np.ndarray:
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
