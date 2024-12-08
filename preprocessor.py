import cv2
import numpy as np
import os


def __order_points(points: np.ndarray) -> np.ndarray:
    """
    Order the points in the contour in: top-left, top-right, bottom-right, bottom-left.

    Parameters:
    - points: np.ndarray
        Array of points in the contour

    Returns:
    - rect: np.ndarray
        Array of points in the contour in the correct order
    """

    # Convert to numpy array for easier manipulation
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]  # Top-left
    rect[2] = points[np.argmax(s)]  # Bottom-right

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]  # Top-right
    rect[3] = points[np.argmax(diff)]  # Bottom-left

    return rect


def fix_perspective(image: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Fix the perspective of the image by applying a series of filters.
    """

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    threshold_image = cv2.threshold(
        blur_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Apply morphology
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # Get largest contour
    contours = cv2.findContours(
        morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    area_thresh = 0
    big_contour = None

    for c in contours:
        area = cv2.contourArea(c)
        if area > area_thresh:
            area_thresh = area
            big_contour = c

    if big_contour is None:
        raise ValueError("No valid contour found in the image.")

    # Get perimeter and approximate a polygon
    peri = cv2.arcLength(big_contour, True)
    corners = cv2.approxPolyDP(big_contour, 0.04 * peri, True)

    # Ensure we have at least 4 points
    points = corners.reshape(-1, 2)

    if len(points) < 4:
        raise ValueError(
            "Contour has fewer than 4 points, cannot perform perspective transform.")

    # If more than 4 points, reduce to the 4 most extreme points
    if len(points) > 4:
        # Compute convex hull to reduce unnecessary points
        hull = cv2.convexHull(points)
        points = hull.reshape(-1, 2)

        # If still more than 4 points, sort and pick the most extreme 4
        if len(points) > 4:
            points = __order_points(points[:4])  # Pick first 4 ordered points
        else:
            points = __order_points(points)
    else:
        points = __order_points(points)

    # Now we have exactly 4 points, ordered correctly
    ordered_corners = points

    # Compute the width and height of the new image
    width_top = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
    width_bottom = np.linalg.norm(ordered_corners[2] - ordered_corners[3])
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(ordered_corners[3] - ordered_corners[0])
    height_right = np.linalg.norm(ordered_corners[2] - ordered_corners[1])
    max_height = int(max(height_left, height_right))

    # Define the destination points for the warp
    destination_corners = np.array([
        [0, 0],                    # Top-left
        [max_width - 1, 0],        # Top-right
        [max_width - 1, max_height - 1],  # Bottom-right
        [0, max_height - 1]        # Bottom-left
    ], dtype="float32")

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(ordered_corners, destination_corners)

    # Perform the warp
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    # Write results
    if debug:
        os.makedirs("temp", exist_ok=True)
        cv2.imwrite("temp/debug_image_thresh.jpg", threshold_image)
        cv2.imwrite("temp/debug_image_morph.jpg", morph)
        cv2.imwrite("temp/debug_image_warped.jpg", warped)

        polygon = image.copy()
        cv2.polylines(polygon, [corners], True, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite("temp/debug_image_polygon.jpg", polygon)

    return warped


def boost_readability(image: np.ndarray) -> np.ndarray:
    """
    Boost the readability of the image by applying a series of filters.
    This version connects the letters while enhancing readability.
    """

    # Denoise the image
    # dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpen_image = cv2.GaussianBlur(gray_image, (0, 0), 3)
    sharpen_image = cv2.addWeighted(gray_image, 1.5, sharpen_image, -0.5, 0)

    return sharpen_image


if __name__ == "__main__":
    input_image = cv2.imread("./dataset/angled_small_receipt_2.jpeg")

    warped_image = fix_perspective(input_image, debug=True)

    cv2.imwrite("temp/final_result.jpg", warped_image)

    boosted_image = boost_readability(warped_image)

    cv2.imwrite("temp/final_result_boosted.jpg", boosted_image)
