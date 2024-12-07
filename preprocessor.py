import cv2
import numpy as np


def fix_perspective(image: np.ndarray) -> np.ndarray:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    threshold_image = cv2.threshold(
        blur_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # Apply morphology
    kernel = np.ones((5, 5), np.uint8)
    morphology_image = cv2.morphologyEx(
        threshold_image, cv2.MORPH_CLOSE, kernel)
    morphology_image = cv2.morphologyEx(
        morphology_image, cv2.MORPH_OPEN, kernel)

    # Get largest contour
    contours = cv2.findContours(
        morphology_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    area_thresh = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > area_thresh:
            area_thresh = area
            big_contour = c

    # Draw white filled largest contour on black just as a check to see it got the correct region
    page = np.zeros_like(image)
    cv2.drawContours(page, [big_contour], 0, (255, 255, 255), -1)

    # Get perimeter and approximate a polygon
    peri = cv2.arcLength(big_contour, True)
    corners = cv2.approxPolyDP(big_contour, 0.04 * peri, True)

    # Draw polygon on input image from detected corners
    polygon_image = image.copy()
    cv2.polylines(polygon_image, [corners], True, (0, 0, 255), 1, cv2.LINE_AA)
    # cv2.drawContours(page,[corners],0,(0,0,255),1)

    print(len(corners))
    print(corners)

    # Reformat corners to a 2D array and sort them
    icorners = np.array([corner[0]
                        for corner in corners])  # Extract (x, y) pairs
    sorted_corners = __order_points(icorners)

    # Calculate width as the average of the top and bottom sides
    # Top: distance between top-left and top-right
    top_width = np.linalg.norm(sorted_corners[1] - sorted_corners[0])
    # Bottom: distance between bottom-left and bottom-right
    bottom_width = np.linalg.norm(sorted_corners[2] - sorted_corners[3])
    width = int((top_width + bottom_width) / 2)

    # Calculate height as the average of the left and right sides
    # Left: distance between top-left and bottom-left
    left_height = np.linalg.norm(sorted_corners[3] - sorted_corners[0])
    # Right: distance between top-right and bottom-right
    right_height = np.linalg.norm(sorted_corners[2] - sorted_corners[1])
    height = int((left_height + right_height) / 2)

    # Ensure width and height are positive
    width = abs(width)
    height = abs(height)

    # Define the output corners
    ocorners = np.array([[0, 0], [width, 0], [width, height], [
                        0, height]], dtype="float32")

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(sorted_corners, ocorners)

    # Apply the perspective warp
    warped_image = cv2.warpPerspective(image, M, (width, height))

    # Write the results
    cv2.imwrite("temp/efile_thresh.jpg", threshold_image)
    cv2.imwrite("temp/efile_morph.jpg", morphology_image)
    cv2.imwrite("temp/efile_polygon.jpg", polygon_image)
    cv2.imwrite("temp/efile_warped.jpg", warped_image)


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


if __name__ == "__main__":
    input_image = cv2.imread("./dataset/finger_receipt_1.jpeg")

    fix_perspective(input_image)
