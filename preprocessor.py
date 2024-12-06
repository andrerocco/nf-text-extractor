import cv2
import numpy as np


def fix_perspective(image: np.ndarray) -> np.ndarray:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    threshold_image = cv2.threshold(
        blur_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # Apply morphology
    kernel = np.ones((7, 7), np.uint8)
    morph = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # Get largest contour
    contours = cv2.findContours(
        morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    polygon = image.copy()
    cv2.polylines(polygon, [corners], True, (0, 0, 255), 1, cv2.LINE_AA)
    # cv2.drawContours(page,[corners],0,(0,0,255),1)

    print(len(corners))
    print(corners)

    # For simplicity get average of top/bottom side widths and average of left/right side heights
    # note: probably better to get average of horizontal lengths and of vertical lengths
    width = 0.5*((corners[0][0][0] - corners[1][0][0]) +
                 (corners[3][0][0] - corners[2][0][0]))
    height = 0.5*((corners[2][0][1] - corners[1][0][1]) +
                  (corners[3][0][1] - corners[0][0][1]))
    width = int(width)
    height = int(height)

    # Reformat input corners to x,y list
    icorners = []
    for corner in corners:
        pt = [corner[0][0], corner[0][1]]
        icorners.append(pt)
    icorners = np.float32(icorners)

    # Get corresponding output corners from width and height
    ocorners = [[width, 0], [0, 0], [0, height], [width, height]]
    ocorners = np.float32(ocorners)

    # Get perspective tranformation matrix
    M = cv2.getPerspectiveTransform(icorners, ocorners)

    # Do perspective
    warped = cv2.warpPerspective(image, M, (width, height))

    # Write results
    cv2.imwrite("temp/efile_thresh.jpg", threshold_image)
    cv2.imwrite("temp/efile_morph.jpg", morph)
    cv2.imwrite("temp/efile_polygon.jpg", polygon)
    cv2.imwrite("temp/efile_warped.jpg", warped)


if __name__ == "__main__":
    input_image = cv2.imread("./dataset/angled_receipt_2.jpeg")

    fix_perspective(input_image)
