import cv2
import numpy as np
import os


def load_image(image_path):
    """Loads an image from a file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    return image


def preprocess_image(image):
    """Processes the image for OCR by converting to grayscale, denoising, thresholding, and deskewing."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Remove noise with Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary threshold
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Detect edges for deskewing
    edges = cv2.Canny(binary, 50, 150)
    coords = np.column_stack(np.where(edges > 0))
    
    # Compute rotation angle
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Rotate image to correct skew
    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(binary, rot_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def save_image(image, output_path):
    """Saves the processed image to a file."""
    cv2.imwrite(output_path, image)
    print(f"Processed image saved at {output_path}")
