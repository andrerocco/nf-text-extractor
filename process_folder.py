import cv2
import os
import glob

from preprocessor import fix_perspective

# Ensure directories exist
input_dir = './dataset'
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)


def process_image(input_path: str, output_path: str):
    # Read the image
    image = cv2.imread(input_path)

    # Apply the fix_perspective function
    warped_image = fix_perspective(image)

    # Save the processed image to the result directory
    cv2.imwrite(output_path, warped_image)

    print(f"Processed and saved: {input_path} -> {output_path}")


def crawl_and_process_images():
    # Find all image files in the /dataset directory
    image_paths = glob.glob(os.path.join(input_dir, '*.*'))

    # Supported image extensions
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    for image_path in image_paths:
        if any(image_path.endswith(ext) for ext in supported_extensions):
            # Set output file path
            output_image_path = os.path.join(
                output_dir, os.path.basename(image_path))
            # Process the image
            process_image(image_path, output_image_path)


if __name__ == "__main__":
    crawl_and_process_images()
