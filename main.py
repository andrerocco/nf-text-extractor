import os
from image_processor import load_image, preprocess_image, save_image


INPUT_IMAGE_PATH = "./dataset/receipt_1.jpg"
OUTPUT_IMAGE_PATH = "./dataset/processed_receipt.jpg"

def main():
    os.makedirs("./dataset", exist_ok=True)

    image = load_image(INPUT_IMAGE_PATH)

    processed_image = preprocess_image(image)

    save_image(processed_image, OUTPUT_IMAGE_PATH)

if __name__ == "__main__":
    main()