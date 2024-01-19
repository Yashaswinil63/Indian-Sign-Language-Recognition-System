import os
import cv2
import numpy as np

def preprocess_image(image_path):
    # Load the original image
    original_image = cv2.imread(image_path)

    # Normalize the image to the range [0, 1]
    normalized_image = original_image.astype(np.float32) / 255.0

    # Create the output directory if it doesn't exist
    output_directory = os.path.dirname(image_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Construct the output image path for the normalized image
    output_normalized_image_path = os.path.join(output_directory, 'normalized_image.jpg')
    cv2.imwrite(output_normalized_image_path, (normalized_image * 255).astype(np.uint8))

    # Convert the image to grayscale
    img = cv2.imread(output_normalized_image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is successfully loaded
    if img is None:
        print(f"Error: Unable to read the image at {output_normalized_image_path}")
        return None

    # Apply GaussianBlur to the image to reduce noise and improve edge detection
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_img, 50, 150)  # Adjust the threshold values based on your specific case

    # Construct the output image path for the preprocessed image
    preprocessed_image_path = os.path.join(output_directory, f"preprocessed_{os.path.basename(image_path)}")
    cv2.imwrite(preprocessed_image_path, edges)

    return preprocessed_image_path
