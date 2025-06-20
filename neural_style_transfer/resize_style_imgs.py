"""Just run once to save teh style images in an optimised format"""

import cv2
import os

# Define parameters
input_folder = 'neural_style_transfer/style_images'
output_folder = 'neural_style_transfer/style_images/resize'
resizing_shape = (256, 256)
size_threshold = 256  # Only resize if either dimension exceeds this

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Supported image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Process all files in input folder
for filename in os.listdir(input_folder):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        filepath = os.path.join(input_folder, filename)
        image = cv2.imread(filepath)

        if image is None:
            print(f"Failed to read {filepath}")
            continue

        height, width = image.shape[:2]

        # Only resize if one dimension exceeds threshold
        if height > size_threshold or width > size_threshold:
            resized_image = cv2.resize(image, resizing_shape)
        else:
            resized_image = image  # leave original size

        # Save resized image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, resized_image)

    print(f"Processed and saved: {output_path}")

print("All done!")
