import cv2
import numpy as np
import random
import os

def process_image(frame, output_folder=None, st=0):
    num_augmented_copies = 30  # Number of augmented copies per frame
    resize_width = 224  # Set desired width to 224
    resize_height = 224  # Set desired height to 224

    if output_folder is not None:
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

    # Define augmentation functions
    def apply_gaussian_blur(image):
        kernel_size = (5, 5)  # Size of the kernel for Gaussian blur
        blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
        return blurred_image

    def apply_random_rotation(image):
        angle = random.randint(-30, 30)
        rows, cols, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        return rotated_image

    def apply_brightness(image):
        brightness_offset = random.randint(-50, 50)
        modified_image = np.clip(image.astype(np.int32) + brightness_offset, 0, 255).astype(np.uint8)
        return modified_image

    def apply_random_zoom(image, max_zoom=0.2):
        rows, cols, _ = image.shape
        zoom_factor = 1 + random.uniform(0, max_zoom)
        zoomed_image = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

        # Crop to the original size
        new_rows, new_cols, _ = zoomed_image.shape
        start_row = (new_rows - rows) // 2
        start_col = (new_cols - cols) // 2
        cropped_image = zoomed_image[start_row:start_row+rows, start_col:start_col+cols]

        return cropped_image

    # Resize the frame
    resized_image = cv2.resize(frame, (resize_width, resize_height))

    # Augment the image
    augmented_images = []
    for i in range(num_augmented_copies):
        image_copy = resized_image.copy()

        # Apply augmentations
        image_copy = apply_gaussian_blur(image_copy)
        image_copy = apply_random_rotation(image_copy)
        image_copy = apply_brightness(image_copy)
        image_copy = apply_random_zoom(image_copy)

        augmented_images.append(image_copy)

        # Save to disk if output_folder is specified
        if output_folder is not None:
            filename = f"{output_folder}/{st:05d}_{i:05d}.jpg"
            cv2.imwrite(filename, image_copy)

    print(f"Processing completed for frame {st}")

    return augmented_images
