import cv2
import numpy as np
import random
import os
from numba import jit

def process_image(frame, output_dir="../model/images/me", st=0):
    num_augmented_copies = 28  # Number of augmented copies per frame
    resize_width = 224  # Set desired width to 224
    resize_height = 224  # Set desired height to 224

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Custom augmentation functions
    @jit(nopython=True)
    def apply_custom_noise(image):
        square_size = 2  # Velikost črnih kvadratov
        for _ in range(50):
            y = random.randint(0, image.shape[0] - square_size)
            x = random.randint(0, image.shape[1] - square_size)
            for i in range(square_size):
                for j in range(square_size):
                    image[y + i, x + j] = [0, 0, 0]
        return image


    @jit(nopython=True)
    def apply_custom_brightness(image):
        brightness_offset = random.randint(-50, 50)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(3):
                    new_value = image[y, x, c] + brightness_offset
                    image[y, x, c] = max(0, min(255, new_value))
        return image


    @jit(nopython=True)
    def apply_saturation(image):
        factor = random.uniform(0.5, 1.5)  # Naključni faktor nasičenosti
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                r, g, b = image[y, x]
                gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  # Izračun sivinske vrednosti
                r = gray + factor * (r - gray)
                g = gray + factor * (g - gray)
                b = gray + factor * (b - gray)
                image[y, x] = [int(max(0, min(255, r))), int(max(0, min(255, g))), int(max(0, min(255, b)))]
        return image


    @jit(nopython=True)
    def apply_contrast(image):
        factor = random.uniform(0.5, 1.5)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(3):
                    new_value = 128 + factor * (image[y, x, c] - 128)
                    image[y, x, c] = int(max(0, min(255, new_value)))
        return image

    def prepare_image(image):
        # Odstranjevanje šuma
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # Pretvorba v LAB barvni model in nazaj v BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

        # Linearizacija sivin
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lut = np.array([np.clip((i / 255.0) ** 2.2 * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)
        gray = cv2.LUT(gray, lut)

        return gray

    # Resize the frame
    resized_image = cv2.resize(frame, (resize_width, resize_height))

    # Augment the image
    for i in range(num_augmented_copies):

        image_copy = resized_image.copy()

        image_copy = prepare_image(image_copy)

        # Apply augmentations
        image_copy = apply_custom_noise(image_copy)
        image_copy = apply_custom_brightness(image_copy)
        image_copy = apply_saturation(image_copy)
        image_copy = apply_contrast(image_copy)

        # Format the filename based on the value of i
        filename = f"{output_dir}/"
        filename += f"{st:05d}_" if st < 10000 else f"{st:05d}_"
        filename += f"{i:05d}.jpg" if i < 10000 else f"{i:05d}.jpg"

        cv2.imwrite(filename, image_copy)
        #print(f"Saved: {filename}")

    print(f"Processing completed for frame {st}")
