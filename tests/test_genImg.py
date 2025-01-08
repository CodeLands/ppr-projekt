import numpy as np
import sys
from pathlib import Path

project_root = str(Path(__file__).parents[1])  # Adjust the number of parents based on actual path to project root
sys.path.insert(0, project_root)

from genImg import process_image
import os


# Create a dummy image for testing
def create_dummy_image(width, height, color=(255, 0, 0)):
    """Create a dummy image with the specified color."""
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = color
    return image


def test_process_image():
    # Create a dummy image
    dummy_image = create_dummy_image(224, 224)

    # Directory to save processed images
    test_output_dir = "test_output_images"
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    # Process the dummy image
    process_image(dummy_image, test_output_dir, num_augmented_copies=2)

    # Check if images were created
    processed_images = os.listdir(test_output_dir)
    assert len(processed_images) == 2, "Expected 2 processed images, found: {}".format(len(processed_images))
    print("Test passed: {} images created.".format(len(processed_images)))

    # Clean up
    for f in processed_images:
        os.remove(os.path.join(test_output_dir, f))
    os.rmdir(test_output_dir)


if __name__ == '__main__':
    test_process_image()
