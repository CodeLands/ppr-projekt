from mpi4py import MPI
import os
import cv2
import logging
from genImg import process_image
from com_decom import compress, decompress
import numpy as np
import time

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Debug mode and timing analysis toggle
DEBUG_MODE = True  # Set to False to reduce logging verbosity
TIMING_ENABLED = True  # Set to True to enable timing analysis

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format=f'[Rank {rank}] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(f'mpi_log_rank_{rank}.log'),
        logging.StreamHandler()
    ]
)

# Global timing dictionary
timing_data = {
    "Compressing images": 0,
    "Decompressing images": 0,
    "Augmenting images": 0,
}

def log_timing(step_name, start_time):
    global timing_data
    elapsed_time = time.time() - start_time
    if step_name not in timing_data:
        timing_data[step_name] = 0
    timing_data[step_name] += elapsed_time
    logging.info(f"{step_name} completed in {elapsed_time:.2f} seconds")


def calculate_percentage_timing(timing_data):
    total_time = sum(timing_data.values())
    if total_time == 0:
        return {}
    return {step: (time / total_time) * 100 for step, time in timing_data.items()}


def aggregate_timing_data():
    global timing_data
    # Serialize timing data into a list of tuples
    local_timing = list(timing_data.items())
    all_timings = comm.gather(local_timing, root=0)

    if rank == 0:
        # Aggregate timing data from all processes
        aggregated_timing = {}
        for process_timing in all_timings:
            for step, elapsed in process_timing:
                if step not in aggregated_timing:
                    aggregated_timing[step] = 0
                aggregated_timing[step] += elapsed
        return aggregated_timing
    return None

def get_image_paths(input_folder):
    return [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]

def split_channels(image):
    return cv2.split(image)

def merge_channels(channels):
    return cv2.merge(channels)

def master_task(input_folder, output_folder):
    image_paths = get_image_paths(input_folder)
    num_images = len(image_paths)
    logging.info(f"Found {num_images} images in {input_folder}")

    if num_images == 0:
        logging.warning("No images to process. Exiting.")
        return

    num_slaves = size - 1
    images_per_slave = num_images // num_slaves
    extra_images = num_images % num_slaves

    # Distribute images to slaves
    logging.info("Compressing and distributing images to slaves...")
    offset = 0
    for i in range(1, size):
        end = offset + images_per_slave + (1 if i <= extra_images else 0)
        images_chunk = image_paths[offset:end]
        data_to_send = []
        for img_path in images_chunk:
            image = cv2.imread(img_path)
            if image is None:
                logging.warning(f"Failed to load image: {img_path}")
                continue

            compress_start = time.time()
            channels = split_channels(image)
            compressed_channels = [compress(ch.astype(np.int32)) for ch in channels]
            log_timing("Compressing images", compress_start)

            data_to_send.append((img_path, compressed_channels))
        comm.send(data_to_send, dest=i, tag=11)
        offset = end

    # Receive augmented images from slaves
    logging.info("Waiting for augmented data from slaves...")
    for i in range(1, size):
        compressed_data = comm.recv(source=i, tag=22)
        for image_path, compressed_channels in compressed_data:
            decompress_start = time.time()
            decompressed_channels = [decompress(ch) for ch in compressed_channels]
            log_timing("Decompressing images", decompress_start)

            final_image = merge_channels(decompressed_channels)
            final_image = np.clip(final_image, 0, 255).astype(np.uint8)

            unique_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_final.png"
            final_path = os.path.join(output_folder, unique_filename)
            cv2.imwrite(final_path, final_image)

    # Aggregate timing data from all processes
    aggregated_timing = aggregate_timing_data()
    if TIMING_ENABLED and rank == 0:
        logging.info("Processing Timing statistics:")
        percentages = calculate_percentage_timing(aggregated_timing)
        for step, percentage in percentages.items():
            logging.info(f"{step}: {percentage:.2f}% of total active processing time")

    logging.info(f"All processing complete. Results saved in {output_folder}")

def slave_task():
    logging.info("Awaiting images from master...")
    data_received = comm.recv(source=0, tag=11)
    logging.info("Images from master received...")

    logging.info("Processing images...")
    processed_data = []
    for image_path, compressed_channels in data_received:
        decompress_start = time.time()
        decompressed_channels = [decompress(ch) for ch in compressed_channels]
        log_timing("Decompressing images", decompress_start)

        reconstructed_image = merge_channels(decompressed_channels)
        reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

        augment_start = time.time()
        augmented_images = process_image(reconstructed_image, output_folder=None)
        log_timing("Augmenting images", augment_start)

        for i, augmented_image in enumerate(augmented_images):
            compress_start = time.time()
            augmented_channels = split_channels(augmented_image)
            compressed_augmented_channels = [compress(ch.astype(np.int32)) for ch in augmented_channels]
            log_timing("Compressing images", compress_start)

            unique_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_rank{rank}_aug{i}.png"
            processed_data.append((unique_filename, compressed_augmented_channels))

    logging.info("Sending augmented images to master...")
    comm.send(processed_data, dest=0, tag=22)

    # Send timing data to the master
    aggregate_timing_data()

if __name__ == "__main__":
    input_folder = "integration/input_images"
    output_folder = "integration/processed_images"

    os.makedirs(output_folder, exist_ok=True)

    if rank == 0:
        logging.info("Starting master task")
        master_task(input_folder, output_folder)
    else:
        logging.info("Starting slave task")
        slave_task()
