import numpy as np
import math
import time

# --------------------------------------------------------------
#                 Helper / Logging Functions
# --------------------------------------------------------------
def log_array(title, array, debug=False):
    """Log an array (with optional debug)."""
    if debug:
        print(f"{title}: {array} (Total: {len(array)})")

# --------------------------------------------------------------
#                 JPEG-LS Prediction Functions
# --------------------------------------------------------------
def predict(image, height, width, debug=False):
    """
    Perform JPEG-LS style prediction on an image (2D NumPy array).
    Returns a flattened (column-major order) array of prediction errors.
    """
    error = np.zeros((height, width), dtype=np.int32)

    for x in range(width):
        for y in range(height):
            if x == 0 and y == 0:
                # First pixel uses its own value
                error[y, x] = image[0, 0]
            elif y == 0:
                # First row uses difference from left neighbor
                error[y, x] = image[0, x - 1] - image[0, x]
            elif x == 0:
                # First column uses difference from top neighbor
                error[y, x] = image[y - 1, 0] - image[y, 0]
            else:
                left = image[y, x - 1]
                top = image[y - 1, x]
                top_left = image[y - 1, x - 1]

                # JPEG-LS prediction rule
                if top_left >= max(left, top):
                    pred = min(left, top)
                elif top_left <= min(left, top):
                    pred = max(left, top)
                else:
                    pred = left + top - top_left

                error[y, x] = pred - image[y, x]

    flat_error = error.flatten(order='F')  # Flatten in column-major order
    log_array("Predicted Values (E)", flat_error, debug)
    return flat_error

def predict_inverse(E, height, width, debug=False):
    """
    Inverse of the JPEG-LS style prediction.
    Takes a flattened array of prediction errors (column-major),
    reconstructs the original 2D array of shape (height, width).
    """
    image = np.zeros((height, width), dtype=np.int32)
    img_size = height * width

    for i in range(img_size):
        x = i // height  # Notice we're flattening in column-major order
        y = i % height

        if x == 0 and y == 0:
            image[y, x] = E[i]
        elif y == 0:
            pred = image[y, x - 1]
            image[y, x] = pred - E[i]
        elif x == 0:
            pred = image[y - 1, x]
            image[y, x] = pred - E[i]
        else:
            left = image[y, x - 1]
            top = image[y - 1, x]
            top_left = image[y - 1, x - 1]

            if top_left >= max(left, top):
                pred = min(left, top)
            elif top_left <= min(left, top):
                pred = max(left, top)
            else:
                pred = left + top - top_left

            image[y, x] = pred - E[i]

    flat_image = image.flatten(order='F')
    log_array("Reconstructed Image (Flattened)", flat_image, debug)
    return image

# --------------------------------------------------------------
#        Differential & Cumulative Sum Encoding/Decoding
# --------------------------------------------------------------
def differential_encoding(errors, debug=False):
    """
    Convert signed prediction errors to non-negative integers.
    """
    diff_encoded = [errors[0]]  # First value remains unchanged
    for val in errors[1:]:
        # Map negative/positive integers to unique non-negative integers
        diff_encoded.append(2 * val if val >= 0 else 2 * abs(val) - 1)

    log_array("Differential Encoded Values (N)", diff_encoded, debug)
    return diff_encoded

def differential_decoding(differential, debug=False):
    """
    Convert non-negative integers back to signed prediction errors.
    """
    errors = [differential[0]]
    for val in differential[1:]:
        if val % 2 == 0:
            errors.append(val // 2)
        else:
            errors.append(-(val + 1) // 2)

    log_array("Decoded Prediction Errors (E)", errors, debug)
    return errors

def cumulative_sum_encoding(differential, debug=False):
    """
    Perform cumulative sum over the differential-encoded values.
    """
    cumulative = [differential[0]]
    for i in range(1, len(differential)):
        cumulative.append(cumulative[-1] + differential[i])

    log_array("Cumulative Sum Encoded (C)", cumulative, debug)
    return cumulative

def cumulative_sum_decoding(cumulative, debug=False):
    """
    Inverse of cumulative_sum_encoding.
    """
    diff_decoded = [cumulative[0]]
    for i in range(1, len(cumulative)):
        diff_decoded.append(cumulative[i] - cumulative[i - 1])

    log_array("Decoded Differential Values (N)", diff_decoded, debug)
    return diff_decoded

# --------------------------------------------------------------
#              Interpolative Coding (IC) & Decoding
# --------------------------------------------------------------
def IC(B, C, L, H, debug=True):
    """
    Interpolative Coding (recursive).
    B = list of bits (output).
    C = array of cumulative sums.
    L, H = interval bounds.
    """
    if H - L > 1:
        if C[H] != C[L]:
            m = (H + L) // 2
            g = math.ceil(math.log2(C[H] - C[L] + 1))
            value = C[m] - C[L]
            encoded_bits = format(value, f'0{g}b')  # Convert to binary

            # Append bits to output
            B.extend(int(bit) for bit in encoded_bits)

            if debug:
                print(f"IC Encoding: L={L}, H={H}, m={m}, g={g}, C[L]={C[L]}, "
                      f"C[H]={C[H]}, Value={value}, Encoded Bits={encoded_bits}")

            IC(B, C, L, m, debug)
            IC(B, C, m, H, debug)

def decode_IC(binary_stream, C, L, H, current_bit, debug=True):
    """
    Decoding for Interpolative Coding (recursive).
    binary_stream = list of bits (input).
    C = array of cumulative sums being reconstructed.
    L, H = interval bounds.
    current_bit = index into binary_stream.
    """
    if H - L > 1:
        if C[H] != C[L]:
            m = (H + L) // 2
            g = math.ceil(math.log2(C[H] - C[L] + 1))

            if current_bit + g > len(binary_stream):
                raise ValueError(
                    f"Not enough bits to decode: Expected {g}, "
                    f"available {len(binary_stream) - current_bit}"
                )

            # Extract bits for this value
            encoded_bits = binary_stream[current_bit : current_bit + g]
            value = int(''.join(map(str, encoded_bits)), 2)
            C[m] = C[L] + value

            if debug:
                print(f"Decoding IC: L={L}, H={H}, m={m}, g={g}, "
                      f"Encoded Bits={encoded_bits}, Value={value}, C[m]={C[m]}")

            current_bit = decode_IC(binary_stream, C, L, m, current_bit + g, debug)
            current_bit = decode_IC(binary_stream, C, m, H, current_bit, debug)
        else:
            # If C[H] == C[L], fill in the gap
            for i in range(L + 1, H):
                C[i] = C[L]
                if debug:
                    print(f"Filling gap: C[{i}] = {C[L]}")

    return current_bit

def initialize_C(n, first_element, last_element):
    """
    Initialize an array of length n for the cumulative sums.
    """
    C = [0] * n
    C[0] = first_element
    C[-1] = last_element
    return C

# --------------------------------------------------------------
#                  Bits <-> Bytes Packing
# --------------------------------------------------------------
def pack_bits_to_bytes(bits):
    """
    Take a list of bits (0 or 1) and pack them into bytes.
    Returns a `bytearray`.
    """
    bytes_data = bytearray()
    current_byte = 0
    bit_count = 0

    for bit in bits:
        current_byte = (current_byte << 1) | bit
        bit_count += 1

        if bit_count == 8:
            bytes_data.append(current_byte)
            current_byte = 0
            bit_count = 0

    # If there are leftover bits, pad them
    if bit_count > 0:
        current_byte = current_byte << (8 - bit_count)
        bytes_data.append(current_byte)

    return bytes_data

def unpack_bytes_to_bits(byte_data):
    """
    Take a `bytes` or `bytearray` object and unpack into a list of bits.
    """
    bits = []
    for byte in byte_data:
        for bit_pos in range(7, -1, -1):
            bits.append((byte >> bit_pos) & 1)
    return bits

# --------------------------------------------------------------
#              Updated Compress / Decompress
# --------------------------------------------------------------
def compress(image_matrix, debug=False):
    """
    Compress a NumPy 2D matrix (e.g., grayscale image).

    Returns:
        A bytes object containing:
          - 2 bytes for height (uint16)
          - 2 bytes for width (uint16)
          - 1 byte for the first_element (uint8)
          - 4 bytes for the last_element (uint32)
          - 4 bytes for total_size (uint32)
          - followed by the Interpolative Coding bitstream (packed as bytes).
    """
    start_time = time.time()

    height, width = image_matrix.shape

    # 1) Predict & flatten errors
    predicted_vals = predict(image_matrix, height, width, debug)

    # 2) Differential encode
    diff_encoded = differential_encoding(predicted_vals, debug)

    # 3) Cumulative sum encode
    sum_encoded = cumulative_sum_encoding(diff_encoded, debug)

    # Prepare header data
    img_size = len(sum_encoded)
    first_element = sum_encoded[0]
    last_element = sum_encoded[-1]

    # 4) Interpolative coding
    binary_stream = []
    IC(binary_stream, sum_encoded, 0, img_size - 1, debug=debug)

    # Convert bits to bytes
    binary_bytes = pack_bits_to_bytes(binary_stream)

    # Build header:
    #   - height (2 bytes, uint16)
    #   - width  (2 bytes, uint16)
    #   - first_element (1 byte, uint8)
    #   - last_element (4 bytes, uint32)
    #   - total_size (4 bytes, uint32)
    header = (
        np.array([height], dtype=np.uint16).tobytes() +
        np.array([width], dtype=np.uint16).tobytes() +
        np.array([first_element], dtype=np.uint8).tobytes() +
        np.array([last_element], dtype=np.uint32).tobytes() +
        np.array([img_size], dtype=np.uint32).tobytes()
    )

    compressed_data = header + binary_bytes

    end_time = time.time()
    if debug:
        print(f"Compression time: {end_time - start_time:.2f} seconds")
        print(f"Compressed data size (bytes): {len(compressed_data)}")

    return compressed_data

def decompress(compressed_data, debug=False):
    """
    Decompress the bytes object produced by `compress`.

    Returns:
        A NumPy 2D matrix reconstructed from the compressed data.
    """
    start_time = time.time()

    # Read header
    #  - 2 bytes = height (uint16)
    #  - 2 bytes = width  (uint16)
    #  - 1 byte  = first_element (uint8)
    #  - 4 bytes = last_element  (uint32)
    #  - 4 bytes = total_size    (uint32)
    header_size = 2 + 2 + 1 + 4 + 4  # 13 bytes
    header = compressed_data[:header_size]

    height = np.frombuffer(header[0:2], dtype=np.uint16)[0]
    width  = np.frombuffer(header[2:4], dtype=np.uint16)[0]
    first_element = np.frombuffer(header[4:5], dtype=np.uint8)[0]
    last_element  = np.frombuffer(header[5:9], dtype=np.uint32)[0]
    img_size      = np.frombuffer(header[9:13], dtype=np.uint32)[0]

    if debug:
        print(f"Header Read: Height={height}, Width={width}, "
              f"First={first_element}, Last={last_element}, Size={img_size}")

    # Extract bitstream
    binary_bytes = compressed_data[header_size:]
    binary_stream = unpack_bytes_to_bits(binary_bytes)

    # Initialize and decode C
    C = initialize_C(img_size, first_element, last_element)
    decode_IC(binary_stream, C, 0, img_size - 1, 0, debug=debug)
    log_array("Decoded Cumulative Array (C)", C, debug)

    # Reverse cumulative sum -> differential
    diff_decoded = cumulative_sum_decoding(C, debug)
    # Reverse differential -> prediction errors
    prediction_errors = differential_decoding(diff_decoded, debug)
    # Reconstruct image
    image_matrix = predict_inverse(prediction_errors, height, width, debug)

    end_time = time.time()
    if debug:
        print(f"Decompression time: {end_time - start_time:.2f} seconds")

    return image_matrix
