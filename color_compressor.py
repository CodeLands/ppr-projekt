#!/usr/bin/env python3

"""
color_compressor.py

This script demonstrates how to use the existing JPEG-LS–style
compress/decompress code (in com_decom.py) for color images,
with integer-overflow fixes applied.

Changelog:
- Convert each RGB channel to `int32` before calling `compress(...)`.
- Convert each channel back to `uint8` after `decompress(...)` for saving.
"""

import argparse
import numpy as np
from PIL import Image

# Import from your original code file:
from com_decom import compress, decompress

def compress_color_image(input_path, output_path, debug=False):
    """
    Steps:
    1) Read an RGB image from `input_path`.
    2) Split into three channels (R, G, B).
    3) Convert channels to int32 to avoid overflow in `pred - pixel`.
    4) Use `compress(...)` on each channel.
    5) Concatenate the results into a single file, with simple length headers.
    """
    # Load image as RGB
    img = Image.open(input_path).convert("RGB")
    img_array = np.array(img)  # shape = (height, width, 3)

    # Split into channels
    R = img_array[..., 0]
    G = img_array[..., 1]
    B = img_array[..., 2]

    # --- Fix 1: Cast to int32 before compression to avoid overflow ---
    R = R.astype(np.int32)
    G = G.astype(np.int32)
    B = B.astype(np.int32)

    # Compress each channel separately
    comp_R = compress(R, debug=debug)
    comp_G = compress(G, debug=debug)
    comp_B = compress(B, debug=debug)

    # Prepare a header of 3 uint32 lengths: len_R, len_G, len_B
    len_R = len(comp_R)
    len_G = len(comp_G)
    len_B = len(comp_B)

    header = np.array([len_R, len_G, len_B], dtype=np.uint32).tobytes()
    combined_data = header + comp_R + comp_G + comp_B

    # Save to file
    with open(output_path, 'wb') as f:
        f.write(combined_data)

    if debug:
        print(f"Compressed color image -> {output_path}")
        print(f"Channel sizes: R={len_R}, G={len_G}, B={len_B}")

def decompress_color_image(input_path, output_path, debug=False):
    """
    Steps:
    1) Read the combined file with 3 compressed channels.
    2) Parse out the lengths of R, G, B data.
    3) Call `decompress(...)` from com_decom.py on each channel.
    4) Convert each channel to `uint8` for final output.
    5) Combine the 3 decompressed 2D arrays into an RGB image.
    6) Save the reconstructed image to `output_path`.
    """
    with open(input_path, 'rb') as f:
        combined_data = f.read()

    # Read lengths (3 uint32)
    offset = 0
    lengths = np.frombuffer(combined_data[offset : offset + 12], dtype=np.uint32)
    offset += 12

    len_R, len_G, len_B = lengths
    if debug:
        print(f"Reading compressed color from {input_path}")
        print(f"Channel sizes: R={len_R}, G={len_G}, B={len_B}")

    # Extract each channel's data
    comp_R = combined_data[offset : offset + len_R]
    offset += len_R
    comp_G = combined_data[offset : offset + len_G]
    offset += len_G
    comp_B = combined_data[offset : offset + len_B]
    offset += len_B

    # Decompress each channel
    R_dec = decompress(comp_R, debug=debug)
    G_dec = decompress(comp_G, debug=debug)
    B_dec = decompress(comp_B, debug=debug)

    # --- Fix 2: Cast to uint8 before forming the final image ---
    R_dec = R_dec.astype(np.uint8)
    G_dec = G_dec.astype(np.uint8)
    B_dec = B_dec.astype(np.uint8)

    # Combine into a 3D array
    height, width = R_dec.shape
    img_rec = np.zeros((height, width, 3), dtype=np.uint8)
    img_rec[..., 0] = R_dec
    img_rec[..., 1] = G_dec
    img_rec[..., 2] = B_dec

    # Save as PNG
    out_img = Image.fromarray(img_rec, mode="RGB")
    out_img.save(output_path)

    if debug:
        print(f"Decompressed color image -> {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Use JPEG-LS–style compress/decompress (from com_decom.py) on color images."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sub-command: compress
    parser_compress = subparsers.add_parser("compress", help="Compress a color image.")
    parser_compress.add_argument("--input",  required=True, help="Path to input color image (e.g. PNG).")
    parser_compress.add_argument("--output", required=True, help="Path to output combined compressed file.")
    parser_compress.add_argument("--debug",  action="store_true", help="Enable debug messages.")

    # Sub-command: decompress
    parser_decompress = subparsers.add_parser("decompress", help="Decompress a color image.")
    parser_decompress.add_argument("--input",  required=True, help="Path to input combined compressed file.")
    parser_decompress.add_argument("--output", required=True, help="Path to output reconstructed color image (PNG).")
    parser_decompress.add_argument("--debug",  action="store_true", help="Enable debug messages.")

    args = parser.parse_args()

    if args.command == "compress":
        compress_color_image(args.input, args.output, debug=args.debug)
    elif args.command == "decompress":
        decompress_color_image(args.input, args.output, debug=args.debug)

if __name__ == "__main__":
    main()
