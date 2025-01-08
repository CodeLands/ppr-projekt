from flask import Flask, request, jsonify
import cv2
import numpy as np
import sys
import os
import base64
from genImg import process_image

# Import the model files from ../model
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
if model_path not in sys.path:
    sys.path.append(model_path)

# Import the train_model function
from train import train_model
from verification import main as verification_main

app = Flask(__name__)

def extract_frames(video_data_base64):
    # Decode the base64 video data
    video_data = base64.b64decode(video_data_base64)

    # Save the raw video data for debugging
    raw_video_path = 'raw_video_received.mp4'
    with open(raw_video_path, 'wb') as f:
        f.write(video_data)

    print(f"Raw video data saved to {raw_video_path}")

    # Capture frames using VideoCapture
    cap = cv2.VideoCapture(raw_video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1

    cap.release()

    if not frames:
        print("Error: No frames extracted from the video.")
    else:
        print(f"Extracted {frame_count} frames from the video.")

    return frames, raw_video_path

def process_frames_for_verification(frames, userId, video_path):
    print("Processing frame for verification...")
    if not frames or frames[0] is None or not frames[0].size:
        print("Error: The frame is empty or not valid.")
        return False
    verification_result = verification_main(userId, frames[0])
    os.remove(video_path)  # Delete the video file after processing
    return verification_result

def process_frames_for_registration(frames, userId, video_path):
    st = 0
    for frame in frames:
        print(f"Processing frame for registration. Frame shape: {frame.shape}")
        process_image(frame, "../model/images/me", st)
        st += 1

    print("Starting model training...")
    train_model(userId)
    os.remove(video_path)  # Delete the video file after processing
    return 'true'

@app.route('/login-face', methods=['POST'])
def verify_face():
    try:
        video_file_base64 = request.data.decode('utf-8')
        if not video_file_base64:
            print("Error: No video received.")
            return jsonify({
                'success': False,
                'message': 'No video received',
                'wasRecognized': False
                }), 400
        userId = request.args.get('userId')
        if not userId:
            print("Error: No user id received.")
            return jsonify({
                'success': False,
                'message': 'No user id received',
                'wasRecognized': False
                }), 400
        frames, video_path = extract_frames(video_file_base64)
        verification_result = process_frames_for_verification(frames, userId, video_path)
        return jsonify({
            'success': True,
            'message': 'Video received and face recognition ran successfully',
            "wasRecognized": verification_result
            }), 200
    except Exception as e:
        print(f"Exception in verify_face: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'An error occurred during face verification',
            'wasRecognized': False
            }), 500

@app.route('/register-face', methods=['POST'])
def register_face():
    try:
        video_file_base64 = request.data.decode('utf-8')
        if not video_file_base64:
            print("Error: No video received.")
            return jsonify({
                'success': False,
                'message': 'No video received',
                'wasSetup': False
                }), 400
        userId = request.args.get('userId')
        if not userId:
            print("Error: No user id received.")
            return jsonify({
                'success': False,
                'message': 'No user id received',
                'wasSetup': False
                }), 400
        frames, video_path = extract_frames(video_file_base64)
        training_result = process_frames_for_registration(frames, userId, video_path)
        return jsonify({
            'success': True,
            'message': 'Video received and face registration ran successfully',
            'wasSetup': True #training_result
            }), 200
    except Exception as e:
        print(f"Exception in register_face: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'An error occurred during face registration',
            'wasSetup': False
            }), 500

if __name__ == '__main__':
    app.run(debug=True, port=4000)
