#!/usr/bin/env python3
import cv2
import time
import os

print("Camera Test Script for Raspberry Pi")
print("===================================")

# Create directory for test images
if not os.path.exists('camera_tests'):
    os.makedirs('camera_tests')

# From v4l2-ctl output we know the webcam is on video0 and video1
# But let's test a few more indices just to be sure
test_indices = [0, 1, 2, 3]

for idx in test_indices:
    print(f"\nTesting camera index {idx}...")
    try:
        # Try with default backend
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"  Camera {idx} opened successfully")
            # Try to read a frame
            for i in range(5):  # Try up to 5 times
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"  SUCCESS: Camera {idx} works! Frame shape: {frame.shape}")
                    # Save test image
                    test_filename = f'camera_tests/camera_{idx}_test.jpg'
                    cv2.imwrite(test_filename, frame)
                    print(f"  Saved test image to {test_filename}")
                    break
                else:
                    print(f"  Attempt {i+1}: No frame received, retrying...")
                    time.sleep(0.5)
            else:
                print(f"  Failed to get frame from camera {idx}")
        else:
            print(f"  Failed to open camera {idx}")
    except Exception as e:
        print(f"  Error testing camera {idx}: {e}")
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
    
    # Wait between tests
    time.sleep(1)

print("\nTesting complete! Check the camera_tests directory for any captured images.") 