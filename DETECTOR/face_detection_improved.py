import cv2
import numpy as np
import time
import os
from datetime import datetime
from flask import Flask
from flask_socketio import SocketIO, emit
import base64
import threading
from zeroconf import ServiceInfo, Zeroconf
import socket

# Create Flask and Socket.io app
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for sharing data
faces_data = []
current_frame_encoded = None
stop_signal = False

# Create directory for storing captured faces
if not os.path.exists('detected_faces'):
    os.makedirs('detected_faces')

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def register_service():
    local_ip = get_local_ip()
    port = 5000
    
    zeroconf = Zeroconf()
    service_info = ServiceInfo(
        "_http._tcp.local.",
        "FaceDetectionServer._http._tcp.local.",
        addresses=[socket.inet_aton(local_ip)],
        port=port,
        properties={'path': '/'},
        server=f"face-detection-server.local."
    )
    
    print(f"Registering service on {local_ip}:{port}")
    zeroconf.register_service(service_info)
    return zeroconf, service_info

def camera_processing():
    global faces_data, current_frame_encoded, stop_signal
    
    # For better accuracy, use DNN-based face detector
    face_detector_path = 'face_detector/deploy.prototxt'
    face_model_path = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'

    # If DNN model files don't exist, fall back to Haar cascade with stricter parameters
    try:
        face_detector = cv2.dnn.readNetFromCaffe(face_detector_path, face_model_path)
        use_dnn = True
        print("Using DNN face detector")
    except:
        use_dnn = False
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        print("Using Haar cascade face detector with strict parameters")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    face_detected = False
    last_notification_time = 0
    notification_cooldown = 5  # seconds
    min_confidence = 0.7  # Minimum confidence for DNN detection
    last_frame_time = 0
    frame_interval = 0.15  # ~7 FPS instead of 10

    print("Face detection system started. Press 'q' to quit.")

    while not stop_signal:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        faces = []
        
        if use_dnn:
            # DNN-based detection (more accurate)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))
            face_detector.setInput(blob)
            detections = face_detector.forward()
            
            # Extract faces with good confidence
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Filter out weak detections
                if confidence < min_confidence:
                    continue
                    
                # Compute bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure the bounding box falls within the frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Add face coordinates and confidence score
                faces.append((startX, startY, endX - startX, endY - startY, confidence))
        else:
            # Haar cascade with stricter parameters for fewer false positives
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frontal_faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=8,
                minSize=(50, 50)
            )
            
            profile_faces = profile_face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=8,
                minSize=(50, 50)
            )
            
            faces = [(x, y, w, h, 1.0) for x, y, w, h in frontal_faces]
            faces += [(x, y, w, h, 0.9) for x, y, w, h in profile_faces]
            
            faces = [face for face in faces if 0.5 <= face[2]/face[3] <= 1.5]
        
        # Draw rectangles around faces
        for i, (x, y, w, h, conf) in enumerate(faces):
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            conf_text = f"Face {i+1} ({conf:.2f})"
            cv2.putText(frame, conf_text, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Format faces data for the app
        formatted_faces = []
        for x, y, w, h, conf in faces:
            formatted_faces.append({
                'x': int(x), 
                'y': int(y), 
                'width': int(w), 
                'height': int(h),
                'confidence': float(conf)
            })
        
        # Update the global variables
        faces_data = formatted_faces
        
        # Convert current frame to base64 for streaming
        # Send frame at regular intervals regardless of face detection
        current_time = time.time()
        if current_time - last_frame_time >= frame_interval:
            # Higher quality, consistent compression
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            
            # Try to make the image smaller but clearer by scaling down if needed
            max_width = 640  # Limit width for better network performance
            if frame.shape[1] > max_width:
                scale = max_width / frame.shape[1]
                new_width = int(frame.shape[1] * scale)
                new_height = int(frame.shape[0] * scale)
                frame_resized = cv2.resize(frame, (new_width, new_height))
                _, buffer = cv2.imencode('.jpg', frame_resized, encode_params)
            else:
                _, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            current_frame_encoded = base64.b64encode(buffer).decode('utf-8')
            
            # Emit frame to all connected clients
            socketio.emit('frame_update', {
                'image': current_frame_encoded,
                'faces': faces_data,
                'timestamp': current_time
            })
            
            last_frame_time = current_time
        
        # Check if faces were detected
        if len(faces) > 0:
            if not face_detected or (current_time - last_notification_time > notification_cooldown):
                print(f"ALERT: {len(faces)} face(s) detected!")
                last_notification_time = current_time
                face_detected = True
                
                # Emit the face detection event through Socket.io
                socketio.emit('face_detected', {
                    'count': len(faces),
                    'timestamp': time.time(),
                    'image': current_frame_encoded,
                    'faces': formatted_faces
                })
        else:
            face_detected = False
        
        # Display the number of faces detected
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Face Detection System', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            stop_signal = True
            break
        elif key == ord('c') and len(faces) > 0:
            # Generate a timestamp for this capture session
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Save the full frame
            full_frame_filename = f"detected_faces/full_capture_{timestamp}.jpg"
            cv2.imwrite(full_frame_filename, frame)
            print(f"Full frame saved as {full_frame_filename}")
            
            # 2. Save each detected face individually
            for i, (x, y, w, h, conf) in enumerate(faces):
                expand = 20
                x_exp = max(0, x - expand)
                y_exp = max(0, y - expand)
                w_exp = min(frame.shape[1] - x_exp, w + 2*expand)
                h_exp = min(frame.shape[0] - y_exp, h + 2*expand)
                
                face_img = frame[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp]
                face_filename = f"detected_faces/face_{i+1}_conf_{conf:.2f}_{timestamp}.jpg"
                cv2.imwrite(face_filename, face_img)
                print(f"Face {i+1} saved as {face_filename}")
                
            print(f"Captured {len(faces)+1} images total ({len(faces)} faces + 1 full frame)")
            
        elif key >= ord('1') and key <= ord('9'):
            face_index = key - ord('1')
            if face_index < len(faces):
                x, y, w, h, conf = faces[face_index]
                expand = 20
                x = max(0, x - expand)
                y = max(0, y - expand)
                w = min(frame.shape[1] - x, w + 2*expand)
                h = min(frame.shape[0] - y, h + 2*expand)
                
                face_img = frame[y:y+h, x:x+w]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"detected_faces/face_{face_index+1}_conf_{conf:.2f}_{timestamp}.jpg"
                cv2.imwrite(filename, face_img)
                print(f"Face {face_index+1} captured and saved as {filename}")

    cap.release()
    cv2.destroyAllWindows()

# Create a route to get the latest frame
@app.route('/video_feed')
def video_feed():
    return f'<img src="data:image/jpeg;base64,{current_frame_encoded}" />'

@socketio.on('request_stream')
def handle_stream_request():
    print("Client requested video stream")
    # Just acknowledge - actual streaming happens in camera_processing loop
    emit('stream_acknowledged', {'status': 'streaming'})

@socketio.on('set_quality')
def handle_quality_change(data):
    global frame_interval
    quality = data.get('quality', 'medium')
    print(f"Stream quality changed to {quality}")
    
    if quality == 'low':
        frame_interval = 0.4  # ~2.5 FPS - extremely stable
    elif quality == 'medium':
        frame_interval = 0.25  # ~4 FPS - good balance
    else:  # high
        frame_interval = 0.18  # ~5.5 FPS - faster but may flicker on slower devices

if __name__ == '__main__':
    # Start camera processing in a separate thread
    camera_thread = threading.Thread(target=camera_processing)
    camera_thread.daemon = True
    camera_thread.start()
    
    # Register mDNS/Zeroconf service
    print("Registering service")
    zeroconf, service_info = register_service()
    print(f"Service registered - IP: {get_local_ip()}")
    
    # Start Flask app
    socketio.run(app, host='0.0.0.0', allow_unsafe_werkzeug=True)
    
    # Clean up on exit
    zeroconf.unregister_service(service_info)
    zeroconf.close() 
