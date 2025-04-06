#!/bin/bash
# Setup script for Face Detection on Raspberry Pi

echo "===== Setting up Face Detection System on Raspberry Pi ====="

# Update package lists
echo "Updating package lists..."
sudo apt update

# Install required system packages
echo "Installing OpenCV and related libraries..."
sudo apt install -y python3-opencv libopencv-dev libatlas-base-dev

echo "Installing other Python libraries..."
sudo apt install -y python3-numpy python3-flask python3-requests 
sudo apt install -y python3-zeroconf python3-pil python3-imutils

# Install Raspberry Pi specific packages
echo "Installing Raspberry Pi camera libraries..."
sudo apt install -y python3-libcamera python3-rpi.gpio

# Install Flask-SocketIO (might be missing from apt)
echo "Installing Flask-SocketIO..."
sudo pip3 install --break-system-packages flask-socketio

# Create required directories
echo "Setting up directories..."
mkdir -p ~/DETECTOR/cascades
mkdir -p ~/DETECTOR/face_detector

# Download required Haar cascade files
echo "Downloading required Haar cascade files..."
wget -O ~/DETECTOR/cascades/haarcascade_frontalface_default.xml \
  https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

# Set permissions for camera and video
echo "Setting up camera permissions..."
sudo usermod -a -G video $USER

# Create a run script
echo "Creating run script..."
cat > ~/run_face_detector.sh << 'EOF'
#!/bin/bash
cd ~/DETECTOR
python3 face_detection_improved.py
EOF

chmod +x ~/run_face_detector.sh

echo ""
echo "===== Setup Complete! ====="
echo ""
echo "To run the face detection system:"
echo "  ~/run_face_detector.sh"
echo ""
echo "You may need to reboot for camera permissions to take effect:"
echo "  sudo reboot"
echo ""
echo "After reboot, the face detection server will be accessible at:"
echo "  http://$(hostname -I | awk '{print $1}'):5000" 