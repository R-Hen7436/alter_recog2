import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, Image, Alert } from 'react-native';
import io from 'socket.io-client';
import { uploadImage } from '../utils/firebase';
import Zeroconf from 'react-native-zeroconf';

export default function LiveFeedScreen() {
  const [connected, setConnected] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [faceData, setFaceData] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [serverAddress, setServerAddress] = useState(null);
  const [scanning, setScanning] = useState(true);
  const socketRef = useRef(null);
  const zeroconfRef = useRef(null);

  // Initialize zeroconf and scan for the server
  useEffect(() => {
    zeroconfRef.current = new Zeroconf();
    
    zeroconfRef.current.on('resolved', service => {
      if (service.name === 'FaceDetectionServer') {
        const host = service.addresses[0];
        const port = service.port;
        setServerAddress(`http://${host}:${port}`);
        setScanning(false);
      }
    });
    
    zeroconfRef.current.on('error', err => {
      console.error('Zeroconf error:', err);
      setScanning(false);
      Alert.alert(
        "Connection Error",
        "Couldn't find the face detection server. Make sure it's running on the same network."
      );
    });
    
    // Start scanning
    zeroconfRef.current.scan('http', 'tcp', 'local.');
    
    return () => {
      if (zeroconfRef.current) {
        zeroconfRef.current.stop();
      }
    };
  }, []);

  // Connect to socket.io server once we have the address
  useEffect(() => {
    if (!serverAddress) return;
    
    console.log(`Connecting to server at ${serverAddress}`);
    socketRef.current = io(serverAddress);
    
    socketRef.current.on('connect', () => {
      setConnected(true);
      console.log('Connected to camera server');
    });
    
    socketRef.current.on('disconnect', () => {
      setConnected(false);
      console.log('Disconnected from camera server');
    });
    
    socketRef.current.on('face_detected', (data) => {
      setCurrentFrame(`data:image/jpeg;base64,${data.image}`);
      setFaceData(data.faces || []);
    });
    
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, [serverAddress]);

  const captureFace = async (faceIndex) => {
    if (!currentFrame) return;
    
    try {
      setUploading(true);
      
      // Create a filename with timestamp
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const filename = `face_${faceIndex}_${timestamp}.jpg`;
      
      // Extract the base64 data from the image string
      const base64Data = currentFrame.split(',')[1];
      
      // Create a URI for the image
      const imageUri = `data:image/jpeg;base64,${base64Data}`;
      
      // Upload to Firebase
      const downloadURL = await uploadImage(imageUri, filename);
      
      Alert.alert(
        "Face Captured", 
        "Face has been captured and uploaded to cloud storage.",
        [{ text: "OK" }]
      );
    } catch (error) {
      Alert.alert("Error", "Failed to capture and upload face image.");
      console.error(error);
    } finally {
      setUploading(false);
    }
  };

  const captureFull = async () => {
    if (!currentFrame) return;
    
    try {
      setUploading(true);
      
      // Create a filename with timestamp
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const filename = `full_capture_${timestamp}.jpg`;
      
      // Extract the base64 data from the image string
      const base64Data = currentFrame.split(',')[1];
      
      // Create a URI for the image
      const imageUri = `data:image/jpeg;base64,${base64Data}`;
      
      // Upload to Firebase
      const downloadURL = await uploadImage(imageUri, filename);
      
      Alert.alert(
        "Image Captured", 
        "Full frame has been captured and uploaded to cloud storage.",
        [{ text: "OK" }]
      );
    } catch (error) {
      Alert.alert("Error", "Failed to capture and upload image.");
      console.error(error);
    } finally {
      setUploading(false);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.statusBar}>
        {scanning ? (
          <Text style={styles.scanningText}>Scanning for camera server...</Text>
        ) : (
          <>
            <Text style={[styles.statusText, { color: connected ? 'green' : 'red' }]}>
              {connected ? 'Connected' : 'Disconnected'}
            </Text>
            <Text style={styles.facesText}>Faces: {faceData.length}</Text>
          </>
        )}
      </View>
      
      <View style={styles.feedContainer}>
        {currentFrame ? (
          <Image
            source={{ uri: currentFrame }}
            style={styles.cameraFeed}
            resizeMode="contain"
          />
        ) : (
          <View style={styles.noFeedContainer}>
            <Text style={styles.noFeedText}>Waiting for camera feed...</Text>
          </View>
        )}
        
        {/* Render face indicators */}
        {faceData.map((face, index) => (
          <TouchableOpacity
            key={index}
            style={[
              styles.faceIndicator,
              {
                left: face.x,
                top: face.y,
                width: face.width,
                height: face.height,
              }
            ]}
            onPress={() => captureFace(index)}
            disabled={uploading}
          >
            <Text style={styles.faceText}>{index + 1}</Text>
          </TouchableOpacity>
        ))}
      </View>
      
      <View style={styles.buttonContainer}>
        <TouchableOpacity 
          style={[styles.button, uploading && styles.buttonDisabled]} 
          onPress={captureFull}
          disabled={uploading || !currentFrame}
        >
          <Text style={styles.buttonText}>
            {uploading ? 'Uploading...' : 'Capture Full Frame'}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  statusBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 10,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
  },
  statusText: {
    fontWeight: 'bold',
  },
  facesText: {
    fontWeight: 'bold',
  },
  feedContainer: {
    flex: 1,
    position: 'relative',
  },
  cameraFeed: {
    flex: 1,
    width: '100%',
    height: '100%',
  },
  noFeedContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
  },
  noFeedText: {
    color: '#fff',
    fontSize: 16,
  },
  faceIndicator: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#00ff00',
    backgroundColor: 'transparent',
    justifyContent: 'center',
    alignItems: 'center',
  },
  faceText: {
    color: '#00ff00',
    fontWeight: 'bold',
    fontSize: 16,
  },
  buttonContainer: {
    padding: 15,
  },
  button: {
    backgroundColor: '#2196F3',
    padding: 15,
    borderRadius: 5,
    alignItems: 'center',
  },
  buttonDisabled: {
    backgroundColor: '#b0c4de',
  },
  buttonText: {
    color: 'white',
    fontWeight: 'bold',
  },
  scanningText: {
    fontWeight: 'bold',
    color: '#ff9900',
  },
});
