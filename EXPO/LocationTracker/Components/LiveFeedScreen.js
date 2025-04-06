import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, Image, Alert } from 'react-native';
import io from 'socket.io-client';
import { uploadImage } from '../utils/firebase';
import Zeroconf from 'react-native-zeroconf';
import { WebView } from 'react-native-webview';

export default function LiveFeedScreen() {
  const [connected, setConnected] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [nextFrame, setNextFrame] = useState(null); // For double buffering
  const [isFrameLoaded, setIsFrameLoaded] = useState(false);
  const [activeImageIndex, setActiveImageIndex] = useState(0); // Track which image is active
  const [faceData, setFaceData] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [serverAddress, setServerAddress] = useState(null);
  const [scanning, setScanning] = useState(true);
  const socketRef = useRef(null);
  const zeroconfRef = useRef(null);
  const [lastFrameTime, setLastFrameTime] = useState(0);
  const minFrameInterval = 50; // Minimum 50ms between frames (maximum 20 FPS)
  const [streamQuality, setStreamQuality] = useState('medium'); // 'low', 'medium', 'high'
  const [imageOpacity, setImageOpacity] = useState(1);
  const [frameBuffer, setFrameBuffer] = useState([]);
  const frameProcessorRef = useRef(null);
  const [frameData, setFrameData] = useState(null); // Single source of truth
  const [isReady, setIsReady] = useState(false);    // Track when the component is fully initialized
  const imageElement1 = useRef(null);
  const imageElement2 = useRef(null);
  const [visibleImageIndex, setVisibleImageIndex] = useState(0);
  const lastFrameReceived = useRef('');
  const frameTimer = useRef(null);

  // Initialize zeroconf and scan for the server
  useEffect(() => {
    try {
      // Create a slight delay before initializing Zeroconf
      setTimeout(() => {
        try {
          // Initialize Zeroconf with error handling
          const zeroconf = new Zeroconf();
          zeroconfRef.current = zeroconf;
          
          zeroconf.on('resolved', service => {
            if (service.name === 'FaceDetectionServer') {
              const host = service.addresses[0];
              const port = service.port;
              setServerAddress(`http://${host}:${port}`);
              setScanning(false);
            }
          });
          
          zeroconf.on('error', err => {
            console.error('Zeroconf error:', err);
            setScanning(false);
            Alert.alert(
              "Connection Error",
              "Couldn't find the face detection server. Please connect manually."
            );
          });
          
          // Add error handling for scan
          try {
            zeroconf.scan('http', 'tcp', 'local.');
            console.log('Started Zeroconf scan successfully');
          } catch (scanError) {
            console.error('Error scanning:', scanError);
            setScanning(false);
            // Fall back to manual connection
            setServerAddress('http://192.168.1.4:5000'); // Updated to your actual server IP
          }
        } catch (initError) {
          console.error('Error initializing Zeroconf:', initError);
          setScanning(false);
          // Fall back to manual connection
          setServerAddress('http://192.168.1.4:5000'); // Updated to your actual server IP
        }
      }, 1000); // Add a 1 second delay
      
      return () => {
        if (zeroconfRef.current) {
          try {
            zeroconfRef.current.stop();
          } catch (error) {
            console.error('Error stopping Zeroconf:', error);
          }
        }
      };
    } catch (error) {
      console.error('Top level error in Zeroconf initialization:', error);
      setScanning(false);
      // Fall back to manual connection
      setServerAddress('http://192.168.1.4:5000'); // Updated to your actual server IP
    }
  }, []);

  // Connect to socket.io server once we have the address
  useEffect(() => {
    if (!serverAddress) return;
    
    console.log(`Connecting to server at ${serverAddress}`);
    const socket = io(serverAddress, {
      transports: ['websocket'],  // Force WebSocket transport
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000
    });
    socketRef.current = socket;
    
    socket.on('connect', () => {
      setConnected(true);
      console.log('Connected to camera server');
      
      // Request video stream on connection
      socket.emit('request_stream');
      console.log('Requested video stream');
    });
    
    socket.on('disconnect', () => {
      setConnected(false);
      console.log('Disconnected from camera server');
    });
    
    // Handle face detection events (may contain face data)
    socket.on('face_detected', (data) => {
      if (data && data.image) {
        setCurrentFrame(`data:image/jpeg;base64,${data.image}`);
        if (data.faces) {
          setFaceData(data.faces);
          console.log(`Detected ${data.faces.length} faces`);
        }
      }
    });
    
    // Handle continuous frame updates
    socket.on('frame_update', (data) => {
      if (data && data.image && isReady) {
        const now = Date.now();
        if (now - lastFrameTime >= 150) { // Keep the rate limit
          // Store the latest frame data but don't render it immediately
          lastFrameReceived.current = `data:image/jpeg;base64,${data.image}`;
          setLastFrameTime(now);
          
          // Update face data if present
          if (data.faces) {
            setFaceData(data.faces);
          }
          
          // Schedule frame processing if not already scheduled
          if (!frameTimer.current) {
            frameTimer.current = setTimeout(processNewFrame, 10);
          }
        }
      }
    });
    
    return () => {
      if (socket) {
        console.log('Disconnecting socket');
        socket.disconnect();
      }
    };
  }, [serverAddress, isReady]);

  // Add this useEffect to handle image loaded events and swap buffers
  useEffect(() => {
    if (isFrameLoaded) {
      // Toggle which image is displayed to swap between buffers
      setActiveImageIndex(prev => prev === 0 ? 1 : 0);
      setIsFrameLoaded(false);
    }
  }, [isFrameLoaded]);

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
        "Person Captured", 
        "Person has been captured and uploaded to cloud storage.",
        [{ text: "OK" }]
      );
    } catch (error) {
      Alert.alert("Error", "Failed to capture and upload image.");
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

  const requestQualityChange = (quality) => {
    if (socketRef.current && socketRef.current.connected) {
      socketRef.current.emit('set_quality', { quality });
      setStreamQuality(quality);
    }
  };

  // Add this cleanup logic to prevent memory leaks
  useEffect(() => {
    // Cleanup function to run when component unmounts
    return () => {
      // Clear any pending frame processing
      if (frameProcessorRef.current) {
        clearTimeout(frameProcessorRef.current);
      }
      
      // Disconnect socket
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
      
      // Clear frame data to free memory
      setCurrentFrame(null);
      setNextFrame(null);
      setFaceData([]);
    };
  }, []);

  // Add this function to process new frames with better timing control
  const processNewFrame = () => {
    // Clear the timer reference
    frameTimer.current = null;
    
    // Only update if we have a new frame
    if (lastFrameReceived.current) {
      // Update the invisible image first
      const nextIndex = visibleImageIndex === 0 ? 1 : 0;
      const nextRef = nextIndex === 0 ? imageElement1 : imageElement2;
      
      // Use a setTimeout to ensure the DOM has a chance to update
      setFrameData({
        uri: lastFrameReceived.current,
        timestamp: Date.now()
      });
      
      // Switch the visible image after a short delay
      setTimeout(() => {
        setVisibleImageIndex(nextIndex);
      }, 50); // Small delay to ensure the image is loaded
    }
  };

  // Add this effect to mark when the component is ready for frame processing
  useEffect(() => {
    const readyTimer = setTimeout(() => {
      setIsReady(true);
    }, 500); // Give time for initial setup
    
    return () => clearTimeout(readyTimer);
  }, []);

  return (
    <View style={styles.container}>
      <View style={styles.statusBar}>
        {scanning ? (
          <>
            <Text style={styles.scanningText}>Scanning for camera server...</Text>
            <TouchableOpacity 
              style={styles.manualButton}
              onPress={() => {
                setServerAddress('http://192.168.1.4:5000'); // Updated to your actual server IP
                setScanning(false);
              }}
            >
              <Text style={styles.buttonText}>Connect Manually</Text>
            </TouchableOpacity>
          </>
        ) : (
          <>
            <Text style={[styles.statusText, { color: connected ? 'green' : 'red' }]}>
              {connected ? 'Connected' : 'Disconnected'}
            </Text>
            <Text style={styles.facesText}>Persons: {faceData.length}</Text>
          </>
        )}
      </View>
      
      <View style={styles.feedContainer}>
        {frameData ? (
          <>
            <Image
              ref={imageElement1}
              source={{ uri: frameData.uri }}
              style={[
                styles.cameraFeed,
                {
                  opacity: visibleImageIndex === 0 ? 1 : 0,
                  zIndex: visibleImageIndex === 0 ? 2 : 1
                }
              ]}
              resizeMode="contain"
              fadeDuration={0}
            />
            <Image
              ref={imageElement2}
              source={{ uri: frameData.uri }}
              style={[
                styles.cameraFeed,
                {
                  opacity: visibleImageIndex === 1 ? 1 : 0,
                  zIndex: visibleImageIndex === 1 ? 2 : 1
                }
              ]}
              resizeMode="contain"
              fadeDuration={0}
            />
          </>
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

      <View style={styles.qualityControls}>
        <Text style={styles.controlLabel}>Stream Quality:</Text>
        <View style={styles.qualityButtons}>
          <TouchableOpacity 
            style={[styles.qualityButton, streamQuality === 'low' && styles.activeQuality]} 
            onPress={() => requestQualityChange('low')}
          >
            <Text style={styles.qualityText}>Low</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.qualityButton, streamQuality === 'medium' && styles.activeQuality]} 
            onPress={() => requestQualityChange('medium')}
          >
            <Text style={styles.qualityText}>Medium</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.qualityButton, streamQuality === 'high' && styles.activeQuality]} 
            onPress={() => requestQualityChange('high')}
          >
            <Text style={styles.qualityText}>High</Text>
          </TouchableOpacity>
        </View>
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
    backgroundColor: '#000',
    overflow: 'hidden',
    backfaceVisibility: 'hidden',
  },
  cameraFeed: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: '#000',
    width: '100%',
    height: '100%',
    borderWidth: 0,
    backfaceVisibility: 'hidden',
    transform: [{ perspective: 1000 }],
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
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#ddd',
  },
  button: {
    backgroundColor: '#2196F3',
    padding: 15,
    borderRadius: 5,
    alignItems: 'center',
  },
  buttonDisabled: {
    backgroundColor: '#cccccc',
  },
  buttonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
  scanningText: {
    fontStyle: 'italic',
    color: '#666',
  },
  manualButton: {
    backgroundColor: '#4CAF50',
    padding: 8,
    borderRadius: 5,
    marginTop: 10,
  },
  qualityControls: {
    marginTop: 10,
    marginBottom: 10,
  },
  controlLabel: {
    fontSize: 14,
    marginBottom: 5,
  },
  qualityButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  qualityButton: {
    padding: 8,
    borderRadius: 4,
    backgroundColor: '#e0e0e0',
    marginRight: 8,
  },
  activeQuality: {
    backgroundColor: '#2196F3',
  },
  qualityText: {
    fontSize: 12,
    color: '#333',
  },
  activeQualityText: {
    color: 'white',
  },
}); 