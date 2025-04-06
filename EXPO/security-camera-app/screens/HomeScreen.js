import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Alert } from 'react-native';
import io from 'socket.io-client';
import Zeroconf from 'react-native-zeroconf';

export default function HomeScreen({ navigation }) {
  const [connected, setConnected] = useState(false);
  const [lastDetection, setLastDetection] = useState(null);
  const [faceCount, setFaceCount] = useState(0);
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
      console.log('Connected to server');
    });
    
    socketRef.current.on('disconnect', () => {
      setConnected(false);
      console.log('Disconnected from server');
    });
    
    socketRef.current.on('connect_error', (error) => {
      console.log('Connection error:', error);
      Alert.alert(
        "Connection Error",
        "Cannot connect to server. Check your WiFi connection and make sure the server is running."
      );
    });
    
    socketRef.current.on('face_detected', (data) => {
      console.log('Face detected:', data);
      setFaceCount(data.count);
      setLastDetection(new Date().toLocaleTimeString());
      
      // If faces are detected, show alert and offer to navigate to live feed
      Alert.alert(
        "Face Detected",
        `${data.count} face(s) detected at ${new Date().toLocaleTimeString()}`,
        [
          {
            text: "View Live Feed",
            onPress: () => navigation.navigate('LiveFeed')
          },
          {
            text: "Dismiss",
            style: "cancel"
          }
        ]
      );
    });
    
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, [serverAddress, navigation]);

  return (
    <View style={styles.container}>
      <View style={styles.statusContainer}>
        <Text style={styles.title}>Security Camera Status</Text>
        {scanning ? (
          <Text style={styles.scanningText}>Scanning for camera server...</Text>
        ) : (
          <Text style={[styles.status, { color: connected ? 'green' : 'red' }]}>
            {connected ? 'Connected' : 'Disconnected'}
          </Text>
        )}
        
        {lastDetection && (
          <View style={styles.detectionInfo}>
            <Text style={styles.infoText}>Last detection: {lastDetection}</Text>
            <Text style={styles.infoText}>Faces detected: {faceCount}</Text>
          </View>
        )}
      </View>
      
      <View style={styles.menuContainer}>
        <TouchableOpacity 
          style={styles.menuButton} 
          onPress={() => navigation.navigate('LiveFeed')}
        >
          <Text style={styles.buttonText}>View Live Feed</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={styles.menuButton} 
          onPress={() => navigation.navigate('CapturedFaces')}
        >
          <Text style={styles.buttonText}>View Captured Faces</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={styles.menuButton} 
          onPress={() => navigation.navigate('Settings')}
        >
          <Text style={styles.buttonText}>Settings</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
  statusContainer: {
    marginBottom: 30,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  status: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  detectionInfo: {
    marginTop: 15,
    padding: 15,
    backgroundColor: '#f0f0f0',
    borderRadius: 10,
  },
  infoText: {
    fontSize: 16,
    marginBottom: 5,
  },
  menuContainer: {
    flex: 1,
  },
  menuButton: {
    backgroundColor: '#2196F3',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 15,
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  scanningText: {
    fontSize: 16,
    color: '#666',
    fontStyle: 'italic',
  },
});