import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ActivityIndicator } from 'react-native';
import * as Network from 'expo-network';
import * as Device from 'expo-device';
import { ref, set, get, push } from 'firebase/database';
import { db } from './firebaseConfig';
import AsyncStorage from '@react-native-async-storage/async-storage';

// This function can be called automatically on app startup
export const registerDeviceMacAddress = async () => {
  try {
    // Check if we've already registered this device
    const storedUserId = await AsyncStorage.getItem('user_device_id');
    
    // Get the best MAC address or device identifier we can
    const deviceInfo = await getBestDeviceIdentifier();
    
    if (!deviceInfo.identifier) {
      console.error('Could not get device identifier');
      return false;
    }
    
    if (storedUserId) {
      // Update existing user device
      await updateDeviceInFirebase(storedUserId, deviceInfo);
    } else {
      // First time registration
      const userId = await registerNewDeviceInFirebase(deviceInfo);
      if (userId) {
        // Save the user ID locally for future reference
        await AsyncStorage.setItem('user_device_id', userId);
      }
    }
    
    return true;
  } catch (error) {
    console.error('Error registering device:', error);
    return false;
  }
};

// Get the best device identifier available
const getBestDeviceIdentifier = async () => {
  // Try to get network info first
  try {
    const networkInfo = await Network.getNetworkStateAsync();
    
    // On some Android devices, this might work
    if (networkInfo?.type === Network.NetworkStateType.WIFI) {
      if (networkInfo.details?.ipAddress) {
        return {
          identifier: networkInfo.details.ipAddress,
          type: 'ip_address',
          name: Device.deviceName || 'Unknown Device'
        };
      }
    }
  } catch (err) {
    console.log('Network info error:', err);
  }
  
  // Use device info to create a unique identifier
  try {
    // Create a unique device identifier
    const deviceId = Device.deviceId || '';
    const deviceModel = Device.modelName || '';
    const deviceName = Device.deviceName || 'Unknown Device';
    
    const uniqueId = `${deviceId}-${deviceModel.replace(/\s+/g, '')}-${
      deviceName.replace(/\s+/g, '').substring(0, 8)
    }`;
    
    return {
      identifier: uniqueId,
      type: 'device_id',
      name: deviceName
    };
  } catch (err) {
    console.error('Device info error:', err);
    
    // Last resort - generate a random ID
    return {
      identifier: `device-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`,
      type: 'generated',
      name: 'Unknown Device'
    };
  }
};

// Register a new device in Firebase
const registerNewDeviceInFirebase = async (deviceInfo) => {
  try {
    // First check if this device is already registered by identifier
    const devicesRef = ref(db, 'doorsentinel/devices');
    const snapshot = await get(devicesRef);
    
    if (snapshot.exists()) {
      const devices = snapshot.val();
      
      // Check if device already exists
      for (const userId in devices) {
        if (devices[userId].identifier === deviceInfo.identifier) {
          console.log('Device already registered, using existing ID');
          return userId;
        }
      }
    }
    
    // If not found, create new entry
    const newDeviceRef = push(devicesRef);
    const userId = newDeviceRef.key;
    
    // Set device data
    await set(newDeviceRef, {
      identifier: deviceInfo.identifier,
      type: deviceInfo.type,
      name: deviceInfo.name,
      lastSeen: new Date().toISOString(),
      active: true
    });
    
    // Also set as the current target for backward compatibility
    const targetRef = ref(db, 'doorsentinel/targetMacAddress');
    await set(targetRef, deviceInfo.identifier);
    
    console.log('Device registered with ID:', userId);
    return userId;
  } catch (error) {
    console.error('Error registering device in Firebase:', error);
    return null;
  }
};

// Update existing device data
const updateDeviceInFirebase = async (userId, deviceInfo) => {
  try {
    const deviceRef = ref(db, `doorsentinel/devices/${userId}`);
    
    // Update only what's needed
    await set(deviceRef, {
      identifier: deviceInfo.identifier,
      type: deviceInfo.type,
      name: deviceInfo.name,
      lastSeen: new Date().toISOString(),
      active: true
    });
    
    // Also update target MAC address for backward compatibility
    const targetRef = ref(db, 'doorsentinel/targetMacAddress');
    await set(targetRef, deviceInfo.identifier);
    
    console.log('Device updated:', userId);
    return true;
  } catch (error) {
    console.error('Error updating device in Firebase:', error);
    return false;
  }
};

// React component for displaying status (can be used or not)
export default function MacAddress() {
  const [deviceInfo, setDeviceInfo] = useState(null);
  const [userId, setUserId] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const initializeComponent = async () => {
      setIsLoading(true);
      
      try {
        // Get saved user ID
        const storedUserId = await AsyncStorage.getItem('user_device_id');
        setUserId(storedUserId);
        
        // Get device info
        const info = await getBestDeviceIdentifier();
        setDeviceInfo(info);
        
        // Make sure device is registered
        await registerDeviceMacAddress();
      } catch (error) {
        console.error('Error in MacAddress component:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    initializeComponent();
  }, []);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Device Registration</Text>
      
      {isLoading ? (
        <ActivityIndicator size="large" color="#0000ff" />
      ) : (
        <View style={styles.infoContainer}>
          <Text style={styles.label}>Device Type:</Text>
          <Text style={styles.value}>{deviceInfo?.type || 'Unknown'}</Text>
          
          <Text style={styles.label}>Device Name:</Text>
          <Text style={styles.value}>{deviceInfo?.name || 'Unknown'}</Text>
          
          <Text style={styles.label}>Device Identifier:</Text>
          <Text style={styles.value}>{deviceInfo?.identifier || 'Not available'}</Text>
          
          <Text style={styles.statusText}>
            ✓ Your device is registered with the door sentinel
          </Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  infoContainer: {
    backgroundColor: '#f5f5f5',
    padding: 15,
    borderRadius: 10,
    width: '100%',
  },
  label: {
    fontSize: 16,
    color: '#555',
    marginTop: 10,
  },
  value: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  statusText: {
    marginTop: 20,
    color: 'green',
    fontWeight: 'bold',
    textAlign: 'center',
  }
});
