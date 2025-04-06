import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, Switch } from 'react-native';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import { ref, set, serverTimestamp } from 'firebase/database';
import { db } from './firebaseConfig';
import AsyncStorage from '@react-native-async-storage/async-storage';

const SENSOR_UPDATE_INTERVAL = 1000; // Update every 1 second

export default function MovementTracker() {
  const [isTracking, setIsTracking] = useState(false);
  const [userId, setUserId] = useState(null);
  const [sensorData, setSensorData] = useState({
    accelerometer: { x: 0, y: 0, z: 0 },
    gyroscope: { x: 0, y: 0, z: 0 }
  });

  // Load user ID on component mount
  useEffect(() => {
    AsyncStorage.getItem('user_device_id').then(id => setUserId(id));
  }, []);

  // Handle sensor subscriptions
  useEffect(() => {
    let accelerometerSubscription = null;
    let gyroscopeSubscription = null;

    if (isTracking && userId) {
      // Configure update intervals
      Accelerometer.setUpdateInterval(SENSOR_UPDATE_INTERVAL);
      Gyroscope.setUpdateInterval(SENSOR_UPDATE_INTERVAL);

      // Subscribe to accelerometer
      accelerometerSubscription = Accelerometer.addListener(accelerometerData => {
        setSensorData(prev => ({
          ...prev,
          accelerometer: accelerometerData
        }));
      });

      // Subscribe to gyroscope
      gyroscopeSubscription = Gyroscope.addListener(gyroscopeData => {
        setSensorData(prev => ({
          ...prev,
          gyroscope: gyroscopeData
        }));
      });
    }

    // Cleanup subscriptions
    return () => {
      if (accelerometerSubscription) {
        accelerometerSubscription.remove();
      }
      if (gyroscopeSubscription) {
        gyroscopeSubscription.remove();
      }
    };
  }, [isTracking, userId]);

  // Update Firebase when sensor data changes
  useEffect(() => {
    if (isTracking && userId) {
      const updateFirebase = async () => {
        try {
          const sensorRef = ref(db, `device_sensors/${userId}`);
          await set(sensorRef, {
            ...sensorData,
            timestamp: serverTimestamp(),
            isMoving: isDeviceMoving(sensorData),
          });
        } catch (error) {
          console.error('Error updating sensor data:', error);
        }
      };

      updateFirebase();
    }
  }, [sensorData, isTracking, userId]);

  // Detect if device is moving based on sensor data
  const isDeviceMoving = (data) => {
    const accelerationThreshold = 0.1; // Adjust this value based on testing
    const rotationThreshold = 0.1;     // Adjust this value based on testing

    // Check acceleration magnitude
    const accelerationMagnitude = Math.sqrt(
      Math.pow(data.accelerometer.x, 2) +
      Math.pow(data.accelerometer.y, 2) +
      Math.pow(data.accelerometer.z, 2)
    );

    // Check rotation magnitude
    const rotationMagnitude = Math.sqrt(
      Math.pow(data.gyroscope.x, 2) +
      Math.pow(data.gyroscope.y, 2) +
      Math.pow(data.gyroscope.z, 2)
    );

    return (
      Math.abs(accelerationMagnitude - 9.81) > accelerationThreshold || // 9.81 is gravity
      rotationMagnitude > rotationThreshold
    );
  };

  // Format sensor values for display
  const formatSensorValue = (value) => {
    return value.toFixed(3);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Movement Tracking</Text>

      <View style={styles.switchContainer}>
        <Text style={styles.switchLabel}>Enable Tracking</Text>
        <Switch
          value={isTracking}
          onValueChange={setIsTracking}
        />
      </View>

      {isTracking && (
        <View style={styles.dataContainer}>
          <Text style={styles.sectionTitle}>Accelerometer:</Text>
          <Text style={styles.sensorValue}>
            X: {formatSensorValue(sensorData.accelerometer.x)}
          </Text>
          <Text style={styles.sensorValue}>
            Y: {formatSensorValue(sensorData.accelerometer.y)}
          </Text>
          <Text style={styles.sensorValue}>
            Z: {formatSensorValue(sensorData.accelerometer.z)}
          </Text>

          <Text style={styles.sectionTitle}>Gyroscope:</Text>
          <Text style={styles.sensorValue}>
            X: {formatSensorValue(sensorData.gyroscope.x)}
          </Text>
          <Text style={styles.sensorValue}>
            Y: {formatSensorValue(sensorData.gyroscope.y)}
          </Text>
          <Text style={styles.sensorValue}>
            Z: {formatSensorValue(sensorData.gyroscope.z)}
          </Text>

          <Text style={[
            styles.movementStatus,
            { color: isDeviceMoving(sensorData) ? '#e74c3c' : '#2ecc71' }
          ]}>
            Device is {isDeviceMoving(sensorData) ? 'Moving' : 'Stationary'}
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
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  switchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 20,
    padding: 15,
    backgroundColor: '#f5f5f5',
    borderRadius: 10,
  },
  switchLabel: {
    fontSize: 18,
    fontWeight: '500',
  },
  dataContainer: {
    backgroundColor: '#f5f5f5',
    padding: 15,
    borderRadius: 10,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginTop: 15,
    marginBottom: 10,
  },
  sensorValue: {
    fontSize: 16,
    marginBottom: 5,
  },
  movementStatus: {
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
    marginTop: 20,
    padding: 10,
    backgroundColor: '#fff',
    borderRadius: 8,
  },
}); 