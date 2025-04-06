import React, { useState, useRef, useEffect } from "react";
import { View, Text, StyleSheet, TouchableOpacity, TextInput, Alert } from "react-native";
import MapView, { Marker, Polygon, Circle } from "react-native-maps";
import * as Location from "expo-location";
import * as Network from 'expo-network';  // Make sure this is installed
import * as Device from 'expo-device';    // Make sure this is installed
import { getDatabase, ref, set, push, get, remove, child, onValue } from "firebase/database";
import { db } from "./firebaseConfig";

export default function App() {
  const mapRef = useRef(null);
  const [currentLocation, setCurrentLocation] = useState(null);
  const [isTracking, setIsTracking] = useState(false);
  const [deviceInfo, setDeviceInfo] = useState(null);
  const [initialRegion, setInitialRegion] = useState({
    latitude: 10.3157, // Cebu City coordinates
    longitude: 123.8854,
    latitudeDelta: 0.1,
    longitudeDelta: 0.1,
  });

  // Get device identifier on app start
  useEffect(() => {
    const getDeviceIdentifier = async () => {
      try {
        // Try to get network info first
        let identifier = null;
        let deviceType = 'unknown';
        
        try {
          const networkInfo = await Network.getNetworkStateAsync();
          
          // On some Android devices, this might work to get IP or MAC
          if (networkInfo?.type === Network.NetworkStateType.WIFI) {
            if (networkInfo.details?.ipAddress) {
              identifier = networkInfo.details.ipAddress;
              deviceType = 'ip_address';
            }
          }
        } catch (err) {
          console.log('Network info error:', err);
        }
        
        // If we couldn't get network info, use device info
        if (!identifier) {
          // Create a unique device identifier
          const deviceId = Device.deviceId || '';
          const deviceModel = Device.modelName || '';
          const deviceName = Device.deviceName || 'Unknown Device';
          
          identifier = `${deviceId}-${deviceModel.replace(/\s+/g, '')}-${
            deviceName.replace(/\s+/g, '').substring(0, 8)
          }`;
          deviceType = 'device_id';
        }
        
        setDeviceInfo({
          identifier: identifier,
          type: deviceType,
          name: Device.deviceName || 'Unknown Device'
        });
        
        // Save to Firebase
        const doorRef = ref(db, "doorsentinel/targetMacAddress");
        await set(doorRef, identifier);
        
        console.log('Device identifier saved:', identifier);
        
      } catch (error) {
        console.error('Error getting device identifier:', error);
      }
    };

    getDeviceIdentifier();
  }, []);

  // Initial location setup
  useEffect(() => {
    const getInitialLocation = async () => {
      try {
        let { status } = await Location.requestForegroundPermissionsAsync();
        if (status !== "granted") {
          console.error("Permission to access location was denied");
          return;
        }

        if (isTracking) {
          let location = await Location.getCurrentPositionAsync({});
          const { latitude, longitude } = location.coords;
          
          setCurrentLocation({ latitude, longitude });

          if (mapRef.current) {
            mapRef.current.animateToRegion({
              latitude,
              longitude,
              latitudeDelta: 0.01,
              longitudeDelta: 0.01,
            });
          }

          // Initial location save to Firebase
          const dbRef = ref(db, "UsersCurrentLocation");
          await set(dbRef, { Latitude: latitude, Longitude: longitude });
        }
      } catch (error) {
        console.error("Error getting initial location:", error);
      }
    };

    getInitialLocation();
  }, [isTracking]);

  // Location tracking
  useEffect(() => {
    let locationSubscription = null;

    const startLocationTracking = async () => {
      if (!isTracking) return;

      locationSubscription = await Location.watchPositionAsync(
        {
          accuracy: Location.Accuracy.BestForNavigation,
          timeInterval: 1000,    // 1 second between updates
          distanceInterval: 0.1, // Update every 0.1 meters
          mayShowUserSettingsDialog: true // Prompts user to enable high accuracy
        },
        async (location) => {
          const { latitude, longitude } = location.coords;
          
          // Remove the accuracy check to see all updates
          console.log('Location Update:', {
            latitude,
            longitude,
            accuracy: location.coords.accuracy,
            timestamp: new Date().toISOString()
          });
          
          setCurrentLocation({ latitude, longitude });
          
          try {
            const dbRef = ref(db, "UsersCurrentLocation");
            await set(dbRef, { 
              Latitude: latitude, 
              Longitude: longitude,
              Accuracy: location.coords.accuracy,
              Timestamp: new Date().toISOString()
            });
          } catch (error) {
            console.error("Error updating location in Firebase:", error);
          }
        }
      );
    };

    startLocationTracking();

    return () => {
      if (locationSubscription) {
        locationSubscription.remove();
      }
    };
  }, [currentLocation, isTracking]);

  // Add a useEffect to verify tracking state
  useEffect(() => {
    console.log('Tracking state changed:', isTracking);
  }, [isTracking]);

  // Helper function for significant movement check
  const isSignificantMove = (prevLocation, newLocation) => {
    if (!prevLocation) return true;
    const distance = calculateDistance(
      { latitude: prevLocation.latitude, longitude: prevLocation.longitude },
      { latitude: newLocation.latitude, longitude: newLocation.longitude }
    );
    return distance > 5;
  };

  return (
    <View style={styles.container}>
      <MapView 
        ref={mapRef} 
        style={styles.map} 
        initialRegion={initialRegion}
        showsUserLocation={isTracking}
        followsUserLocation={isTracking}
        userLocationUpdateInterval={1000}
        showsMyLocationButton={true}
        userLocationAnnotationTitle=""
        userLocationCalloutEnabled={false}
        mapType="standard"
        customMapStyle={{
          userLocationAnnotationFillColor: 'rgba(0, 122, 255, 0.3)',
          userLocationAnnotationStrokeColor: 'rgba(0, 122, 255, 0.8)',
          userLocationAnnotationStrokeWidth: 2,
        }}
      >
      </MapView>
      
      {/* Device identifier info */}
      {deviceInfo && (
        <View style={styles.deviceInfoContainer}>
          <Text style={styles.deviceInfoText}>
            ID: {deviceInfo.identifier ? 
              deviceInfo.identifier.substring(0, 10) + '...' : 
              'Not available'}
          </Text>
        </View>
      )}
      
      <TouchableOpacity 
        style={styles.trackingButton}
        onPress={() => setIsTracking(!isTracking)}
      >
        <Text style={styles.buttonText}>
          {isTracking ? 'Stop Tracking' : 'RealtimeTracking'}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  map: {
    width: "100%",
    height: "100%",
  },
  deviceInfoContainer: {
    position: 'absolute',
    top: 60,
    alignSelf: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 10,
    borderRadius: 20,
  },
  deviceInfoText: {
    color: 'white',
    fontSize: 12,
  },
  trackingButton: {
    position: 'absolute',
    bottom: 30,
    alignSelf: 'center',
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 10,
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
