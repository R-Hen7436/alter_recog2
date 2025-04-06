import React, { useState, useRef, useEffect } from "react";
import { View, Text, StyleSheet, TouchableOpacity, Animated } from "react-native";
import MapView, { Marker, Circle, Polygon } from "react-native-maps";
import * as Location from "expo-location";
import { ref, set, get } from "firebase/database";
import { db } from "./firebaseConfig";
import { useNavigation } from '@react-navigation/native';

export default function LocationTracker() {
  const navigation = useNavigation();
  const mapRef = useRef(null);
  const [currentLocation, setCurrentLocation] = useState(null);
  const [isFetchingLocation, setIsFetchingLocation] = useState(false);
  const [locationButtonText, setLocationButtonText] = useState("My Location");
  const [initialRegion, setInitialRegion] = useState(null);
  const [finalLocation, setFinalLocation] = useState(null);
  const [initialLocation, setInitialLocation] = useState(null);
  const [heading, setHeading] = useState(0);
  const pulseAnimation = useRef(new Animated.Value(1)).current;

  // Start pulse animation - subtle pulsing effect
  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnimation, {
          toValue: 1.05,
          duration: 2000,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnimation, {
          toValue: 1,
          duration: 2000,
          useNativeDriver: true,
        }),
      ])
    ).start();
  }, []);

  // Start heading subscription
  useEffect(() => {
    let headingSubscription = null;

    const startHeadingTracking = async () => {
      try {
        // Request permission to use the magnetometer
        const { status } = await Location.requestForegroundPermissionsAsync();
        if (status !== 'granted') {
          console.error('Permission to access location was denied');
          return;
        }

        // Start watching heading updates
        headingSubscription = await Location.watchHeadingAsync((headingData) => {
          setHeading(headingData.trueHeading || headingData.magHeading || 0);
        });
      } catch (error) {
        console.error('Error starting heading tracking:', error);
      }
    };

    startHeadingTracking();

    return () => {
      if (headingSubscription) {
        headingSubscription.remove();
      }
    };
  }, []);

  // Function to fetch and check final coordinates
  const fetchFinalCoordinates = async () => {
    try {
      const dbRef = ref(db, "UserCurrentLocation");
      const snapshot = await get(dbRef);
      const data = snapshot.val();
      
      if (data?.Final_latitude && data?.Final_longitude) {
        return {
          latitude: data.Final_latitude,
          longitude: data.Final_longitude
        };
      }
      return null;
    } catch (error) {
      console.error("Error fetching final coordinates:", error);
      return null;
    }
  };

  // Get initial location when app opens
  useEffect(() => {
    const getInitialLocation = async () => {
      setIsFetchingLocation(true);
      try {
        let { status } = await Location.requestForegroundPermissionsAsync();
        if (status !== "granted") {
          console.error("Permission to access location was denied");
          return;
        }

        let location = await Location.getCurrentPositionAsync({
          accuracy: Location.Accuracy.BestForNavigation
        });
        const { latitude, longitude, heading: currentHeading } = location.coords;
        
        // Set heading if available from GPS
        if (currentHeading) {
          setHeading(currentHeading);
        }
        
        // Save initial location
        const initialCoords = { latitude, longitude };
        setInitialLocation(initialCoords);

        // Save to Firebase
        const dbRef = ref(db, "UserCurrentLocation");
        await set(dbRef, { 
          latitude: latitude,
          longitude: longitude,
          heading: heading
        });

        // Check for final coordinates (but don't use them yet)
        const finalCoords = await fetchFinalCoordinates();
        setFinalLocation(finalCoords);

        // MODIFIED: Always use phone's location instead of final coordinates
        // const displayCoords = finalCoords || initialCoords;
        const displayCoords = initialCoords; // Always use phone's location
        
        setCurrentLocation(displayCoords);
        setLocationButtonText("Remove Loc");

        // Set initial region
        if (!initialRegion) {
          const region = {
            ...displayCoords,
            latitudeDelta: 0.01,
            longitudeDelta: 0.01,
          };
          setInitialRegion(region);
          mapRef.current?.animateToRegion(region);
        }
      } catch (error) {
        console.error("Error getting initial location:", error);
      }
      setIsFetchingLocation(false);
    };

    getInitialLocation();
  }, []);

  // Real-time location tracking
  useEffect(() => {
    let locationSubscription = null;

    const startLocationTracking = async () => {
      if (currentLocation) {
        locationSubscription = await Location.watchPositionAsync(
          {
            accuracy: Location.Accuracy.BestForNavigation,
            timeInterval: 1000,
            distanceInterval: 0.5,
          },
          async (location) => {
            const { latitude, longitude, heading: currentHeading } = location.coords;
            
            // Update heading if available from GPS
            if (currentHeading) {
              setHeading(currentHeading);
            }
            
            try {
              // Update coordinates in Firebase
              const dbRef = ref(db, "UserCurrentLocation");
              const snapshot = await get(dbRef);
              const existingData = snapshot.val() || {};
              
              await set(dbRef, {
                ...existingData, // Preserve any existing final coordinates
                latitude: latitude,
                longitude: longitude,
                heading: heading,
                timestamp: new Date().toISOString()
              });

              // Force a complete re-render by creating a new object
              const newLocation = { latitude, longitude };
              
              // Important: Set state with new objects to trigger re-renders
              setInitialLocation(newLocation);
              setCurrentLocation(newLocation);

              // Check for final coordinates
              const finalCoords = await fetchFinalCoordinates();
              if (finalCoords) {
                setFinalLocation({ ...finalCoords });
              }
              
            } catch (error) {
              console.error("Error updating location:", error);
            }
          }
        );
      }
    };

    startLocationTracking();
    return () => {
      if (locationSubscription) {
        locationSubscription.remove();
      }
    };
  }, [currentLocation, heading]);

  // Toggle location tracking
  const toggleCurrentLocation = async () => {
    if (currentLocation) {
      setCurrentLocation(null);
      setLocationButtonText("My Location");
      console.log("Location tracking stopped");
    } else {
      setIsFetchingLocation(true);
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== "granted") {
        console.error("Permission to access location was denied");
        setIsFetchingLocation(false);
        return;
      }
      try {
        let location = await Location.getCurrentPositionAsync({
          accuracy: Location.Accuracy.BestForNavigation
        });
        const { latitude, longitude, heading: currentHeading } = location.coords;
        
        // Update heading if available
        if (currentHeading) {
          setHeading(currentHeading);
        }
        
        setCurrentLocation({ latitude, longitude });
        setLocationButtonText("Remove Loc");
        
        // Animate to new location when toggling
        if (mapRef.current) {
          mapRef.current.animateToRegion({
            latitude,
            longitude,
            latitudeDelta: 0.01,
            longitudeDelta: 0.01,
          });
        }
      } catch (error) {
        console.error("Error fetching location:", error);
      }
      setIsFetchingLocation(false);
    }
  };

  // Calculate direction beam coordinates
  const getDirectionBeamCoordinates = (center, heading, distance = 0.0003) => {
    // Convert heading to radians (0° is north, increases clockwise)
    const headingRad = (heading * Math.PI) / 180;
    
    // Calculate beam width (40 degrees total, 20 degrees on each side)
    const beamWidthRad = (20 * Math.PI) / 180;
    
    // Calculate the three points of the beam triangle
    const point1 = center; // The user's location
    
    // Calculate point 2 (left side of beam)
    const leftAngle = headingRad - beamWidthRad;
    const point2 = {
      latitude: center.latitude + distance * Math.cos(leftAngle),
      longitude: center.longitude + distance * Math.sin(leftAngle),
    };
    
    // Calculate point 3 (right side of beam)
    const rightAngle = headingRad + beamWidthRad;
    const point3 = {
      latitude: center.latitude + distance * Math.cos(rightAngle),
      longitude: center.longitude + distance * Math.sin(rightAngle),
    };
    
    return [point1, point2, point3];
  };

  // Custom Google Maps style marker with direction beam
  const GoogleMapsStyleMarker = ({ coordinate }) => {
    // Calculate beam coordinates
    const beamCoordinates = getDirectionBeamCoordinates(coordinate, heading);
    
    // Generate a unique key based on the coordinates
    const markerKey = `${coordinate.latitude}-${coordinate.longitude}-${heading}`;
    
    return (
      <>
        {/* Accuracy circle - light blue area (smaller and lighter) */}
        <Animated.View 
          style={{
            opacity: 0.15,
            transform: [{ scale: pulseAnimation }]
          }}
          key={`circle-${markerKey}`}
        >
          <Circle
            center={coordinate}
            radius={5} // Smaller radius
            fillColor="#81D4FA" // Lighter blue color
            strokeColor="transparent"
          />
        </Animated.View>
        
        {/* Direction beam */}
        <Polygon
          key={`beam-${markerKey}`}
          coordinates={beamCoordinates}
          fillColor="rgba(129, 212, 250, 0.6)" // Lighter blue with more opacity
          strokeColor="rgba(129, 212, 250, 0.8)" // Lighter blue stroke
          strokeWidth={1}
        />
        
        {/* Blue dot */}
        <Circle
          key={`dot-${markerKey}`}
          center={coordinate}
          radius={5} // Slightly smaller dot
          fillColor="#2196F3" // Standard blue
          strokeColor="white"
          strokeWidth={2}
        />
      </>
    );
  };

  if (!db) {
    console.error("Firebase database not initialized");
    return;
  }

  return (
    <View style={styles.container}>
      <MapView 
        ref={mapRef} 
        style={styles.map}
        initialRegion={initialRegion}
        showsUserLocation={false} // Disable default user location dot
      >
        {/* Phone location with Google Maps style */}
        {initialLocation && (
          <GoogleMapsStyleMarker 
            coordinate={initialLocation}
            key={`marker-${initialLocation.latitude}-${initialLocation.longitude}-${heading}`}
          />
        )}
        
        {/* Final location marker (if available) */}
        {finalLocation && (
          <Marker 
            coordinate={finalLocation} 
            title="Indoor Location (Final)" 
            pinColor="red" 
            key={`final-${finalLocation.latitude}-${finalLocation.longitude}`}
          />
        )}
      </MapView>

      <TouchableOpacity 
        style={styles.buttonSecondary} 
        onPress={toggleCurrentLocation}
      >
        <Text style={styles.buttonText}>
          {isFetchingLocation ? "Loading..." : locationButtonText}
        </Text>
      </TouchableOpacity>
      
      <TouchableOpacity 
        style={styles.wifiButton} 
        onPress={() => navigation.navigate('WiFiNetworks')}
      >
        <Text style={styles.buttonText}>Wi-Fi Networks</Text>
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
  buttonSecondary: {
    position: "absolute",
    bottom: 20,
    left: "50%",
    transform: [{ translateX: -75 }],
    backgroundColor: "green",
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 10,
    width: 150,
    alignItems: "center",
    justifyContent: "center",
  },
  buttonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "bold",
  },
  wifiButton: {
    position: "absolute",
    bottom: 80, // Position above the location button
    left: "50%",
    transform: [{ translateX: -75 }],
    backgroundColor: "#2196F3",
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 10,
    width: 150,
    alignItems: "center",
    justifyContent: "center",
  }
});
