import React, { useState, useEffect } from "react";
import { View, Text, StyleSheet, ActivityIndicator } from "react-native";
import MapView, { Marker, Polyline } from "react-native-maps";
import * as Location from "expo-location";

export default function App() {
  const [location, setLocation] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [region, setRegion] = useState({
    latitude: 10.237003,
    longitude: 123.775338,
    latitudeDelta: 0.002,
    longitudeDelta: 0.002,
  });

  useEffect(() => {
    (async () => {
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== "granted") {
        setErrorMsg("Permission to access location was denied");
        return;
      }

      let loc = await Location.getCurrentPositionAsync({});
      setLocation(loc.coords);
      setRegion({
        latitude: loc.coords.latitude,
        longitude: loc.coords.longitude,
        latitudeDelta: 0.002,
        longitudeDelta: 0.002,
      });
    })();
  }, []);

  // Define Diamond Shape Points
  const shapeCoordinates = [
    { latitude: 10.237803, longitude: 123.775338 }, // Top Point
    { latitude: 10.237003, longitude: 123.776138 }, // Right Point
    { latitude: 10.236203, longitude: 123.775338 }, // Bottom Point
    { latitude: 10.237003, longitude: 123.774538 }, // Left Point
    { latitude: 10.237803, longitude: 123.775338 }, // Back to Top Point (to close the shape)
  ];

  return (
    <View style={styles.container}>
      {region ? (
        <>
          <MapView
            style={styles.map}
            initialRegion={region}
            region={region}
            showsUserLocation={true}
            followsUserLocation={true}
          >
            {/* Marker for user's location */}
            {location && (
              <Marker
                coordinate={{
                  latitude: location.latitude,
                  longitude: location.longitude,
                }}
                title="You are here"
                description="Your current location"
              />
            )}

            {/* Polyline to create a diamond shape */}
            <Polyline
              coordinates={shapeCoordinates}
              strokeColor="red"
              strokeWidth={3}
            />
          </MapView>

          {/* Display Latitude & Longitude in a Floating Box */}
          <View style={styles.locationBox}>
            {location ? (
              <Text style={styles.locationText}>
                📍 Latitude: {location.latitude.toFixed(6)}
                {"\n"}📍 Longitude: {location.longitude.toFixed(6)}
              </Text>
            ) : (
              <Text style={styles.locationText}>Fetching location...</Text>
            )}
          </View>
        </>
      ) : (
        <ActivityIndicator size="large" color="#0000ff" />
      )}
      {errorMsg && <Text style={styles.error}>{errorMsg}</Text>}
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
  locationBox: {
    position: "absolute",
    bottom: 20,
    left: "50%",
    transform: [{ translateX: -100 }],
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    padding: 10,
    borderRadius: 10,
    width: 250,
    alignItems: "center",
  },
  locationText: {
    color: "white",
    fontSize: 16,
    fontWeight: "bold",
    textAlign: "center",
  },
  error: {
    fontSize: 18,
    color: "red",
    textAlign: "center",
    marginTop: 20,
  },
});

