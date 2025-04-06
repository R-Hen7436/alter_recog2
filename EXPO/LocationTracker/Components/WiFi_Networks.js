import 'react-native-get-random-values';
import React, { useState, useEffect } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  TouchableOpacity, 
  FlatList,
  Modal,
  TextInput,
  Alert,
  ActivityIndicator
} from 'react-native';
import * as Location from 'expo-location';
import * as Network from 'expo-network';
import { ref, set, get, push, onValue } from 'firebase/database';
import { db } from './firebaseConfig';
import { v4 as uuidv4 } from 'uuid';
import { useNavigation } from '@react-navigation/native';

export default function WiFi_Networks() {
  const navigation = useNavigation();
  const [calibrationMode, setCalibrationMode] = useState(false);
  const [calibrationModalVisible, setCalibrationModalVisible] = useState(false);
  const [locationName, setLocationName] = useState('');
  const [currentLocation, setCurrentLocation] = useState(null);
  const [calibrationPoints, setCalibrationPoints] = useState([]);
  const [calibrationInProgress, setCalibrationInProgress] = useState(false);
  const [calibrationStatus, setCalibrationStatus] = useState('');
  const [wifiNetworks, setWifiNetworks] = useState([]);
  const [loading, setLoading] = useState(false);

  // Request location permissions on component mount
  useEffect(() => {
    (async () => {
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Location permission is required for calibration');
      }
    })();

    // Load existing calibration points
    loadCalibrationPoints();
    
    // Listen for Raspberry Pi Wi-Fi scans
    const wifiRef = ref(db, 'RaspberryPi/wifi_scan');
    const unsubscribe = onValue(wifiRef, (snapshot) => {
      if (snapshot.exists()) {
        const data = snapshot.val();
        if (data.access_points) {
          setWifiNetworks(data.access_points);
        }
      }
    });
    
    // Listen for calibration status updates
    const calibrationRequestRef = ref(db, 'CalibrationRequests/current');
    const calibrationUnsubscribe = onValue(calibrationRequestRef, (snapshot) => {
      if (snapshot.exists()) {
        const data = snapshot.val();
        if (data.status === 'completed') {
          setCalibrationInProgress(false);
          Alert.alert('Calibration Complete', `Successfully calibrated "${data.location_name}"`);
          loadCalibrationPoints();
        } else if (data.status === 'failed') {
          setCalibrationInProgress(false);
          Alert.alert('Calibration Failed', data.error || 'Unknown error occurred');
        } else if (data.status === 'pending') {
          setCalibrationStatus(`Calibrating ${data.location_name}...`);
        }
      }
    });
    
    return () => {
      unsubscribe();
      calibrationUnsubscribe();
    };
  }, []);

  // Load existing calibration points from Firebase
  const loadCalibrationPoints = async () => {
    setLoading(true);
    try {
      const calibrationRef = ref(db, 'Calibration');
      const snapshot = await get(calibrationRef);
      
      if (snapshot.exists()) {
        const data = snapshot.val();
        const points = Object.keys(data).map(key => ({
          id: key,
          name: data[key].name || 'Unnamed Location',
          timestamp: data[key].timestamp
        }));
        
        setCalibrationPoints(points);
      }
    } catch (error) {
      console.error('Error loading calibration points:', error);
    } finally {
      setLoading(false);
    }
  };

  // Start calibration process for a location
  const startCalibration = async () => {
    if (!locationName.trim()) {
      Alert.alert('Error', 'Please enter a location name');
      return;
    }
    
    setCalibrationInProgress(true);
    setCalibrationModalVisible(false);
    
    try {
      // Get current location with high accuracy
      let location = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.Highest
      });
      
      // Create a calibration request
      const locationId = uuidv4().replace(/-/g, '');
      const requestRef = ref(db, 'CalibrationRequests/current');
      
      await set(requestRef, {
        location_id: locationId,
        location_name: locationName,
        latitude: location.coords.latitude,
        longitude: location.coords.longitude,
        timestamp: new Date().toISOString(),
        status: 'pending'
      });
      
      setCalibrationStatus(`Calibrating ${locationName}...`);
      
    } catch (error) {
      console.error('Error starting calibration:', error);
      Alert.alert('Calibration Error', 'Failed to start calibration: ' + error.message);
      setCalibrationInProgress(false);
    }
  };

  // Delete a calibration point
  const deleteCalibrationPoint = async (id) => {
    try {
      const calibrationRef = ref(db, `Calibration/${id}`);
      await set(calibrationRef, null);
      
      setCalibrationPoints(prev => prev.filter(point => point.id !== id));
      Alert.alert('Success', 'Calibration point deleted');
    } catch (error) {
      console.error('Error deleting calibration point:', error);
      Alert.alert('Error', 'Failed to delete calibration point');
    }
  };

  // Toggle calibration mode
  const toggleCalibrationMode = () => {
    setCalibrationMode(!calibrationMode);
    if (!calibrationMode) {
      loadCalibrationPoints();
    }
  };

  // Render a Wi-Fi network item
  const renderWifiItem = ({ item }) => (
    <View style={styles.networkItem}>
      <View style={styles.networkInfo}>
        <Text style={styles.networkName}>{item.ssid || 'Unknown Network'}</Text>
        <Text style={styles.networkDetails}>
          BSSID: {item.bssid || 'Unknown'} • Signal: {item.rssi} dBm
        </Text>
        <Text style={styles.networkDetails}>
          Frequency: {item.frequency ? `${item.frequency} MHz` : 'Unknown'}
        </Text>
      </View>
      <View style={[styles.signalStrength, { backgroundColor: getSignalColor(item.rssi) }]}>
        <Text style={styles.signalText}>{getSignalBars(item.rssi)}</Text>
      </View>
    </View>
  );

  // Render a calibration point item
  const renderCalibrationItem = ({ item }) => (
    <View style={styles.calibrationItem}>
      <View style={styles.calibrationInfo}>
        <Text style={styles.calibrationName}>{item.name}</Text>
        <Text style={styles.calibrationDetails}>
          {new Date(item.timestamp).toLocaleString()}
        </Text>
      </View>
      <TouchableOpacity 
        style={styles.deleteButton}
        onPress={() => {
          Alert.alert(
            'Delete Calibration Point',
            `Are you sure you want to delete "${item.name}"?`,
            [
              { text: 'Cancel', style: 'cancel' },
              { text: 'Delete', onPress: () => deleteCalibrationPoint(item.id), style: 'destructive' }
            ]
          );
        }}
      >
        <Text style={styles.deleteButtonText}>Delete</Text>
      </TouchableOpacity>
    </View>
  );

  // Helper function to get signal strength color
  const getSignalColor = (rssi) => {
    if (rssi >= -65) return '#4CAF50';
    if (rssi >= -75) return '#FFC107';
    return '#F44336';
  };

  // Helper function to get signal bars representation
  const getSignalBars = (rssi) => {
    if (rssi >= -65) return '●●●●';
    if (rssi >= -75) return '●●●○';
    if (rssi >= -85) return '●●○○';
    return '●○○○';
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>
          {calibrationMode ? 'Calibration Mode' : 'Wi-Fi Networks'}
        </Text>
        <TouchableOpacity 
          style={[styles.modeButton, calibrationMode && styles.calibrationModeActive]} 
          onPress={toggleCalibrationMode}
        >
          <Text style={styles.modeButtonText}>
            {calibrationMode ? 'Exit Calibration' : 'Calibration Mode'}
          </Text>
        </TouchableOpacity>
      </View>

      <View style={styles.infoBox}>
        <Text style={styles.infoText}>
          Wi-Fi networks are being scanned by your Raspberry Pi. This app is used to manage calibration points.
        </Text>
      </View>

      <TouchableOpacity 
        style={styles.mapButton}
        onPress={() => navigation.navigate('LocationTracker')}
      >
        <Text style={styles.mapButtonText}>View Location Map</Text>
      </TouchableOpacity>

      {calibrationMode ? (
        <>
          <View style={styles.calibrationHeader}>
            <Text style={styles.calibrationHeaderText}>Calibration Points</Text>
            <TouchableOpacity 
              style={styles.addButton}
              onPress={() => setCalibrationModalVisible(true)}
              disabled={calibrationInProgress}
            >
              <Text style={styles.addButtonText}>Add Location</Text>
            </TouchableOpacity>
          </View>

          {calibrationInProgress ? (
            <View style={styles.calibrationProgress}>
              <ActivityIndicator size="large" color="#0066cc" />
              <Text style={styles.calibrationProgressText}>
                {calibrationStatus}
              </Text>
            </View>
          ) : loading ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#0066cc" />
              <Text style={styles.loadingText}>Loading calibration points...</Text>
            </View>
          ) : calibrationPoints.length > 0 ? (
            <FlatList
              data={calibrationPoints}
              renderItem={renderCalibrationItem}
              keyExtractor={item => item.id}
              style={styles.list}
            />
          ) : (
            <View style={styles.emptyState}>
              <Text style={styles.emptyStateText}>
                No calibration points yet. Add locations to improve positioning accuracy.
              </Text>
            </View>
          )}
        </>
      ) : (
        <>
          <Text style={styles.sectionTitle}>Networks Detected by Raspberry Pi</Text>
          
          {wifiNetworks.length > 0 ? (
            <FlatList
              data={wifiNetworks}
              renderItem={renderWifiItem}
              keyExtractor={(item, index) => item.bssid || `network-${index}`}
              style={styles.list}
            />
          ) : (
            <View style={styles.emptyState}>
              <Text style={styles.emptyStateText}>
                No Wi-Fi networks detected by the Raspberry Pi yet. Make sure your Raspberry Pi is running and connected to Firebase.
              </Text>
            </View>
          )}
        </>
      )}

      {/* Calibration Modal */}
      <Modal
        animationType="slide"
        transparent={true}
        visible={calibrationModalVisible}
        onRequestClose={() => setCalibrationModalVisible(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Add Calibration Point</Text>
            
            <Text style={styles.modalLabel}>Location Name:</Text>
            <TextInput
              style={styles.modalInput}
              value={locationName}
              onChangeText={setLocationName}
              placeholder="e.g., Living Room, Kitchen"
              placeholderTextColor="#999"
            />
            
            <Text style={styles.modalDescription}>
              Stand in the center of the location you want to calibrate. 
              The Raspberry Pi will collect Wi-Fi scans to improve positioning accuracy.
            </Text>
            
            <View style={styles.modalButtons}>
              <TouchableOpacity 
                style={[styles.modalButton, styles.modalCancelButton]}
                onPress={() => {
                  setCalibrationModalVisible(false);
                  setLocationName('');
                }}
              >
                <Text style={styles.modalButtonText}>Cancel</Text>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={[styles.modalButton, styles.modalSaveButton]}
                onPress={startCalibration}
              >
                <Text style={styles.modalButtonText}>Start Calibration</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Camera Feature Navigation Buttons */}
      <View style={styles.cameraFeatureContainer}>
        <Text style={styles.sectionTitle}>Camera Features</Text>
        <TouchableOpacity 
          style={styles.featureButton}
          onPress={() => navigation.navigate('LiveFeed')}
        >
          <Text style={styles.buttonText}>Live Camera Feed</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={styles.featureButton}
          onPress={() => navigation.navigate('CapturedFaces')}
        >
          <Text style={styles.buttonText}>View Detected Persons</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    padding: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  modeButton: {
    backgroundColor: '#0066cc',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
  },
  calibrationModeActive: {
    backgroundColor: '#ff6600',
  },
  modeButtonText: {
    color: 'white',
    fontWeight: 'bold',
  },
  scanButton: {
    backgroundColor: '#0066cc',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 16,
  },
  scanningButton: {
    backgroundColor: '#999',
  },
  scanButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
  infoBox: {
    backgroundColor: '#e8f4fd',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#0066cc',
  },
  infoText: {
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
  },
  list: {
    flex: 1,
  },
  networkItem: {
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 16,
    marginBottom: 8,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 1.41,
  },
  networkInfo: {
    flex: 1,
  },
  networkName: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
    color: '#333',
  },
  networkDetails: {
    fontSize: 12,
    color: '#666',
  },
  signalStrength: {
    width: 50,
    height: 50,
    borderRadius: 25,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 8,
  },
  signalText: {
    color: 'white',
    fontWeight: 'bold',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#666',
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  emptyStateText: {
    textAlign: 'center',
    fontSize: 16,
    color: '#666',
  },
  locationInfo: {
    backgroundColor: '#e0e0e0',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  locationText: {
    fontSize: 14,
    color: '#333',
  },
  calibrationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  calibrationHeaderText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  addButton: {
    backgroundColor: '#4CAF50',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
  },
  addButtonText: {
    color: 'white',
    fontWeight: 'bold',
  },
  calibrationItem: {
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 16,
    marginBottom: 8,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 1.41,
  },
  calibrationInfo: {
    flex: 1,
  },
  calibrationName: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
    color: '#333',
  },
  calibrationDetails: {
    fontSize: 12,
    color: '#666',
  },
  deleteButton: {
    backgroundColor: '#F44336',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    marginLeft: 8,
  },
  deleteButtonText: {
    color: 'white',
    fontWeight: 'bold',
  },
  modalContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  modalContent: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 24,
    width: '85%',
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 16,
    color: '#333',
    textAlign: 'center',
  },
  modalLabel: {
    fontSize: 16,
    marginBottom: 8,
    color: '#333',
  },
  modalInput: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    marginBottom: 16,
  },
  modalDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 24,
    lineHeight: 20,
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  modalButton: {
    flex: 1,
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  modalCancelButton: {
    backgroundColor: '#ccc',
    marginRight: 8,
  },
  modalSaveButton: {
    backgroundColor: '#4CAF50',
    marginLeft: 8,
  },
  modalButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
  calibrationProgress: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  calibrationProgressText: {
    marginTop: 16,
    fontSize: 16,
    color: '#333',
    textAlign: 'center',
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginVertical: 12,
    color: '#333',
  },
  mapButton: {
    backgroundColor: '#2196F3',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 16,
    flexDirection: 'row',
    justifyContent: 'center',
  },
  mapButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
    marginLeft: 8,
  },
  cameraFeatureContainer: {
    marginTop: 20,
    padding: 15,
    backgroundColor: '#fff',
    borderRadius: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  featureButton: {
    backgroundColor: '#2196F3',
    padding: 15,
    borderRadius: 5,
    alignItems: 'center',
    marginBottom: 10,
  },
  buttonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
});
