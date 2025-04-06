import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, Alert, ActivityIndicator, ScrollView, Modal, Image, FlatList } from 'react-native';
import io from 'socket.io-client';
import { uploadImage, getUploadedImagesInfo, listImages, trackUpload } from '../utils/imgbbStorage';
import { WebView } from 'react-native-webview';

export default function LiveFeedScreen() {
  const [connected, setConnected] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [faceData, setFaceData] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [serverAddress, setServerAddress] = useState('http://192.168.1.4:5000');
  const [streamUrl, setStreamUrl] = useState(null);
  const [loadingWebView, setLoadingWebView] = useState(true);
  const [checkingStorage, setCheckingStorage] = useState(false);
  const [storageInfo, setStorageInfo] = useState(null);
  const [showStorageModal, setShowStorageModal] = useState(false);
  const [uploadedImages, setUploadedImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const socketRef = useRef(null);
  const webViewRef = useRef(null);
  const [forceHideLoading, setForceHideLoading] = useState(false);
  
  useEffect(() => {
    console.log('Using server address:', serverAddress);
    connectToServer();
    
    return () => {
      disconnectServer();
    };
  }, []);

  const connectToServer = () => {
    console.log(`Connecting to server at ${serverAddress}`);
    const socket = io(serverAddress, {
      transports: ['websocket'],
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
      
      // Set quality to medium immediately after connection
      socket.emit('set_quality', { quality: 'medium' });
    });
    
    socket.on('disconnect', () => {
      setConnected(false);
      console.log('Disconnected from camera server');
    });
    
    socket.on('face_detected', (data) => {
      if (data && data.image) {
        setCurrentFrame(`data:image/jpeg;base64,${data.image}`);
        if (data.faces) {
          setFaceData(data.faces);
          console.log(`Detected ${data.faces.length} faces`);
        }
      }
    });
    
    socket.on('stream_acknowledged', (data) => {
      console.log('Stream acknowledged:', data);
      if (data.stream_url) {
        setStreamUrl(data.stream_url);
      }
    });
    
    socket.on('frame_update', (data) => {
      if (data && data.faces) {
        setFaceData(data.faces);
      }
    });
  };

  const disconnectServer = () => {
    if (socketRef.current) {
      console.log('Disconnecting socket');
      socketRef.current.disconnect();
    }
  };

  const changeServerIP = () => {
    const newIP = prompt("Enter server IP address:", "192.168.1.4");
    if (newIP) {
      disconnectServer();
      setServerAddress(`http://${newIP}:5000`);
      setTimeout(connectToServer, 500);
    }
  };

  const captureFrame = async () => {
    if (!socketRef.current || !socketRef.current.connected) {
      Alert.alert('Connection Error', 'Not connected to detection server');
      return;
    }
    
    setUploading(true);
    
    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      
      console.log('Capturing frame...');
      socketRef.current.emit('request_capture');
      
      // Add timeout for capture request
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Capture request timed out')), 10000);
      });
      
      // Wait for the capture result
      const resultPromise = new Promise((resolve) => {
        socketRef.current.once('capture_result', resolve);
      });
      
      // Race between timeout and result
      const captureData = await Promise.race([resultPromise, timeoutPromise]);
      
      if (captureData.error) {
        throw new Error(captureData.error);
      }
      
      if (!captureData.image) {
        throw new Error('No image data received');
      }
      
      // Upload the captured frame
      const filename = `capture_${timestamp}.jpg`;
      const imageUri = `data:image/jpeg;base64,${captureData.image}`;
      
      console.log('Uploading captured frame...');
      const result = await uploadImage(imageUri, filename);
      trackUpload(result);
      
      console.log('Successfully uploaded frame to ImgBB:', result.url);
      
      Alert.alert(
        "Frame Captured",
        "Image has been captured and uploaded to cloud storage. You can view it by clicking 'View Uploaded Images'.",
        [{ text: "OK" }]
      );
    } catch (error) {
      console.error("Error in capture process:", error);
      Alert.alert("Error", `Failed to capture image: ${error.message}`);
    } finally {
      setUploading(false);
    }
  };

  // Function to check ImgBB storage
  const checkStorage = async () => {
    try {
      setCheckingStorage(true);
      
      const info = getUploadedImagesInfo();
      setStorageInfo(info);
      
      const images = listImages();
      setUploadedImages(images);
      
      setShowStorageModal(true);
    } catch (error) {
      console.error("Error checking storage:", error);
      Alert.alert("Storage Check Error", `Could not check storage: ${error.message}`);
    } finally {
      setCheckingStorage(false);
    }
  };

  // Add the WebView message handler function
  const handleWebViewMessage = (event) => {
    try {
      const data = JSON.parse(event.nativeEvent.data);
      
      if (data.type === 'contentCheck' && data.hasContent) {
        setLoadingWebView(false);
        console.log('Content detected in WebView');
      }
      
      if (data.type === 'pageLoaded') {
        setLoadingWebView(false);
        console.log('WebView page loaded');
      }
    } catch (error) {
      console.log('Error parsing WebView message:', error);
    }
  };

  // Add the WebView injection script
  const WEBVIEW_INJECT_SCRIPT = `
    (function() {
      window.ReactNativeWebView.postMessage(JSON.stringify({
        type: 'pageStartedLoading',
        time: Date.now()
      }));

      function checkContent() {
        const hasContent = document.body && 
          (document.getElementsByTagName('video').length > 0 ||
           document.getElementsByTagName('img').length > 0 ||
           document.getElementsByTagName('canvas').length > 0);

        window.ReactNativeWebView.postMessage(JSON.stringify({
          type: 'contentCheck',
          hasContent: hasContent,
          time: Date.now()
        }));

        if (!hasContent) {
          setTimeout(checkContent, 500);
        }
      }

      checkContent();

      window.addEventListener('load', function() {
        window.ReactNativeWebView.postMessage(JSON.stringify({
          type: 'pageLoaded',
          time: Date.now()
        }));
        checkContent();
      });

      true;
    })();
  `;

  // Add useEffect for loading indicator timeout
  useEffect(() => {
    if (streamUrl) {
      const hideTimer = setTimeout(() => {
        setForceHideLoading(true);
        setLoadingWebView(false);
      }, 5000); // Force hide after 5 seconds
      
      return () => clearTimeout(hideTimer);
    }
  }, [streamUrl]);

  return (
    <View style={styles.container}>
      <View style={styles.statusBar}>
        <Text style={[styles.statusText, { color: connected ? 'green' : 'red' }]}>
          {connected ? 'Connected' : 'Disconnected'}
        </Text>
        <Text style={styles.facesText}>Persons: {faceData.length}</Text>
        {!connected && (
          <TouchableOpacity 
            style={styles.manualButton}
            onPress={changeServerIP}
          >
            <Text style={styles.buttonText}>Change IP</Text>
          </TouchableOpacity>
        )}
      </View>
      
      <View style={styles.feedContainer}>
        {streamUrl ? (
          <>
            <WebView
              ref={webViewRef}
              source={{ uri: streamUrl }}
              style={styles.webView}
              onLoadStart={() => {
                if (!forceHideLoading) {
                  setLoadingWebView(true);
                }
              }}
              onLoad={() => setLoadingWebView(false)}
              onLoadEnd={() => setLoadingWebView(false)}
              onError={(e) => {
                console.log('WebView error:', e.nativeEvent);
                setLoadingWebView(false);
              }}
              onMessage={handleWebViewMessage}
              injectedJavaScript={WEBVIEW_INJECT_SCRIPT}
              javaScriptEnabled={true}
              domStorageEnabled={true}
              allowsInlineMediaPlayback={true}
              mediaPlaybackRequiresUserAction={false}
              scrollEnabled={false}
              bounces={false}
              originWhitelist={['*']}
              mixedContentMode="always"
              androidHardwareAccelerationDisabled={false}
              androidLayerType="hardware"
            />
            
            {(streamUrl && !forceHideLoading && loadingWebView) && (
              <TouchableOpacity 
                style={styles.loadingContainer}
                activeOpacity={0.9}
                onPress={() => {
                  setForceHideLoading(true);
                  setLoadingWebView(false);
                }}
              >
                <ActivityIndicator size="large" color="#2196F3" />
                <Text style={styles.loadingText}>Loading video stream...</Text>
                <Text style={styles.tapToDismissText}>Tap anywhere to dismiss</Text>
              </TouchableOpacity>
            )}
          </>
        ) : (
          <View style={styles.noFeedContainer}>
            <Text style={styles.noFeedText}>Waiting for camera stream...</Text>
            {!connected && (
              <TouchableOpacity 
                style={styles.retryButton}
                onPress={() => {
                  if (serverAddress) {
                    setServerAddress(null);
                    setTimeout(() => setServerAddress(serverAddress), 100);
                  } else {
                    setServerAddress('http://192.168.1.4:5000');
                  }
                }}
              >
                <Text style={styles.buttonText}>Retry Connection</Text>
              </TouchableOpacity>
            )}
          </View>
        )}
      </View>
      
      <View style={styles.buttonContainer}>
        <TouchableOpacity 
          style={[styles.button, uploading && styles.buttonDisabled]} 
          onPress={captureFrame}
          disabled={uploading || !streamUrl}
        >
          <Text style={styles.buttonText}>
            {uploading ? 'Uploading...' : 'Capture'}
          </Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.button, styles.storageButton, checkingStorage && styles.buttonDisabled]} 
          onPress={checkStorage}
          disabled={checkingStorage}
        >
          <Text style={styles.buttonText}>
            {checkingStorage ? 'Checking Storage...' : 'View Uploaded Images'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Storage Info Modal - remains unchanged */}
      <Modal
        visible={showStorageModal}
        animationType="slide"
        transparent={false}
        onRequestClose={() => setShowStorageModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Uploaded Images (ImgBB Storage)</Text>
            <TouchableOpacity
              style={styles.closeButton}
              onPress={() => setShowStorageModal(false)}
            >
              <Text style={styles.closeButtonText}>Close</Text>
            </TouchableOpacity>
          </View>
          
          {storageInfo && (
            <View style={styles.storageInfo}>
              <View style={styles.storageInfoRow}>
                <Text style={styles.storageInfoText}>
                  Total Files: {storageInfo.totalFiles}
                </Text>
                <TouchableOpacity 
                  style={styles.refreshButton}
                  onPress={async () => {
                    await checkStorage();
                  }}
                >
                  <Text style={styles.refreshButtonText}>Refresh</Text>
                </TouchableOpacity>
              </View>
              <Text style={styles.storageInfoSubtext}>
                Images are stored on ImgBB's free cloud storage
              </Text>
            </View>
          )}
          
          <FlatList
            data={uploadedImages}
            keyExtractor={(item, index) => `${item.name}-${index}`}
            numColumns={2}
            renderItem={({ item }) => (
              <TouchableOpacity 
                style={styles.gridImageItem}
                onPress={() => {
                  setSelectedImage(item);
                }}
              >
                <Image
                  source={{ uri: item.thumbnail || item.url }}
                  style={styles.gridThumbnail}
                  resizeMode="cover"
                />
                <View style={styles.gridImageDetails}>
                  <Text style={styles.gridImageName} numberOfLines={1} ellipsizeMode="middle">
                    {item.name.length > 20 ? item.name.substring(0, 18) + '...' : item.name}
                  </Text>
                </View>
              </TouchableOpacity>
            )}
            contentContainerStyle={styles.imageGridList}
            ListEmptyComponent={
              <View style={styles.emptyList}>
                <Text style={styles.emptyText}>No images uploaded yet. Capture some faces first!</Text>
              </View>
            }
          />
        </View>
      </Modal>

      {/* Add this modal for viewing selected images */}
      <Modal
        visible={selectedImage !== null}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setSelectedImage(null)}
      >
        <TouchableOpacity 
          style={styles.fullImageModalContainer} 
          activeOpacity={1} 
          onPress={() => setSelectedImage(null)}
        >
          <View style={styles.fullImageModalContent}>
            <View style={styles.fullImageHeader}>
              <Text style={styles.fullImageTitle} numberOfLines={1}>
                {selectedImage?.name || 'Image Preview'}
              </Text>
              <TouchableOpacity 
                style={styles.closeFullImageButton}
                onPress={() => setSelectedImage(null)}
              >
                <Text style={styles.closeButtonText}>Close</Text>
              </TouchableOpacity>
            </View>
            <View style={styles.fullImageWrapper}>
              {selectedImage && (
                <Image
                  source={{ uri: selectedImage.url }}
                  style={styles.fullImage}
                  resizeMode="contain"
                />
              )}
            </View>
          </View>
        </TouchableOpacity>
      </Modal>
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
  },
  webView: {
    flex: 1,
    backgroundColor: '#000',
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
  loadingContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.7)',
    zIndex: 10,
  },
  loadingText: {
    color: '#fff',
    marginTop: 10,
  },
  buttonContainer: {
    padding: 15,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#ddd',
    gap: 10,
  },
  button: {
    backgroundColor: '#2196F3',
    padding: 15,
    borderRadius: 5,
    alignItems: 'center',
  },
  storageButton: {
    backgroundColor: '#FF9800',
  },
  buttonDisabled: {
    backgroundColor: '#cccccc',
  },
  buttonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
  manualButton: {
    backgroundColor: '#4CAF50',
    padding: 8,
    borderRadius: 5,
    marginTop: 10,
  },
  tapToDismissText: {
    color: '#fff',
    marginTop: 15,
    opacity: 0.8,
    fontSize: 14,
  },
  retryButton: {
    backgroundColor: '#4CAF50',
    padding: 10,
    borderRadius: 5,
    marginTop: 15,
  },
  modalContainer: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 15,
    backgroundColor: '#2196F3',
  },
  modalTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  closeButton: {
    padding: 8,
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 4,
  },
  closeButtonText: {
    color: 'white',
    fontWeight: 'bold',
  },
  storageInfo: {
    padding: 15,
    backgroundColor: 'white',
    marginBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
  },
  storageInfoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 5,
  },
  storageInfoText: {
    fontSize: 16,
    color: '#333',
    fontWeight: 'bold',
  },
  storageInfoSubtext: {
    fontSize: 14,
    color: '#666',
  },
  refreshButton: {
    backgroundColor: '#2196F3',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 4,
  },
  refreshButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 12,
  },
  
  // Grid view for images
  imageGridList: {
    padding: 5,
  },
  gridImageItem: {
    flex: 1,
    margin: 5,
    backgroundColor: 'white',
    borderRadius: 8,
    overflow: 'hidden',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 1.5,
    maxWidth: '48%',
  },
  gridThumbnail: {
    width: '100%',
    height: 150,
    borderTopLeftRadius: 8,
    borderTopRightRadius: 8,
  },
  gridImageDetails: {
    padding: 8,
  },
  gridImageName: {
    fontSize: 12,
    fontWeight: '500',
    textAlign: 'center',
  },
  
  emptyList: {
    padding: 50,
    alignItems: 'center',
  },
  emptyText: {
    fontSize: 16,
    color: '#666',
    fontStyle: 'italic',
  },
  viewButton: {
    marginTop: 5,
    padding: 5,
    backgroundColor: '#2196F3',
    borderRadius: 4,
    alignSelf: 'flex-start',
  },
  viewButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '500',
  },
  fullImageModalContainer: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  fullImageBackdrop: {
    flex: 1,
    width: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  fullImageModalContent: {
    backgroundColor: 'white',
    borderRadius: 10,
    width: '90%',
    height: '80%',
    overflow: 'hidden',
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.3,
    shadowRadius: 5,
  },
  fullImageHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
    backgroundColor: '#2196F3',
  },
  fullImageTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: 'white',
    flex: 1,
    paddingRight: 10,
  },
  closeFullImageButton: {
    padding: 8,
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 4,
  },
  fullImageWrapper: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 10,
    backgroundColor: '#000',
  },
  fullImage: {
    width: '100%',
    height: '100%',
  },
  imageActionBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 12,
    backgroundColor: '#333',
  },
  imageTimestamp: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  imageActionButton: {
    paddingVertical: 5,
    paddingHorizontal: 10,
    backgroundColor: '#2196F3',
    borderRadius: 4,
  },
  actionButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
}); 