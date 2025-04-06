import { initializeApp } from 'firebase/app';
import { getStorage, ref, uploadBytes, getDownloadURL, listAll, deleteObject } from 'firebase/storage';

// Your existing Firebase config - this should be the same as the one in Components/firebaseConfig.js
// This configuration will be used specifically for face detection features
const firebaseConfig = {
  apiKey: "AIzaSyALi0028_ngjDDmAFc0BfW5WYnCKsd5W3c",
  authDomain: "geofencing-2fcd0.firebaseapp.com",
  databaseURL: "https://geofencing-2fcd0-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "geofencing-2fcd0",
  storageBucket: "geofencing-2fcd0.appspot.com",
  messagingSenderId: "439986173789",
  appId: "1:439986173789:web:def850b445ffcb0d1adab4"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const storage = getStorage(app);

// Upload an image to Firebase Storage directly from base64
export const uploadImage = async (uri, filename) => {
  try {
    console.log("Starting image upload for:", filename);
    
    // Extract base64 data from URI if it's a data URI
    let base64Data;
    if (uri.startsWith('data:')) {
      base64Data = uri.split(',')[1];
    } else {
      // If it's not a data URI, we need to fetch it and convert to base64
      try {
        const response = await fetch(uri);
        const blob = await response.blob();
        return await uploadBlob(blob, filename);
      } catch (fetchError) {
        console.error("Error fetching image:", fetchError);
        throw new Error(`Network error while fetching image: ${fetchError.message}`);
      }
    }
    
    // Continue with upload using base64 string directly
    const storageRef = ref(storage, `location_faces/${filename}`);
    
    // Convert base64 to Uint8Array for upload
    const binary = atob(base64Data);
    const array = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      array[i] = binary.charCodeAt(i);
    }
    
    // Create metadata
    const metadata = {
      contentType: 'image/jpeg',
    };
    
    // Upload the Uint8Array
    const snapshot = await uploadBytes(storageRef, array, metadata);
    console.log('Uploaded an array!');
    
    // Get download URL
    const downloadURL = await getDownloadURL(snapshot.ref);
    console.log("Image uploaded successfully:", downloadURL);
    
    return downloadURL;
  } catch (error) {
    console.error("Error uploading image:", error);
    throw error;
  }
};

// Helper function to upload a blob
const uploadBlob = async (blob, filename) => {
  try {
    const storageRef = ref(storage, `location_faces/${filename}`);
    const snapshot = await uploadBytes(storageRef, blob);
    const downloadURL = await getDownloadURL(snapshot.ref);
    console.log("Blob uploaded successfully:", downloadURL);
    return downloadURL;
  } catch (error) {
    console.error("Error uploading blob:", error);
    throw error;
  }
};

// Check if a file exists in storage
export const checkFileExists = async (filename) => {
  try {
    const fileRef = ref(storage, `location_faces/${filename}`);
    await getDownloadURL(fileRef);
    return true; // File exists
  } catch (error) {
    if (error.code === 'storage/object-not-found') {
      return false; // File doesn't exist
    }
    console.error("Error checking if file exists:", error);
    throw error;
  }
};

// Get information about all uploaded faces 
export const getUploadedFacesInfo = async () => {
  try {
    const facesRef = ref(storage, 'location_faces');
    const result = await listAll(facesRef);
    
    return {
      totalFiles: result.items.length,
      fileNames: result.items.map(item => item.name)
    };
  } catch (error) {
    console.error("Error getting uploaded files info:", error);
    throw error;
  }
};

// List all images in the faces folder with full details
export const listFaceImages = async () => {
  try {
    const listRef = ref(storage, 'location_faces');
    const result = await listAll(listRef);
    
    const urls = await Promise.all(
      result.items.map(async (itemRef) => {
        const url = await getDownloadURL(itemRef);
        return {
          name: itemRef.name,
          url,
          fullPath: itemRef.fullPath,
        };
      })
    );
    
    return urls;
  } catch (error) {
    console.error("Error listing images:", error);
    throw error;
  }
};

// Delete an image from Firebase Storage
export const deleteImage = async (fullPath) => {
  try {
    const imageRef = ref(storage, fullPath);
    await deleteObject(imageRef);
    console.log("Image deleted successfully");
    return true;
  } catch (error) {
    console.error("Error deleting image:", error);
    throw error;
  }
}; 