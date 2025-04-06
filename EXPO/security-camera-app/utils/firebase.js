import { initializeApp } from 'firebase/app';
import { getStorage, ref, uploadBytes, getDownloadURL, listAll, deleteObject } from 'firebase/storage';

// Your existing Firebase config
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

// Upload an image to Firebase Storage
export const uploadImage = async (uri, filename) => {
  try {
    const response = await fetch(uri);
    const blob = await response.blob();
    const storageRef = ref(storage, `security_faces/${filename}`);
    
    await uploadBytes(storageRef, blob);
    const downloadURL = await getDownloadURL(storageRef);
    console.log("Image uploaded successfully:", downloadURL);
    return downloadURL;
  } catch (error) {
    console.error("Error uploading image:", error);
    throw error;
  }
};

// List all images in the faces folder
export const listFaceImages = async () => {
  try {
    const listRef = ref(storage, 'security_faces');
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