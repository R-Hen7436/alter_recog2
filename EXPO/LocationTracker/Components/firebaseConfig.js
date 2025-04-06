import { initializeApp } from "firebase/app";
import { getDatabase, ref, onValue } from "firebase/database"; // Import Realtime Database

// Your Firebase configuration
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

// Initialize Realtime Database and get a reference to the service
const db = getDatabase(app);

// Add error handling for database connection
const connectedRef = ref(db, '.info/connected');
onValue(connectedRef, (snap) => {
  if (snap.val() === true) {
    console.log('Connected to Firebase');
  } else {
    console.log('Not connected to Firebase');
  }
});

export { db };
