import React, { useState, useEffect } from 'react';
import { View, TouchableOpacity, Text, StyleSheet } from 'react-native';
import { db } from './firebaseConfig';
import { ref, set, onValue } from 'firebase/database';

export default function Smartlock() {
  const [isLocked, setIsLocked] = useState(true);

  // Read initial lock state from Firebase
  useEffect(() => {
    const lockRef = ref(db, 'IOTs/Smartlock');
    onValue(lockRef, (snapshot) => {
      if (snapshot.exists()) {
        setIsLocked(snapshot.val() === 1);
      }
    });
  }, []);

  // Toggle lock status
  const toggleLock = async () => {
    try {
      const newStatus = !isLocked;
      const lockRef = ref(db, 'IOTs/Smartlock');
      await set(lockRef, newStatus ? 1 : 0);
      setIsLocked(newStatus);
      console.log(`Door ${newStatus ? 'Locked' : 'Unlocked'} successfully`);
    } catch (error) {
      console.error('Error toggling lock:', error);
    }
  };

  return (
    <View style={styles.container}>
      <TouchableOpacity 
        style={[
          styles.button,
          { backgroundColor: isLocked ? '#ff4444' : '#44bb44' }
        ]} 
        onPress={toggleLock}
      >
        <Text style={styles.buttonText}>
          {isLocked ? 'Unlock' : 'Lock'}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  button: {
    padding: 15,
    borderRadius: 10,
    width: 200,
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
});
