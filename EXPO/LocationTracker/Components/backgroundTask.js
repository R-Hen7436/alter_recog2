import * as BackgroundFetch from 'expo-background-fetch';
import * as TaskManager from 'expo-task-manager';
import { ref, onValue } from 'firebase/database';
import { db } from './firebaseConfig';
import { sendLocalNotification } from './notificationService';

export const BACKGROUND_FETCH_TASK = 'background-fetch';

let previousValue = null; // Store previous value to detect changes

// Define the background task
TaskManager.defineTask(BACKGROUND_FETCH_TASK, async () => {
  try {
    console.log('Background fetch task running');
    
    // Check smartlock status
    const snapshot = await new Promise((resolve) => {
      const smartlockRef = ref(db, 'IOTs/Smartlock');
      onValue(smartlockRef, (snapshot) => {
        resolve(snapshot);
      }, { onlyOnce: true });
    });

    const currentValue = snapshot.val();
    
    // Only send notification if value changed from something else to 0
    if (previousValue !== null && previousValue !== 0 && currentValue === 0) {
      await sendLocalNotification(
        'Smart Lock Alert',
        'Your smart lock has been deactivated!'
      );
      previousValue = currentValue;
      return BackgroundFetch.BackgroundFetchResult.NewData;
    }
    
    previousValue = currentValue;
    return BackgroundFetch.BackgroundFetchResult.NoData;
  } catch (error) {
    console.error('Background fetch failed:', error);
    return BackgroundFetch.BackgroundFetchResult.Failed;
  }
});

// Register background fetch
export async function registerBackgroundFetch() {
  try {
    await BackgroundFetch.registerTaskAsync(BACKGROUND_FETCH_TASK, {
      minimumInterval: 60, // 1 minute (minimum allowed)
      stopOnTerminate: false,
      startOnBoot: true,
    });
    console.log('Background fetch registered');
  } catch (err) {
    console.error('Task Register failed:', err);
  }
}