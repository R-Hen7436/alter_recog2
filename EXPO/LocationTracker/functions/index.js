/**
 * Import function triggers from their respective submodules:
 *
 * const {onCall} = require("firebase-functions/v2/https");
 * const {onDocumentWritten} = require("firebase-functions/v2/firestore");
 *
 * See a full list of supported triggers at https://firebase.google.com/docs/functions
 */

const {onRequest} = require("firebase-functions/v2/https");
const logger = require("firebase-functions/logger");
const functions = require("firebase-functions");
const admin = require("firebase-admin");
const fetch = require("node-fetch");

// Create and deploy your first functions
// https://firebase.google.com/docs/functions/get-started

// exports.helloWorld = onRequest((request, response) => {
//   logger.info("Hello logs!", {structuredData: true});
//   response.send("Hello from Firebase!");
// });

admin.initializeApp();

exports.sendLockNotification = functions.database
    .ref("/IOTs/Smartlock")
    .onUpdate(async (change, context) => {
        try {
            const newValue = change.after.val();
            console.log("Smartlock value changed to:", newValue);
            
            if (newValue === 0) {
                // Get the stored Expo push token from your database
                const tokenSnapshot = await admin.database()
                    .ref("/userPushTokens")
                    .once("value");
                
                const tokens = [];
                tokenSnapshot.forEach((child) => {
                    tokens.push(child.val().token);
                });

                console.log("Found tokens:", tokens);

                if (tokens.length === 0) {
                    console.log("No tokens found to send notifications to");
                    return;
                }

                // Construct the message
                const message = {
                    to: tokens,
                    sound: "default",
                    title: "Smart Lock Alert",
                    body: "Your smart lock has been deactivated!",
                    data: {type: "smartlock_alert"},
                    priority: "high"
                };

                // Send the notification through Expo's push notification service
                const response = await fetch("https://exp.host/--/api/v2/push/send", {
                    method: "POST",
                    headers: {
                        "Accept": "application/json",
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(message)
                });

                const result = await response.json();
                console.log("Notification send result:", result);
            }
        } catch (error) {
            console.error("Error sending notification:", error);
        }
    });
