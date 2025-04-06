/**
 * ImgBB Storage Integration
 * A completely free alternative to Firebase Storage that doesn't require payment details
 */

// ImgBB API key - you can get a free API key at https://api.imgbb.com/
const IMGBB_API_KEY = '2fd6dd451112e14b78d9795bed49504d'; // Replace with your own API key

/**
 * Upload an image to ImgBB
 * @param {string} uri - The image URI (can be base64 data URI or file URI)
 * @param {string} filename - Name to identify the image
 * @returns {Promise<{url: string, delete_url: string, thumbnail: string}>} Upload result
 */
export const uploadImage = async (uri, filename) => {
  try {
    console.log(`Starting ImgBB upload for: ${filename}, URI type: ${uri.substring(0, 30)}...`);
    
    // Validate input
    if (!uri) {
      throw new Error('No image URI provided');
    }
    
    if (!filename) {
      filename = `image_${new Date().getTime()}.jpg`;
      console.log(`No filename provided, using generated name: ${filename}`);
    }
    
    // Extract base64 data if it's a data URI, otherwise fetch and convert
    let base64Data;
    if (uri.startsWith('data:')) {
      console.log('Processing data URI...');
      base64Data = uri.split(',')[1];
      
      if (!base64Data || base64Data.length < 100) {
        throw new Error(`Invalid base64 data (length: ${base64Data ? base64Data.length : 0})`);
      }
      
      return await uploadBase64(base64Data, filename);
    } else {
      console.log('Processing file URI, fetching content...');
      try {
        const response = await fetch(uri);
        if (!response.ok) {
          throw new Error(`Network response was not ok: ${response.status} ${response.statusText}`);
        }
        
        const blob = await response.blob();
        if (!blob || blob.size === 0) {
          throw new Error('Empty blob received from fetch');
        }
        
        console.log(`Successfully fetched image, size: ${blob.size} bytes, type: ${blob.type}`);
        
        const reader = new FileReader();
        return new Promise((resolve, reject) => {
          reader.onload = () => {
            try {
              if (!reader.result) {
                reject(new Error('FileReader returned empty result'));
                return;
              }
              
              const base64String = reader.result.split(',')[1];
              if (!base64String || base64String.length < 100) {
                reject(new Error(`Invalid base64 data from FileReader (length: ${base64String ? base64String.length : 0})`));
                return;
              }
              
              uploadBase64(base64String, filename)
                .then(resolve)
                .catch(reject);
            } catch (readError) {
              reject(new Error(`Error processing FileReader result: ${readError.message}`));
            }
          };
          reader.onerror = () => reject(new Error('FileReader error'));
          reader.readAsDataURL(blob);
        });
      } catch (fetchError) {
        console.error("Error fetching image:", fetchError);
        throw new Error(`Network error while fetching image: ${fetchError.message}`);
      }
    }
  } catch (error) {
    console.error("Error uploading image to ImgBB:", error);
    throw error;
  }
};

/**
 * Upload base64 image data to ImgBB
 * @param {string} base64Data - Base64 encoded image data (without data URI prefix)
 * @param {string} filename - Name to identify the image
 * @returns {Promise<{url: string, delete_url: string, thumbnail: string}>} Upload result
 */
const uploadBase64 = async (base64Data, filename) => {
  try {
    // Prepare the form data
    const formData = new FormData();
    formData.append('key', IMGBB_API_KEY);
    formData.append('image', base64Data);
    formData.append('name', filename);
    
    // Send the request to ImgBB API
    const response = await fetch('https://api.imgbb.com/1/upload', {
      method: 'POST',
      body: formData,
    });
    
    // Parse the response
    const result = await response.json();
    
    if (!result.success) {
      throw new Error(result.error?.message || 'Unknown error from ImgBB');
    }
    
    console.log("Image uploaded successfully to ImgBB:", result.data.url);
    
    // Return the useful data
    return {
      id: result.data.id,
      url: result.data.url,
      delete_url: result.data.delete_url,
      thumbnail: result.data.thumb?.url || result.data.url,
      name: filename,
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    console.error("Error in ImgBB upload:", error);
    throw error;
  }
};

/**
 * Get image information (placeholder - ImgBB doesn't provide listing API in free tier)
 * @param {string} imageId - The image ID to retrieve (if known)
 * @returns {Promise<Object>} Image information
 */
export const getImageInfo = async (imageId) => {
  // ImgBB free API doesn't provide a way to get image info by ID
  // This is a placeholder for compatibility
  throw new Error("ImgBB free API doesn't support retrieving image info");
};

// Storage tracking - maintained in memory
// For a real app, you could store this in AsyncStorage for persistence
const uploadedImages = [];

/**
 * Track a successful upload for later listing
 * @param {Object} imageData - The image data returned from ImgBB
 */
export const trackUpload = (imageData) => {
  uploadedImages.push({
    ...imageData,
    timestamp: new Date().toISOString()
  });
};

/**
 * Get all tracked uploaded images
 * @returns {Array} Array of tracked uploaded images
 */
export const getUploadedImages = () => {
  return [...uploadedImages]; // Return a copy of the array
};

/**
 * Get information about uploaded images
 * @returns {Object} Info about uploaded images
 */
export const getUploadedImagesInfo = () => {
  return {
    totalFiles: uploadedImages.length,
    fileNames: uploadedImages.map(img => img.name)
  };
};

/**
 * List all tracked images with details
 * @returns {Array} Array of image details
 */
export const listImages = () => {
  return uploadedImages.map(img => ({
    name: img.name,
    url: img.url,
    thumbnail: img.thumbnail,
    timestamp: img.timestamp,
    delete_url: img.delete_url,
    fullPath: `imgbb/${img.id}`
  }));
}; 