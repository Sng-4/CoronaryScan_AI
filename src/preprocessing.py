import numpy as np
from PIL import Image
import io

def preprocess_image(image_data):
    """
    Preprocesses raw image bytes for the EfficientNet model.
    
    Args:
        image_data (bytes): Raw image bytes from an API upload.
        
    Returns:
        np.array: A preprocessed array with shape (1, 224, 224, 3)
                  ready for model.predict().
    """
    # 1. Open the image
    # We use io.BytesIO because the API receives bytes, not a file path
    image = Image.open(io.BytesIO(image_data))
    
    # 2. Ensure Grayscale ('L')
    # Even if the upload is RGB, we convert to Grayscale first to match training
    if image.mode != 'L':
        image = image.convert('L')
        
    # 3. Resize to 224x224
    # This matches the input shape of EfficientNetB0
    image = image.resize((224, 224))
    
    # 4. Convert to NumPy Array
    # Result shape: (224, 224)
    # Values: 0-255 (Integers). Do NOT divide by 255.0 for EfficientNet.
    img_array = np.array(image)
    
    # 5. Add Channel Dimension
    # Result shape: (224, 224, 1)
    img_array = np.expand_dims(img_array, axis=-1)
    
    # 6. Add Batch Dimension
    # Result shape: (1, 224, 224, 1)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 7. RGB Adapter (The "Fake RGB" Trick)
    # Our model expects 3 channels, so we duplicate the grayscale layer 3 times
    # Result shape: (1, 224, 224, 3)
    img_array = np.concatenate([img_array, img_array, img_array], axis=-1)
    
    return img_array

def preprocess_from_path(file_path):
    """
    Helper for local testing or retraining scripts.
    Reads from a file path instead of bytes.
    """
    with open(file_path, "rb") as f:
        return preprocess_image(f.read())