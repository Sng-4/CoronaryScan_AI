import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Concatenate
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import os

def build_model(input_shape=(224, 224, 1), learning_rate=1e-4):
    """
    Constructs the EfficientNetB0 model adapted for Grayscale X-rays.
    
    Architecture:
    1. Input: 224x224x1 (Grayscale, 0-255 values)
    2. Augmentation: Flip, Rotate, Contrast (Active during training only)
    3. Adapter: Concatenates 1 channel -> 3 channels (Fake RGB)
    4. Base: EfficientNetB0 (Pre-trained on ImageNet)
    5. Head: Custom Dense layers for binary classification (Stenosis vs Healthy)
    """
    
    # 1. Input Layer
    input_tensor = Input(shape=input_shape)

    # 2. Augmentation Layers (Built-in)
    # These layers automatically turn off during inference (prediction)
    x = RandomFlip("horizontal")(input_tensor)
    x = RandomRotation(0.2)(x) 
    x = RandomContrast(0.2)(x) 

    # 3. RGB Adapter (1 -> 3 channels)
    # EfficientNet expects 3 channels. We duplicate the grayscale channel.
    x = Concatenate()([x, x, x])

    # 4. Base Model (EfficientNetB0)
    # We use include_top=False to remove the 1000 ImageNet classes
    # EfficientNet handles 0-255 input values internally.
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=x)
    
    # Freeze the base model initially, but unfreeze the top 20 layers for adaptation
    base_model.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True
        
    # 5. Classification Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x) # Dropout to prevent overfitting
    output_tensor = Dense(1, activation='sigmoid')(x)

    # 6. Compile
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def load_trained_model(model_path):
    """
    Loads a saved Keras model (.h5) from disk.
    """
    if not os.path.exists(model_path):
        # If no model exists (e.g., first run), return None or raise error
        print(f"⚠️ Warning: Model file not found at {model_path}")
        return None
        
    print(f"Loading model from {model_path}...")
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None