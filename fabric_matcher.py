import os
import sqlite3
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.spatial.distance import cosine
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MobileNetV2 model with explicit input shape
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
model.trainable = False

# Configuration
DB_PATH = 'database.db'
FABRIC_DIR = 'fabrics'
IMG_SIZE = (224, 224)

def create_database():
    """Initialize SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fabrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT UNIQUE NOT NULL,
            feature_vector BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    logging.info("Database initialized.")

def preprocess_image(image):
    """Preprocess image for MobileNetV2."""
    try:
        image = image.resize(IMG_SIZE)
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        raise

def extract_features(image_path=None, image=None):
    """Extract feature vector from an image."""
    try:
        if image_path:
            image = Image.open(image_path).convert('RGB')
        elif image:
            image = image.convert('RGB')
        else:
            raise ValueError("Either image_path or image must be provided")
        
        img_array = preprocess_image(image)
        features = model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        raise

def store_features(image_path):
    """Store image path and feature vector in SQLite."""
    try:
        features = extract_features(image_path=image_path)
        features_blob = features.tobytes()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('INSERT OR REPLACE INTO fabrics (image_path, feature_vector) VALUES (?, ?)',
                      (image_path, features_blob))
        conn.commit()
        conn.close()
        logging.info(f"Stored features for {image_path}")
        return features
    except Exception as e:
        logging.error(f"Error storing features for {image_path}: {e}")
        raise

def populate_database():
    """Ingest images from FABRIC_DIR into the database."""
    if not os.path.exists(FABRIC_DIR):
        os.makedirs(FABRIC_DIR)
        logging.info(f"Created directory {FABRIC_DIR}. Please add fabric images.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT image_path FROM fabrics')
    existing_paths = set(row[0] for row in cursor.fetchall())
    conn.close()
    
    # Debug: Log all files in the directory
    files = os.listdir(FABRIC_DIR)
    logging.info(f"Found {len(files)} files in {FABRIC_DIR}: {files}")
    
    valid_extensions = ('.png', '.jpg', '.jpeg')
    for filename in files:
        # Debug: Log each file being checked
        logging.info(f"Checking file: {filename}, lowercase: {filename.lower()}")
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(FABRIC_DIR, filename)
            # Normalize path for consistency
            image_path = os.path.normpath(image_path)
            logging.info(f"Valid image found: {image_path}")
            if image_path not in existing_paths:
                try:
                    logging.info(f"Processing {image_path}")
                    store_features(image_path)
                except Exception as e:
                    logging.error(f"Failed to process {image_path}: {e}")
            else:
                logging.info(f"Skipping {image_path}, already in database")
        else:
            logging.info(f"Skipping {filename}, invalid extension")

def get_top_matches(input_features, top_n=5):
    """Find top-N similar images based on cosine similarity."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT image_path, feature_vector FROM fabrics')
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            logging.warning("No images in database")
            return []
        
        similarities = []
        for image_path, features_blob in results:
            stored_features = np.frombuffer(features_blob, dtype=np.float32)
            similarity = 1 - cosine(input_features, stored_features)
            similarities.append((image_path, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_matches = similarities[:top_n]
        
        # Check for exact match (similarity > 95%)
        if top_matches and top_matches[0][1] > 0.95:
            return [(top_matches[0][0], top_matches[0][1])]
        
        return top_matches
    except Exception as e:
        logging.error(f"Error retrieving matches: {e}")
        raise

if __name__ == '__main__':
    create_database()
    populate_database()