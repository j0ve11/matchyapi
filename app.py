from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import logging
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from product_db import product_db
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Configure CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Allow all origins

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Upload folder configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model
model_path = os.path.join(os.getcwd(), 'skin_tone_model.h5')
try:
    model = load_model(model_path)
    logging.info("Skin tone model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise e


# Define skin tone classes
skin_tone_classes = ['dark', 'light', 'lighten', 'mid dark', 'mid light', 'mid-dark', 'mid-light']

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to classify skin tone
def classify_skin_tone(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]

        if 0 <= predicted_class_index < len(skin_tone_classes):
            return skin_tone_classes[predicted_class_index]
        else:
            logging.error(f"Predicted class index {predicted_class_index} is out of bounds.")
            return None
    except Exception as e:
        logging.error(f"Error in classify_skin_tone: {e}")
        return None

# Route for image classification
@app.route('/api/classify', methods=['POST'])
def classify_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded.'}), 400
        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected.'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg.'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        skin_tone = classify_skin_tone(file_path)
        os.remove(file_path)

        if skin_tone:
            return jsonify({'skin_tone': skin_tone}), 200
        return jsonify({'error': 'Failed to classify skin tone.'}), 500
    except Exception as e:
        logging.error(f"Error in classify_image: {e}")
        return jsonify({'error': 'Internal server error.'}), 500

@app.route('/api/recommendations', methods=['POST'])
def recommendations():
    try:
        data = request.get_json()
        logging.info(f"Received data: {data}")
        if not data or 'brand' not in data or 'skin_tone' not in data:
            return jsonify({'error': 'Invalid input. "brand" and "skin_tone" are required fields.'}), 400

        brand = data['brand']
        skin_tone = data['skin_tone']

        logging.info(f"Requested brand: {brand}, skin tone: {skin_tone}")

        if brand not in product_db:
            logging.error(f"Brand '{brand}' not found.")
            return jsonify({'error': f'Brand "{brand}" not found in product database.'}), 404

        if skin_tone not in product_db[brand]:
            logging.error(f"Skin tone '{skin_tone}' not available for brand '{brand}'.")
            return jsonify({'error': f'Skin tone "{skin_tone}" not available for brand "{brand}".'}), 404

        recommendations = product_db[brand][skin_tone]
        return jsonify({'recommendations': recommendations}), 200

    except Exception as e:
        logging.exception("Error processing recommendations")
        return jsonify({'error': 'An internal error occurred.'}), 500
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
