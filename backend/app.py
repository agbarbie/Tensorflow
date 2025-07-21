from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os

# Set up the Flask app
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Serve the frontend (index.html)
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'data' not in data:
            return jsonify({'error': 'Missing input data'}), 400

        input_data = np.array(data['data']).reshape(1, -1)
        prediction = model.predict(input_data)

        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        classes = ['setosa', 'versicolor', 'virginica']  # Replace with your actual classes
        predicted_class = classes[predicted_index]

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Optional performance metrics route
@app.route('/performance')
def performance():
    return jsonify({
        "test_accuracy": 0.98,
        "test_loss": 0.12,
        "total_parameters": 12345,
        "per_class_accuracy": {
            "setosa": 0.99,
            "versicolor": 0.97,
            "virginica": 0.98
        }
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
