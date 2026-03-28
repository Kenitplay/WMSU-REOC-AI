# import os
# import pickle
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # Suppress TensorFlow startup logs for a cleaner UI
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# def load_assets():
#     """Loads the trained model and preprocessing objects."""
#     try:
#         model = tf.keras.models.load_model('research_rnn_model.keras')
        
#         with open('tokenizer.pkl', 'rb') as handle:
#             tokenizer = pickle.load(handle)
            
#         with open('label_encoder.pkl', 'rb') as handle:
#             le = pickle.load(handle)
            
#         return model, tokenizer, le
#     except FileNotFoundError as e:
#         print(f"Error: Missing required file - {e}")
#         return None, None, None

# def main():
#     print("--- Research Review Type Predictor ---")
#     print("Loading AI model, please wait...")
    
#     model, tokenizer, le = load_assets()
    
#     if model is None:
#         return

#     max_len = 20 # Must match the max_len used during training

#     print("\nModel Loaded! Enter a Research Title to classify.")
#     print("Type 'quit' or 'exit' to stop.\n")

#     while True:
#         # Ask for user input
#         title = input("Enter Research Title: ").strip()

#         if title.lower() in ['quit', 'exit', '']:
#             break

#         # 1. Convert text to sequence
#         seq = tokenizer.texts_to_sequences([title])
        
#         # 2. Pad sequence to match model input shape
#         padded = pad_sequences(seq, maxlen=max_len)
        
#         # 3. Predict
#         prediction = model.predict(padded, verbose=0)
        
#         # 4. Get the class with the highest probability
#         class_idx = np.argmax(prediction)
#         confidence = prediction[0][class_idx] * 100
#         result_label = le.inverse_transform([class_idx])[0]

#         print(f"Result: **{result_label}** ({confidence:.2f}% confidence)")
#         print("-" * 30)

# if __name__ == "__main__":
#     main()

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify
from flask_cors import CORS

# Suppress TensorFlow startup logs for a cleaner UI
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global variables to store model and preprocessing objects
model = None
tokenizer = None
le = None
max_len = 20  # Must match the max_len used during training

def load_assets():
    """Loads the trained model and preprocessing objects."""
    global model, tokenizer, le
    
    try:
        model = tf.keras.models.load_model('research_rnn_model.keras')
        
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        with open('label_encoder.pkl', 'rb') as handle:
            le = pickle.load(handle)
            
        print("Model and preprocessing objects loaded successfully!")
        return True
        
    except FileNotFoundError as e:
        print(f"Error: Missing required file - {e}")
        return False
    except Exception as e:
        print(f"Error loading assets: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API status."""
    if model is None or tokenizer is None or le is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Model not loaded'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'message': 'API is ready'
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Predict research type from title."""
    global model, tokenizer, le
    
    # Check if model is loaded
    if model is None or tokenizer is None or le is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please ensure model files are available'
        }), 503
    
    # Get JSON data from request
    data = request.get_json()
    
    if not data:
        return jsonify({
            'error': 'No JSON data provided',
            'message': 'Please provide a title in the request body'
        }), 400
    
    # Extract title from request
    title = data.get('title', '').strip()
    
    if not title:
        return jsonify({
            'error': 'Missing title',
            'message': 'Please provide a "title" field in the JSON request'
        }), 400
    
    try:
        # 1. Convert text to sequence
        seq = tokenizer.texts_to_sequences([title])
        
        # 2. Pad sequence to match model input shape
        padded = pad_sequences(seq, maxlen=max_len)
        
        # 3. Predict
        prediction = model.predict(padded, verbose=0)
        
        # 4. Get the class with the highest probability
        class_idx = np.argmax(prediction)
        confidence = float(prediction[0][class_idx] * 100)
        result_label = le.inverse_transform([class_idx])[0]
        
        # Get all class probabilities
        all_probabilities = {
            label: float(prob * 100) 
            for label, prob in zip(le.classes_, prediction[0])
        }
        
        # Return prediction results
        return jsonify({
            'success': True,
            'title': title,
            'prediction': result_label,
            'confidence': round(confidence, 2),
            'all_probabilities': all_probabilities,
            'class_index': int(class_idx)
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict research types for multiple titles."""
    global model, tokenizer, le
    
    # Check if model is loaded
    if model is None or tokenizer is None or le is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please ensure model files are available'
        }), 503
    
    # Get JSON data from request
    data = request.get_json()
    
    if not data:
        return jsonify({
            'error': 'No JSON data provided',
            'message': 'Please provide a list of titles in the request body'
        }), 400
    
    # Extract titles from request
    titles = data.get('titles', [])
    
    if not titles or not isinstance(titles, list):
        return jsonify({
            'error': 'Missing or invalid titles',
            'message': 'Please provide a "titles" array in the JSON request'
        }), 400
    
    try:
        results = []
        
        for title in titles:
            title = title.strip()
            if not title:
                continue
                
            # Convert text to sequence
            seq = tokenizer.texts_to_sequences([title])
            
            # Pad sequence
            padded = pad_sequences(seq, maxlen=max_len)
            
            # Predict
            prediction = model.predict(padded, verbose=0)
            
            # Get prediction details
            class_idx = np.argmax(prediction)
            confidence = float(prediction[0][class_idx] * 100)
            result_label = le.inverse_transform([class_idx])[0]
            
            results.append({
                'title': title,
                'prediction': result_label,
                'confidence': round(confidence, 2),
                'class_index': int(class_idx)
            })
        
        return jsonify({
            'success': True,
            'total_predictions': len(results),
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    global model, tokenizer, le
    
    if model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 503
    
    return jsonify({
        'success': True,
        'model_loaded': True,
        'max_sequence_length': max_len,
        'available_classes': le.classes_.tolist(),
        'num_classes': len(le.classes_),
        'model_summary': str(model.summary())
    }), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Available endpoints: /health, /predict, /predict_batch, /model_info'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    # Load assets before starting the server
    print("Loading AI model, please wait...")
    if load_assets():
        print("\n" + "="*50)
        print("Research Review Type Predictor API")
        print("="*50)
        print("Available endpoints:")
        print("  GET  /health        - Health check")
        print("  POST /predict       - Predict single title")
        print("  POST /predict_batch - Predict multiple titles")
        print("  GET  /model_info    - Get model information")
        print("\nStarting Flask server...")
        print("Press Ctrl+C to stop the server")
        print("="*50 + "\n")
        
        # Run the Flask app
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Failed to load model. Please check if all required files exist.")
