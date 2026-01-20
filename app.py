"""
Flask Web Application for Wine Cultivar Classification
Uses pickle-saved sklearn SVM model

Author: Modupe - Masters in Bioinformatics
Course: Artificial Intelligence
"""

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Configuration
MODEL_PATH = 'model/wine_cultivar_model.pkl'

# Global variable for loaded model package
model_package = None


def load_model():
    """Load the trained model from pickle file"""
    global model_package
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_package = pickle.load(f)
        
        print("=" * 60)
        print("MODEL LOADED SUCCESSFULLY")
        print("=" * 60)
        print(f"Algorithm: {model_package.get('algorithm', 'SVM')}")
        print(f"Accuracy: {model_package.get('accuracy', 0)*100:.2f}%")
        print(f"Features: {model_package.get('n_features', len(model_package['feature_names']))}")
        print(f"Classes: {len(model_package['target_names'])}")
        print("=" * 60)
        
        return True
        
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return False
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False


# Load model at startup
model_loaded = load_model()


@app.route('/')
def home():
    """Render the home page"""
    if not model_loaded:
        return "Model not loaded. Please check server logs.", 500
    
    feature_names = model_package['feature_names']
    class_names = model_package['target_names']
    
    return render_template('index.html', 
                          feature_names=feature_names,
                          class_names=class_names)


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for wine classification
    
    Expected JSON:
    {
        "features": [13.2, 2.77, 2.51, 18.5, 96.0, 1.09, 0.52, 0.86, 1.69, 5.8, 0.48, 1.03, 415.0]
    }
    
    Returns:
    {
        "success": true,
        "prediction": 0,
        "class_name": "class_0",
        "confidence": 0.95
    }
    """
    try:
        # Check if model is loaded
        if not model_loaded or model_package is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        # Get input data
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "features" in request body'
            }), 400
        
        features = data['features']
        
        # Validate input length
        expected_features = len(model_package['feature_names'])
        if len(features) != expected_features:
            return jsonify({
                'success': False,
                'error': f'Expected {expected_features} features, got {len(features)}'
            }), 400
        
        # Validate numeric values
        try:
            features = [float(f) for f in features]
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': 'All features must be numeric values'
            }), 400
        
        # Prepare features for prediction
        features_array = np.array([features])
        
        # Scale features using saved scaler
        scaler = model_package['scaler']
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        model = model_package['model']
        prediction = model.predict(features_scaled)[0]
        
        # Get prediction probabilities if available
        try:
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = float(probabilities[prediction])
            all_probabilities = [float(p) for p in probabilities]
        except AttributeError:
            # Model doesn't support predict_proba
            confidence = 1.0
            all_probabilities = None
        
        # Prepare response
        response = {
            'success': True,
            'prediction': int(prediction),
            'class_name': model_package['target_names'][prediction],
            'confidence': confidence
        }
        
        if all_probabilities:
            response['probabilities'] = all_probabilities
            response['all_classes'] = model_package['target_names']
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded,
        'algorithm': model_package.get('algorithm', 'Unknown') if model_package else 'None',
        'version': '1.0'
    })


@app.route('/api/info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if not model_loaded or model_package is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'algorithm': model_package.get('algorithm', 'SVM'),
        'kernel': model_package.get('kernel', 'RBF'),
        'accuracy': model_package.get('accuracy', 0),
        'num_features': len(model_package['feature_names']),
        'num_classes': len(model_package['target_names']),
        'feature_names': model_package['feature_names'],
        'class_names': model_package['target_names']
    })


@app.route('/api/example', methods=['GET'])
def example_prediction():
    """
    Example prediction with sample data
    """
    if not model_loaded or model_package is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    # Example wine features (typical Class 0 sample)
    example_features = [
        13.2,    # Alcohol
        2.77,    # Malic Acid
        2.51,    # Ash
        18.5,    # Alcalinity of Ash
        96.0,    # Magnesium
        1.09,    # Total Phenols
        0.52,    # Flavanoids
        0.86,    # Nonflavanoid Phenols
        1.69,    # Proanthocyanins
        5.8,     # Color Intensity
        0.48,    # Hue
        1.03,    # OD280/OD315
        415.0    # Proline
    ]
    
    try:
        # Scale and predict
        features_array = np.array([example_features])
        features_scaled = model_package['scaler'].transform(features_array)
        prediction = model_package['model'].predict(features_scaled)[0]
        
        try:
            probabilities = model_package['model'].predict_proba(features_scaled)[0]
            all_probs = [float(p) for p in probabilities]
        except AttributeError:
            all_probs = None
        
        return jsonify({
            'success': True,
            'example_features': example_features,
            'feature_names': model_package['feature_names'],
            'prediction': int(prediction),
            'class_name': model_package['target_names'][prediction],
            'probabilities': all_probs,
            'note': 'This is an example prediction using typical wine features'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # For local development
    import os
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "=" * 60)
    print("🍷 WINE CULTIVAR CLASSIFICATION API")
    print("=" * 60)
    
    if model_loaded:
        print("\n✓ Model Status: LOADED")
        print(f"✓ Algorithm: {model_package.get('algorithm', 'SVM')}")
        print(f"✓ Accuracy: {model_package.get('accuracy', 0)*100:.2f}%")
    else:
        print("\n✗ Model Status: NOT LOADED")
        print("✗ Check model file path")
    
    print(f"\n🌐 Server starting on port {port}")
    print("\nEndpoints:")
    print(f"  - http://localhost:{port}/")
    print(f"  - http://localhost:{port}/predict")
    print(f"  - http://localhost:{port}/health")
    print(f"  - http://localhost:{port}/api/info")
    print(f"  - http://localhost:{port}/api/example")
    print("\n" + "=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=port)