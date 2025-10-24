"""
Flask Web Application for Zifty Player Classification System
Predicts player performance scores and assigns player classes
"""

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import json
import os

app = Flask(__name__)

# # Load the trained model
# MODEL_PATH = '../models/player_classification_model.pkl'
# METRICS_PATH = '../models/model_metrics.json'

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'player_classification_model.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'model_metrics.json')

print("Loading model...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(" Model loaded successfully")
except FileNotFoundError:
    print("ERROR: Model file not found. Please train the model first.")
    model = None

# Load model metrics
try:
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
    print(" Metrics loaded successfully")
except FileNotFoundError:
    print("WARNING: Metrics file not found")
    metrics = None

def assign_player_class(score):
    """Assign player class based on performance score - NEW BOUNDARIES"""
    if score >= 71:
        return 'A', 'Elite', '#FFD700', 'Top tier competitive player'
    elif score >= 51:
        return 'B', 'Advanced', '#C0C0C0', 'Highly skilled player'
    elif score >= 35:
        return 'C', 'Intermediate', '#CD7F32', 'Average skilled player'
    elif score >= 26:
        return 'D', 'Beginner', '#4682B4', 'Developing player'
    else:
        return 'E', 'Novice', '#708090', 'New player'

@app.route('/')
def index():
    """Home page with input form"""
    return render_template('index.html', metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        kd_ratio = float(request.form['kd_ratio'])
        legendary_streak = int(request.form['legendary_streak'])
        experience_level = int(request.form['experience_level'])
        daily_play_time = request.form['daily_play_time']
        codm_experience = request.form['codm_experience']
        player_name = request.form.get('player_name', 'Player')
        
        # Validate inputs
        if not (0.1 <= kd_ratio <= 10.0):
            return render_template('result.html', 
                                 error="K/D Ratio must be between 0.1 and 10.0")
        
        if not (0 <= legendary_streak <= 50):
            return render_template('result.html',
                                 error="Legendary Streak must be between 0 and 50")
        
        if not (1 <= experience_level <= 400):
            return render_template('result.html',
                                 error="Experience Level must be between 1 and 400")
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'mp_kd_ratio': [kd_ratio],
            'mp_legendary_streak': [legendary_streak],
            'experience_level': [experience_level],
            'daily_play_time': [daily_play_time],
            'codm_experience': [codm_experience]
        })
        
        # Make prediction
        if model is None:
            return render_template('result.html',
                                 error="Model not loaded. Please contact administrator.")
        
        predicted_score = model.predict(input_data)[0]
        predicted_score = round(max(0, min(100, predicted_score)), 1)  # Clip to 0-100
        
        # Get player class
        player_class, class_name, class_color, class_description = assign_player_class(predicted_score)
        
        # Prepare result data
        result = {
            'player_name': player_name,
            'score': predicted_score,
            'class': player_class,
            'class_name': class_name,
            'class_color': class_color,
            'class_description': class_description,
            'inputs': {
                'K/D Ratio': kd_ratio,
                'MP Legendary Seasons': legendary_streak,
                'Experience Level': experience_level,
                'Daily Play Time': daily_play_time,
                'CODM Experience': codm_experience
            }
        }
        
        return render_template('result.html', result=result, metrics=metrics)
        
    except ValueError as e:
        return render_template('result.html', 
                             error=f"Invalid input: {str(e)}")
    except Exception as e:
        return render_template('result.html',
                             error=f"Error making prediction: {str(e)}")

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html', metrics=metrics)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        input_data = pd.DataFrame({
            'mp_kd_ratio': [float(data['kd_ratio'])],
            'mp_legendary_streak': [int(data['legendary_streak'])],
            'experience_level': [int(data['experience_level'])],
            'daily_play_time': [data['daily_play_time']],
            'codm_experience': [data['codm_experience']]
        })
        
        predicted_score = model.predict(input_data)[0]
        predicted_score = round(max(0, min(100, predicted_score)), 1)
        
        player_class, class_name, class_color, class_description = assign_player_class(predicted_score)
        
        return jsonify({
            'success': True,
            'score': predicted_score,
            'class': player_class,
            'class_name': class_name,
            'class_description': class_description
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    print("\n" + "="*70)
    print("ZIFTY PLAYER CLASSIFICATION - WEB APPLICATION")
    print("="*70)
    if model:
        print("\n Model loaded and ready")
        if metrics:
            print(f" Model Accuracy: {metrics['classification_accuracy']*100:.2f}%")
            print(f" Best Model: {metrics['best_model']}")
    print("\n Starting web server...")
    print(" Open your browser and go to: http://127.0.0.1:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)