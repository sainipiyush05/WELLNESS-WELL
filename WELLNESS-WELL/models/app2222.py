from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)

# Instead of using flask-cors, you can add CORS headers directly
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# Load your trained model
model = joblib.load('heartpred.pkl')

@app.route('/')
def home():
    return render_template('disease-selection.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict/<disease>', methods=['POST'])
def predict(disease):
    try:
        # Get data from request
        data = request.get_json(force=True)
        
        if disease == 'heart':
            # Extract heart disease features
            input_data = [
                float(data['age']),
                float(data['sex']),
                float(data['chestPain']),
                float(data['cholesterol'])
            ]
            model = joblib.load('heartpred.pkl')
        
        elif disease == 'cancer':
            # Extract cancer prediction features
            input_data = [
                float(data['mean_radius']),
                float(data['mean_perimeter']),
                float(data['mean_area']),
                float(data['fractal_dimension_error'])
            ]
            model = joblib.load('cancer.pred.pkl')  # Load your trained cancer model
        
        elif disease == 'parkinsons':
            # Extract Parkinson's prediction features
            input_data = [
                float(data['fhi']),
                float(data['flo']),
                float(data['jitter'])
            ]
            model = joblib.load('parkinsonpred.pkl')  # Load your trained Parkinson's model
        
        elif disease == 'diabetes':
            # Extract diabetes prediction features
            input_data = [
                float(data['BMI']),
                float(data['Age']),
                float(data['glucose']),
                float(data['blood_pressure'])
            ]
            model = joblib.load('diabetespred.pkl')  # Load your trained diabetes model

        # Convert to numpy array and reshape
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data_as_numpy_array)
        
        # Return prediction result based on the disease
        if disease == 'heart':
            result_message = 'The Person has Heart Disease' if prediction[0] == 1 else 'The Person does not have Heart Disease'
        elif disease == 'cancer':
            result_message = 'The Person has Cancer' if prediction[0] == 1 else 'The Person does not have Cancer'
        elif disease == 'parkinsons':
            result_message = 'The Person has Parkinson\'s Disease' if prediction[0] == 1 else 'The Person does not have Parkinson\'s Disease'
        elif disease == 'diabetes':
            result_message = 'The Person has Diabetes' if prediction[0] == 1 else 'The Person does not have Diabetes'
        
        return jsonify({
            'success': True,
            'result': result_message
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    app.run(debug=True)