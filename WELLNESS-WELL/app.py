from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Enable CORS by setting headers manually
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# Load trained models for each disease
models = {
    "heart": joblib.load('heartpred.pkl'),
    "cancer": joblib.load('cancer.pred.pkl'),  # Fix the filename if necessary
    "parkinsons": joblib.load('parkisonspred.pkl'),
    "diabetes": joblib.load('diabetespred.pkl')
}

@app.route('/')
def home():
    return render_template('disease-selection.html')

@app.route('/predict/<disease>', methods=['POST'])
def predict(disease):
    try:
        # Get model based on disease type
        model = models.get(disease)
        if not model:
            return jsonify({'success': False, 'error': 'Invalid disease type'}), 400
        
        # Get data from request
        data = request.get_json(force=True)

        # Process data for each disease
        if disease == 'heart':
            input_data = [
                float(data['age']),
                float(data['sex']),
                float(data['chestPain']),
                float(data['cholesterol'])
            ]
        elif disease == 'cancer':
            input_data = [
                float(data['meanRadius']),
                float(data['meanTexture']),
                float(data['meanPerimeter']),
                float(data['meanArea']),
                float(data['meanSmoothness'])
            ]
        elif disease == 'parkinsons':
            input_data = [
                float(data['mdvpFoHz']),
                float(data['mdvpFhiHz']),
                float(data['mdvpFloHz']),
                float(data['mdvpJitter']),
                float(data['mdvpShimmer'])
            ]
        elif disease == 'diabetes':
            input_data = [
                float(data['pregnancies']),
                float(data['glucose']),
                float(data['bloodPressure']),
                float(data['skinThickness']),
                float(data['insulin']),
                float(data['bmi']),
                float(data['diabetesPedigreeFunction']),
                float(data['age'])
            ]
        else:
            return jsonify({'success': False, 'error': 'Invalid disease type'}), 400

        # Convert to numpy array and reshape
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data_as_numpy_array)
        
        # Define results for each disease
        results = {
            "heart": ["The person does not have Heart Disease", "The person has Heart Disease"],
            "cancer": ["The person does not have Cancer", "The person has Cancer"],
            "parkinsons": ["The person does not have Parkinson's", "The person has Parkinson's"],
            "diabetes": ["The person does not have Diabetes", "The person has Diabetes"]
        }
        
        # Return prediction result
        result_text = results[disease][prediction[0]]
        return jsonify({'success': True, 'result': result_text})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
