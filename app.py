from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('diabetes_prediction_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_features = [
        float(data['pregnancies']),
        float(data['glucose']),
        float(data['bloodPressure']),
        float(data['skinThickness']),
        float(data['insulin']),
        float(data['bmi']),
        float(data['diabetesPedigreeFunction']),
        float(data['age'])
    ]
    input_array = np.array([input_features])
    prediction = model.predict(input_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return jsonify(prediction=int(predicted_class))

if __name__ == '__main__':
    app.run(debug=True)
