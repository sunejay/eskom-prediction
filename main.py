import os
from flask import Flask, request, jsonify
import joblib
from prediction import predict_load_shedding


app = Flask(__name__)


# Load your trained model
model = joblib.load("model.pkl")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    date = data['date']
    time = data['time']
    load_block = data['load_block']
    
    # Your prediction logic here
    X = predict_load_shedding(date, time, load_block)
    prediction = model.predict(X)
    
    return jsonify({'predicted_stage': int(prediction[0])})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
