from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your trained model
# Save it first using:
# joblib.dump(best_model, "best_model.pkl")
model = joblib.load("best_model.pkl")


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>ML Prediction App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 700px;
            margin: 50px auto;
            padding: 20px;
        }

        textarea {
            width: 100%;
            height: 150px;
            padding: 10px;
            font-size: 15px;
        }

        button {
            margin-top: 15px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }

        #result {
            margin-top: 20px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Machine Learning Prediction</h1>

    <p>
        Enter your model features as a JSON list.
        Example:
    </p>

    <pre>[25, 50000, 3, 4, 5]</pre>

    <textarea id="features" placeholder="Enter feature values here"></textarea>
    <br>

    <button onclick="makePrediction()">Predict</button>

    <div id="result"></div>

    <script>
        async function makePrediction() {
            const input = document.getElementById("features").value;
            const result = document.getElementById("result");

            try {
                const features = JSON.parse(input);

                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        features: features
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    result.innerHTML =
                        "Prediction: " + data.prediction;
                } else {
                    result.innerHTML =
                        "Error: " + data.error;
                }

            } catch (error) {
                result.innerHTML =
                    "Please enter valid JSON values.";
            }
        }
    </script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "Request must contain JSON data"
            }), 400

        features = data.get("features")

        if features is None:
            return jsonify({
                "error": "Missing 'features' in request"
            }), 400

        # Convert input into a 2D array
        input_data = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]

        response = {
            "prediction": str(prediction)
        }

        # Include prediction probability if the model supports it
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            response["probabilities"] = probabilities.tolist()

        return jsonify(response)

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "Application is running"
    })


if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )