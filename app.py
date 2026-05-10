from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load all trained models
models = {
    'ensemble': joblib.load('cancer_ensemble_model.pkl'),
    'logistic': joblib.load('cancer_logistic_model.pkl'),
    'gradient': joblib.load('cancer_gradient_boosting_model.pkl'),
    'forest':   joblib.load('cancer_random_forest_model.pkl'),
}

MODEL_LABELS = {
    'ensemble': 'Ensemble (Voting Classifier)',
    'logistic': 'Logistic Regression',
    'gradient': 'Gradient Boosting',
    'forest':   'Random Forest',
}

@app.route('/')
def home():
    return render_template("breast.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        radius     = float(request.form['mean_radius'])
        texture    = float(request.form['mean_texture'])
        perimeter  = float(request.form['mean_perimeter'])
        area       = float(request.form['mean_area'])
        smoothness = float(request.form['mean_smoothness'])
        model_key  = request.form.get('selected_model', 'ensemble')

        features = np.array([[radius, texture, perimeter, area, smoothness]])

        model      = models.get(model_key, models['ensemble'])
        prediction = model.predict(features)[0]
        prob       = model.predict_proba(features)[0][1] * 100

        if prediction == 1:
            result = "Malignant (Cancer Detected)"
            advice = "⚠️ High risk detected. Please consult a medical professional immediately."
            style  = "malignant"
        else:
            result = "Benign (No Cancer Detected)"
            advice = "✅ No cancer detected. Maintain healthy lifestyle and regular checkups."
            style  = "benign"

        return render_template(
            "breast.html",
            prediction_text=result,
            advice_text=advice,
            risk=f"{prob:.2f}",
            style=style,
            selected_model=model_key,
            model_label=MODEL_LABELS.get(model_key, model_key),
        )

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)