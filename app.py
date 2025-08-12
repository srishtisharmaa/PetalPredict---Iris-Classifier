from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load("iris_model.pkl")

# Mapping of prediction numbers to Iris species names
species_map = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica"
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])

            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = model.predict(features)[0]
            species_name = species_map.get(prediction, "Unknown")

            return render_template("predict.html", prediction_text=f"🌸 Predicted Iris species: {species_name}")
        except Exception as e:
            return render_template("predict.html", prediction_text="⚠️ Invalid input. Please enter valid numbers.")
    return render_template("predict.html")

if __name__ == '__main__':
    app.run(debug=True)
