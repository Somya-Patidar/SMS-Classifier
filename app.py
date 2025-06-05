# app.py

from flask import Flask, request, render_template
import pickle

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the SMS message from the form
    message = request.form['message']

    # Transform the message using the vectorizer
    transformed_message = vectorizer.transform([message])

    # Predict using the loaded model
    prediction = model.predict(transformed_message)

    result = "🚫 Spam" if prediction[0] == 1 else "✅ Not Spam"

    return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == '__main__':
    app.run(debug=True)
