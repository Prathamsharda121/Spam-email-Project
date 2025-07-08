from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Basic text check: non-empty, has letters
def is_valid_email_text(text):
    return bool(re.search(r'[a-zA-Z]', text)) and len(text.strip()) > 5

# Load model and vectorizer from file created in your notebook
try:
    model, vectorizer = joblib.load('spam_bundle.pkl')
    print("✅ Model and vectorizer loaded successfully.")
    print(f"✅ Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}")
except Exception as e:
    print(f"❌ Failed to load model/vectorizer: {e}")
    model = None
    vectorizer = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email']

    if not model or not vectorizer:
        return render_template('templates/index.html', prediction="❌ Model not loaded correctly.")

    if not is_valid_email_text(email_text):
        return render_template('templates/index.html', prediction="❌ Invalid input.")

    try:
        transformed = vectorizer.transform([email_text])
        print(f"🟡 Transformed input shape: {transformed.shape}")

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(transformed)[0]
            prediction = 1 if prob[1] >= 0.7 else 0
            confidence = prob[prediction] * 100
            result = f"{'✅ Not Spam' if prediction == 1 else '⚠️ Spam'} (Confidence: {confidence:.2f}%)"
        else:
            prediction = model.predict(transformed)[0]
            result = "✅ Not Spam" if prediction == 1 else "⚠️ Spam"

    except Exception as e:
        result = f"❌ Model error: {str(e)}"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
