from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('best_rf_model.pkl')

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for About page
@app.route('/about')
def about():
    return render_template('about.html')

# Define route for Contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Define route for handling form submissions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        english = float(request.form['English'])
        science = float(request.form['Science'])
        maths = float(request.form['Maths'])
        history = float(request.form['History'])
        geography = float(request.form['Geograpgy'])
    
        # Prepare input features
        features = np.array([[english, science, maths, history, geography]])
    
        # Make prediction
        prediction = model.predict(features)
    
        # Convert prediction to string if it's categorical
        predicted_course = prediction[0]
    
        return render_template('index.html', prediction=f'Predicted Course: {predicted_course}')
    except Exception as e:
        return str(e)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5005)  # Use port 5001 if port 5000 is in use
