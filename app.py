from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = {
        'age': int(request.form['age']),
        'job': request.form['job'],
        'marital': request.form['marital'],
        'education': request.form['education'],
        'default': request.form['default'],
        'housing': request.form['housing'],
        'loan': request.form['loan'],
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([data])

    # Add dummy values for other required features
    for col in ['contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome',
                'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']:
        input_df[col] = 0  # or some default value

    # Preprocess input
    processed_input = preprocessor.transform(input_df)

    # Predict
    result = model.predict(processed_input)[0]

    return render_template('result.html', prediction='YES' if result == 1 else 'NO')

if __name__ == '__main__':
    app.run(debug=True)
